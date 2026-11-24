use std::ffi::OsString;
use std::num::NonZeroUsize;
use std::path::PathBuf;

use clap::{CommandFactory, Parser, Subcommand, error::ErrorKind};

use super::terminal::OutputFormat;
use crate::application::{CacheMode, Query, SearchRequest};
use crate::model::ArtifactVerification;
use crate::ranking::{DEFAULT_SIMILARITY_THRESHOLD, SimilarityThreshold};
use crate::timing::TimingDestination;

#[derive(Debug, Parser)]
#[command(
    version,
    subcommand_negates_reqs = true,
    args_conflicts_with_subcommands = true,
    allow_missing_positional = true,
    about = "Rust-native visual grep for local folders, scripts, and AI agents"
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) subcommand: Option<Subcommands>,

    #[arg(
        long,
        global = true,
        help = "Run in-process without using the shared server"
    )]
    pub(crate) no_server: bool,

    #[arg(value_parser = parse_query, required_unless_present = "image", conflicts_with = "image", help = "Natural language description of what to find")]
    query: Option<String>,

    #[arg(
        long,
        value_name = "FILE",
        help = "Find images similar to FILE, excluding FILE itself"
    )]
    image: Option<PathBuf>,

    #[arg(required = true, help = "Directory to search recursively")]
    path: Option<PathBuf>,

    #[arg(
        short = 'n',
        long = "top",
        default_value = "5",
        help = "Number of results to return"
    )]
    top: NonZeroUsize,

    #[arg(short = 't', long = "threshold", default_value_t = DEFAULT_SIMILARITY_THRESHOLD, allow_negative_numbers = true, help = "Minimum raw CLIP cosine similarity from -1.0 to 1.0")]
    threshold: SimilarityThreshold,

    #[arg(
        long = "json",
        conflicts_with = "paths_only",
        help = "Output results as JSON"
    )]
    json: bool,

    #[arg(long = "paths-only", help = "Output only matching paths, one per line")]
    paths_only: bool,

    #[arg(
        short = '0',
        long = "null",
        conflicts_with = "json",
        help = "Output exact paths separated by NUL bytes"
    )]
    null: bool,

    #[arg(
        long = "reindex",
        conflicts_with = "no_cache",
        help = "Force re-embedding of all images"
    )]
    reindex: bool,

    #[arg(long = "no-cache", help = "Skip reading and writing the index cache")]
    no_cache: bool,

    #[arg(
        long = "index-path",
        value_name = "PATH",
        conflicts_with = "no_cache",
        help = "Store the index at PATH instead of inside the searched directory"
    )]
    index_path: Option<PathBuf>,

    #[arg(short = 'q', long = "quiet", help = "Suppress progress output")]
    quiet: bool,

    #[arg(
        long = "verify-models",
        help = "Fully re-hash each model artifact required by this search"
    )]
    verify_models: bool,

    #[arg(
        long = "timing",
        help = "Write one machine-readable phase timing report to stderr"
    )]
    timing: bool,

    #[arg(
        long = "timing-file",
        value_name = "PATH",
        requires = "timing",
        help = "Write the --timing report to PATH instead of stderr"
    )]
    timing_file: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Subcommands {
    /// Index a directory and prepare both models without searching
    Index {
        #[arg(long, value_name = "PATH", help = "Store the index at PATH")]
        index_path: Option<PathBuf>,
        #[arg(short = 'q', long = "quiet", help = "Suppress progress output")]
        quiet: bool,
        /// Directory to index recursively
        path: PathBuf,
    },
    /// Manage the shared socket server
    Server {
        #[command(subcommand)]
        action: ServerAction,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum ServerAction {
    /// Run the shared server in the foreground
    Run,
    /// Ask the shared server to finish active requests and shut down
    Stop,
}

pub(crate) struct Command {
    pub(crate) request: SearchRequest,
    pub(crate) output_format: OutputFormat,
    pub(crate) quiet: bool,
    pub(crate) timing_destination: Option<TimingDestination>,
}

impl Cli {
    pub(crate) fn parse_cli() -> Self {
        Self::parse_args(std::env::args_os().collect()).unwrap_or_else(|error| error.exit())
    }

    fn parse_args(mut args: Vec<OsString>) -> Result<Self, clap::Error> {
        // Clap otherwise treats every leading option as the start of a search.
        if args.get(1).is_some_and(|arg| arg == "--no-server")
            && args
                .get(2)
                .and_then(|arg| arg.to_str())
                .is_some_and(|name| Self::command().find_subcommand(name).is_some())
        {
            args.swap(1, 2);
        }
        let cli = Self::try_parse_from(args)?;
        if cli.no_server
            && matches!(
                cli.subcommand,
                Some(Subcommands::Server {
                    action: ServerAction::Run | ServerAction::Stop
                })
            )
        {
            return Err(Self::command().error(
                ErrorKind::ArgumentConflict,
                "--no-server cannot be used with server commands",
            ));
        }
        Ok(cli)
    }

    pub(crate) fn into_command(self) -> Result<Command, clap::Error> {
        let path = self.path.ok_or_else(|| {
            Self::command().error(
                ErrorKind::MissingRequiredArgument,
                "provide a search directory",
            )
        })?;
        let query = match (self.query, self.image) {
            (Some(text), None) => Query::Text(text),
            (None, Some(path)) => Query::Image(path),
            (None, None) => {
                return Err(Self::command().error(
                    ErrorKind::MissingRequiredArgument,
                    "provide a text query or --image FILE",
                ));
            }
            (Some(_), Some(_)) => {
                return Err(Self::command().error(
                    ErrorKind::ArgumentConflict,
                    "a text query cannot be combined with --image",
                ));
            }
        };
        let cache_mode = if self.no_cache {
            CacheMode::Disabled
        } else if self.reindex {
            CacheMode::Reindex
        } else {
            CacheMode::Use
        };
        let output_format = if self.null {
            OutputFormat::PathsNull
        } else if self.json {
            OutputFormat::Json
        } else if self.paths_only {
            OutputFormat::PathsOnly
        } else {
            OutputFormat::Text
        };

        Ok(Command {
            request: SearchRequest::new(
                query,
                path,
                self.top,
                self.threshold,
                cache_mode,
                self.index_path,
                if self.verify_models {
                    ArtifactVerification::Full
                } else {
                    ArtifactVerification::Fast
                },
            ),
            output_format,
            quiet: self.quiet,
            timing_destination: self
                .timing
                .then(|| TimingDestination::new(self.timing_file)),
        })
    }
}

fn parse_query(value: &str) -> Result<String, String> {
    if value.trim().is_empty() {
        Err("query must contain non-whitespace characters".to_owned())
    } else {
        Ok(value.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_run_takes_no_arguments() {
        let cli =
            Cli::parse_args(["visiongrep", "server", "run"].map(OsString::from).to_vec()).unwrap();
        assert!(matches!(
            cli.subcommand,
            Some(Subcommands::Server {
                action: ServerAction::Run
            })
        ));
        assert!(Cli::try_parse_from(["visiongrep", "server", "run", "photos"]).is_err());
        let args = ["visiongrep", "--no-server", "server", "run"];
        let error = Cli::parse_args(args.map(OsString::from).to_vec()).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
    }

    #[test]
    fn index_requires_a_directory_and_accepts_index_options() {
        let cli = Cli::try_parse_from([
            "visiongrep",
            "index",
            "--index-path",
            "photos.db",
            "--quiet",
            "photos",
        ])
        .unwrap();
        let Some(Subcommands::Index {
            path,
            index_path,
            quiet,
        }) = cli.subcommand
        else {
            panic!("expected the index subcommand");
        };
        assert_eq!(path, PathBuf::from("photos"));
        assert_eq!(index_path, Some(PathBuf::from("photos.db")));
        assert!(quiet);
        for args in [
            vec!["visiongrep", "index"],
            vec!["visiongrep", "index", "photos", "--top", "3"],
            vec!["visiongrep", "index", "photos", "--no-cache"],
        ] {
            assert!(Cli::try_parse_from(&args).is_err(), "accepted {args:?}");
        }
    }

    #[test]
    fn subcommand_name_can_still_be_a_literal_query() {
        let cli = Cli::try_parse_from(["visiongrep", "--top", "5", "server", "photos"]).unwrap();
        assert_eq!(cli.query.as_deref(), Some("server"));
        assert!(cli.subcommand.is_none());
        let cli = Cli::try_parse_from(["visiongrep", "--top", "5", "index", "photos"]).unwrap();
        assert_eq!(cli.query.as_deref(), Some("index"));
        assert!(cli.subcommand.is_none());
    }

    #[test]
    fn global_no_server_can_precede_or_follow_a_subcommand() {
        for args in [
            ["visiongrep", "--no-server", "index", "photos"],
            ["visiongrep", "index", "photos", "--no-server"],
        ] {
            let cli = Cli::parse_args(args.map(OsString::from).to_vec()).unwrap();
            assert!(cli.no_server);
            assert!(cli.subcommand.is_some());
        }
        for args in [
            ["visiongrep", "--no-server", "server", "stop"],
            ["visiongrep", "server", "stop", "--no-server"],
        ] {
            let error = Cli::parse_args(args.map(OsString::from).to_vec()).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
        }
    }

    #[test]
    fn image_query_accepts_options_before_or_after_the_search_path() {
        for args in [
            vec!["visiongrep", "--image", "reference.png", "photos"],
            vec!["visiongrep", "photos", "--image", "reference.png"],
            vec!["visiongrep", "--image", "reference.png", "--", "photos"],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert_eq!(cli.path, Some(PathBuf::from("photos")));
            assert_eq!(cli.image, Some(PathBuf::from("reference.png")));
            assert!(cli.query.is_none());
            assert!(cli.into_command().is_ok());
        }
    }

    #[test]
    fn text_query_keeps_its_original_positional_arguments() {
        let cli = Cli::try_parse_from(["visiongrep", "red bicycle", "photos"]).unwrap();
        assert_eq!(cli.query.as_deref(), Some("red bicycle"));
        assert_eq!(cli.path, Some(PathBuf::from("photos")));
        assert!(cli.image.is_none());
    }

    #[test]
    fn exactly_one_query_and_a_search_path_are_required() {
        for args in [
            vec!["visiongrep"],
            vec!["visiongrep", "photos"],
            vec!["visiongrep", "--image", "reference.png"],
            vec!["visiongrep", "robot", "photos", "--image", "reference.png"],
            vec!["visiongrep", "--image", "reference.png", "robot", "photos"],
            vec![
                "visiongrep",
                "--image",
                "reference.png",
                "photos",
                "--image",
                "other.png",
            ],
            vec!["visiongrep", "--image"],
        ] {
            assert!(Cli::try_parse_from(&args).is_err(), "accepted {args:?}");
        }
    }

    #[test]
    fn image_query_accepts_native_non_utf8_paths() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt;

        let image = OsString::from_vec(b"query-\xff.png".to_vec());
        let cli = Cli::try_parse_from([
            OsString::from("visiongrep"),
            OsString::from("--image"),
            image.clone(),
            OsString::from("photos"),
        ])
        .unwrap();
        assert_eq!(cli.image, Some(PathBuf::from(image)));
    }

    #[test]
    fn cli_accepts_a_negative_threshold() {
        let cli =
            Cli::try_parse_from(["visiongrep", "robot", "photos", "--threshold", "-1"]).unwrap();

        assert_eq!(cli.threshold.get(), -1.0);
    }

    #[test]
    fn model_verification_is_opt_in() {
        let default = Cli::try_parse_from(["visiongrep", "robot", "photos"]).unwrap();
        assert!(!default.verify_models);
        let explicit =
            Cli::try_parse_from(["visiongrep", "robot", "photos", "--verify-models"]).unwrap();
        assert!(explicit.verify_models);
        assert!(explicit.into_command().is_ok());
    }

    #[test]
    fn reindex_conflicts_with_no_cache() {
        assert!(
            Cli::try_parse_from(["visiongrep", "robot", "photos", "--reindex", "--no-cache",])
                .is_err()
        );
    }

    #[test]
    fn blank_query_is_rejected() {
        assert!(Cli::try_parse_from(["visiongrep", "   ", "photos"]).is_err());
    }

    #[test]
    fn null_output_is_accepted_with_paths_only() {
        let cli =
            Cli::try_parse_from(["visiongrep", "robot", "photos", "--paths-only", "-0"]).unwrap();

        assert!(matches!(
            cli.into_command().unwrap().output_format,
            OutputFormat::PathsNull
        ));
    }

    #[test]
    fn timing_file_requires_timing() {
        assert!(
            Cli::try_parse_from([
                "visiongrep",
                "robot",
                "photos",
                "--timing-file",
                "timing.json",
            ])
            .is_err()
        );
    }

    #[test]
    fn custom_index_conflicts_with_no_cache() {
        assert!(
            Cli::try_parse_from([
                "visiongrep",
                "robot",
                "photos",
                "--index-path",
                "index.db",
                "--no-cache",
            ])
            .is_err()
        );
    }
}
