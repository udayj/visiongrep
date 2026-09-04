use std::path::PathBuf;

use clap::{CommandFactory, Parser, error::ErrorKind};

use super::terminal::OutputFormat;
use crate::application::{ArtifactVerification, CacheMode, Query, SearchRequest};
use crate::ranking::DEFAULT_SIMILARITY_THRESHOLD;
use crate::timing::TimingDestination;

#[derive(Debug, Parser)]
#[command(
    version,
    allow_missing_positional = true,
    about = "Rust-native visual grep for local folders, scripts, and AI agents"
)]
pub(crate) struct Cli {
    #[arg(value_parser = parse_query, required_unless_present = "image", conflicts_with = "image", help = "Natural language description of what to find")]
    query: Option<String>,

    #[arg(
        long,
        value_name = "FILE",
        help = "Find images similar to FILE, excluding FILE itself"
    )]
    image: Option<PathBuf>,

    #[arg(help = "Directory to search recursively")]
    path: PathBuf,

    #[arg(short = 'n', long = "top", default_value_t = 5, value_parser = parse_top, help = "Number of results to return")]
    top: usize,

    #[arg(short = 't', long = "threshold", default_value_t = DEFAULT_SIMILARITY_THRESHOLD, allow_negative_numbers = true, value_parser = parse_threshold, help = "Minimum raw CLIP cosine similarity from -1.0 to 1.0")]
    threshold: f32,

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

pub(crate) struct Command {
    pub(crate) request: SearchRequest,
    pub(crate) output_format: OutputFormat,
    pub(crate) quiet: bool,
    pub(crate) timing_destination: Option<TimingDestination>,
}

impl Cli {
    pub(crate) fn into_command(self) -> Result<Command, clap::Error> {
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
                self.path,
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

fn parse_threshold(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|error| format!("invalid threshold: {error}"))?;
    if (-1.0..=1.0).contains(&parsed) {
        Ok(parsed)
    } else {
        Err("threshold must be between -1.0 and 1.0".to_owned())
    }
}

fn parse_top(value: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid result count: {error}"))?;
    if parsed == 0 {
        Err("top must be at least 1".to_owned())
    } else {
        Ok(parsed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_query_accepts_options_before_or_after_the_search_path() {
        for args in [
            vec!["visiongrep", "--image", "reference.png", "photos"],
            vec!["visiongrep", "photos", "--image", "reference.png"],
            vec!["visiongrep", "--image", "reference.png", "--", "photos"],
        ] {
            let cli = Cli::try_parse_from(args).unwrap();
            assert_eq!(cli.path, PathBuf::from("photos"));
            assert_eq!(cli.image, Some(PathBuf::from("reference.png")));
            assert!(cli.query.is_none());
            assert!(cli.into_command().is_ok());
        }
    }

    #[test]
    fn text_query_keeps_its_original_positional_arguments() {
        let cli = Cli::try_parse_from(["visiongrep", "red bicycle", "photos"]).unwrap();
        assert_eq!(cli.query.as_deref(), Some("red bicycle"));
        assert_eq!(cli.path, PathBuf::from("photos"));
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
    fn threshold_accepts_the_full_cosine_range() {
        assert_eq!(parse_threshold("-1").unwrap(), -1.0);
        assert_eq!(parse_threshold("1").unwrap(), 1.0);
        assert!(parse_threshold("-1.01").is_err());
        assert!(parse_threshold("1.01").is_err());
    }

    #[test]
    fn cli_accepts_a_negative_threshold() {
        let cli =
            Cli::try_parse_from(["visiongrep", "robot", "photos", "--threshold", "-1"]).unwrap();

        assert_eq!(cli.threshold, -1.0);
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
