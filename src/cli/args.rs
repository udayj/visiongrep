use std::path::PathBuf;

use clap::Parser;

use super::terminal::OutputFormat;
use crate::application::{CacheMode, SearchRequest};
use crate::ranking::DEFAULT_SIMILARITY_THRESHOLD;

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Rust-native visual grep for local folders, scripts, and AI agents"
)]
pub(crate) struct Cli {
    #[arg(help = "Natural language description of what to find")]
    query: String,

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

    #[arg(long = "reindex", help = "Force re-embedding of all images")]
    reindex: bool,

    #[arg(long = "no-cache", help = "Skip reading and writing the index cache")]
    no_cache: bool,

    #[arg(short = 'q', long = "quiet", help = "Suppress progress output")]
    quiet: bool,
}

pub(crate) struct Command {
    pub(crate) request: SearchRequest,
    pub(crate) output_format: OutputFormat,
    pub(crate) quiet: bool,
}

impl Cli {
    pub(crate) fn into_command(self) -> Command {
        let cache_mode = if self.no_cache {
            CacheMode::Disabled
        } else if self.reindex {
            CacheMode::Reindex
        } else {
            CacheMode::Use
        };
        let output_format = if self.json {
            OutputFormat::Json
        } else if self.paths_only {
            OutputFormat::PathsOnly
        } else {
            OutputFormat::Text
        };

        Command {
            request: SearchRequest::new(
                self.query,
                self.path,
                self.top,
                self.threshold,
                cache_mode,
            ),
            output_format,
            quiet: self.quiet,
        }
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
    fn no_cache_takes_precedence_over_reindex() {
        let cli = Cli::try_parse_from(["visiongrep", "robot", "photos", "--reindex", "--no-cache"])
            .unwrap();

        assert_eq!(cli.into_command().request.cache_mode(), CacheMode::Disabled);
    }
}
