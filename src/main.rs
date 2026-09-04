#![forbid(unsafe_code)]

mod application;
mod cli;
mod embedding;
mod error;
mod index;
mod model;
mod pillow_resize;
mod ranking;
mod timing;

use clap::Parser;

use crate::application::search;
use crate::cli::{Cli, Terminal};
use crate::error::VisionGrepError;
use crate::timing::{Phase, TimingRecorder};

fn main() {
    let exit_status = match run() {
        Ok(status) => status,
        Err(error) => {
            eprintln!("visiongrep: {error}");
            ExitStatus::OperationalError
        }
    };

    std::process::exit(exit_status.code());
}

fn run() -> Result<ExitStatus, VisionGrepError> {
    let command = Cli::parse().into_command();
    let mut timing = TimingRecorder::new(
        command.timing_destination.is_some(),
        crate::model::timing_metadata(),
    );
    let mut terminal = Terminal::new(command.output_format, command.quiet);
    let results = search(
        &command.request,
        &mut |event| terminal.handle_event(event),
        &mut timing,
    )?;
    let status = ExitStatus::from_has_matches(!results.is_empty());

    let output_started = timing.start();
    let output = terminal.write_results(&results);
    timing.record(Phase::OutputSerialization, output_started);
    timing.finish();
    if let Some(destination) = &command.timing_destination {
        timing.write(destination)?;
    }

    preserve_status_on_broken_pipe(status, output)
}

/// A downstream reader closing early is successful pipeline behavior, but it must not change
/// whether the completed search found a match.
fn preserve_status_on_broken_pipe(
    status: ExitStatus,
    output: Result<(), VisionGrepError>,
) -> Result<ExitStatus, VisionGrepError> {
    match output {
        Ok(()) => Ok(status),
        Err(error) if error.is_broken_pipe() => Ok(status),
        Err(error) => Err(error),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExitStatus {
    Found,
    NoMatches,
    OperationalError,
}

impl ExitStatus {
    fn from_has_matches(has_matches: bool) -> Self {
        if has_matches {
            Self::Found
        } else {
            Self::NoMatches
        }
    }

    fn code(self) -> i32 {
        match self {
            Self::Found => 0,
            Self::NoMatches => 1,
            Self::OperationalError => 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    fn broken_pipe() -> Result<(), VisionGrepError> {
        Err(VisionGrepError::Io(io::Error::from(
            io::ErrorKind::BrokenPipe,
        )))
    }

    #[test]
    fn broken_pipe_preserves_no_match_status() {
        assert_eq!(
            preserve_status_on_broken_pipe(ExitStatus::NoMatches, broken_pipe()).unwrap(),
            ExitStatus::NoMatches
        );
    }

    #[test]
    fn broken_pipe_preserves_found_status() {
        assert_eq!(
            preserve_status_on_broken_pipe(ExitStatus::Found, broken_pipe()).unwrap(),
            ExitStatus::Found
        );
    }

    #[test]
    fn other_output_errors_are_not_suppressed() {
        let error = VisionGrepError::Io(io::Error::other("injected output failure"));

        assert!(preserve_status_on_broken_pipe(ExitStatus::Found, Err(error)).is_err());
    }
}
