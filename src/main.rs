#![forbid(unsafe_code)]

mod application;
mod cli;
mod embedding;
mod error;
mod index;
mod model;
mod ranking;

use clap::Parser;

use crate::application::search;
use crate::cli::{Cli, Terminal};
use crate::error::VisionGrepError;

fn main() {
    let exit_status = match run() {
        Ok(status) => status,
        Err(error) if error.is_broken_pipe() => ExitStatus::Found,
        Err(error) => {
            eprintln!("visiongrep: {error}");
            ExitStatus::OperationalError
        }
    };

    std::process::exit(exit_status.code());
}

fn run() -> Result<ExitStatus, VisionGrepError> {
    let command = Cli::parse().into_command();
    let mut terminal = Terminal::new(command.output_format, command.quiet);
    let results = search(&command.request, &mut |event| terminal.handle_event(event))?;
    terminal.write_results(&results)?;

    Ok(ExitStatus::from_has_matches(!results.is_empty()))
}

#[derive(Debug, Clone, Copy)]
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
