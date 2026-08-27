use std::io::{self, Write};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use indicatif::{ProgressBar, ProgressStyle};

use crate::application::SearchEvent;
use crate::error::VisionGrepError;
use crate::index::IngestEvent;
use crate::model::ArtifactEvent;
use crate::ranking::SearchResult;

#[derive(Debug, Clone, Copy)]
pub(crate) enum OutputFormat {
    Text,
    Json,
    PathsOnly,
}

pub(crate) struct Terminal {
    output_format: OutputFormat,
    quiet: bool,
    progress: Option<ProgressBar>,
}

impl Terminal {
    pub(crate) fn new(output_format: OutputFormat, quiet: bool) -> Self {
        Self {
            output_format,
            quiet,
            progress: None,
        }
    }

    pub(crate) fn handle_event(&mut self, event: SearchEvent) {
        match event {
            SearchEvent::Index(event) => self.handle_ingest_event(event),
            SearchEvent::Artifact(event) => self.handle_artifact_event(event),
        }
    }

    /// Serializes results to stdout without lossy path conversion.
    ///
    /// Text and paths-only formats write native Unix path bytes. JSON remains fallible because JSON
    /// strings require Unicode.
    pub(crate) fn write_results(&self, results: &[SearchResult]) -> Result<(), VisionGrepError> {
        let stdout = io::stdout();
        let mut output = io::BufWriter::new(stdout.lock());

        match self.output_format {
            OutputFormat::Json => {
                serde_json::to_writer_pretty(&mut output, results)
                    .map_err(|source| VisionGrepError::JsonOutput { source })?;
                writeln!(output)?;
            }
            OutputFormat::PathsOnly => {
                for result in results {
                    write_path(&mut output, &result.path)?;
                    writeln!(output)?;
                }
            }
            OutputFormat::Text => {
                if !self.quiet {
                    writeln!(output, "score\tpath")?;
                }
                for result in results {
                    write!(output, "{:.3}\t", result.score)?;
                    write_path(&mut output, &result.path)?;
                    writeln!(output)?;
                }
            }
        }

        Ok(())
    }

    fn handle_ingest_event(&mut self, event: IngestEvent) {
        if self.quiet {
            return;
        }

        match event {
            IngestEvent::Started { total } => {
                eprintln!("Indexing {total} images...");
                self.progress = Some(ProgressBar::new(total));
            }
            IngestEvent::ImageProcessed => {
                if let Some(progress) = &self.progress {
                    progress.inc(1);
                }
            }
            IngestEvent::ImageSkipped(error) => eprintln!("warning: {error}"),
            IngestEvent::Finished => self.finish_progress(),
        }
    }

    fn handle_artifact_event(&mut self, event: ArtifactEvent) {
        if self.quiet {
            return;
        }

        match event {
            ArtifactEvent::DownloadStarted { artifact } => {
                eprintln!("Downloading {artifact}...");
            }
            ArtifactEvent::ContentLength { bytes } => {
                let progress = ProgressBar::new(bytes);
                if let Ok(style) = ProgressStyle::with_template(
                    "[{bar:40.cyan/blue}] {bytes}/{total_bytes} {bytes_per_sec} eta {eta}",
                ) {
                    progress.set_style(style);
                }
                self.progress = Some(progress);
            }
            ArtifactEvent::BytesRead { bytes } => {
                if let Some(progress) = &self.progress {
                    progress.inc(bytes);
                }
            }
            ArtifactEvent::DownloadFinished => self.finish_progress(),
        }
    }

    fn finish_progress(&mut self) {
        if let Some(progress) = self.progress.take() {
            progress.finish_and_clear();
        }
    }
}

fn write_path(output: &mut impl Write, path: &Path) -> io::Result<()> {
    output.write_all(path.as_os_str().as_bytes())
}
