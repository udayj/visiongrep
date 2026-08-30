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
    PathsNull,
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
        self.write_results_to(&mut output, results)
    }

    fn write_results_to(
        &self,
        output: &mut impl Write,
        results: &[SearchResult],
    ) -> Result<(), VisionGrepError> {
        match self.output_format {
            OutputFormat::Json => {
                if let Some(result) = results.iter().find(|result| result.path.to_str().is_none()) {
                    return Err(VisionGrepError::NonUtf8JsonPath {
                        path: result.path.clone(),
                    });
                }
                serde_json::to_writer_pretty(&mut *output, results)
                    .map_err(|source| VisionGrepError::JsonOutput { source })?;
                writeln!(output)?;
            }
            OutputFormat::PathsOnly => {
                for result in results {
                    write_path(output, &result.path)?;
                    writeln!(output)?;
                }
            }
            OutputFormat::PathsNull => {
                for result in results {
                    write_path(output, &result.path)?;
                    output.write_all(&[0])?;
                }
            }
            OutputFormat::Text => {
                if !self.quiet {
                    writeln!(output, "score\tpath")?;
                }
                for result in results {
                    write!(output, "{:.3}\t", result.score)?;
                    write_escaped_path(output, &result.path)?;
                    writeln!(output)?;
                }
            }
        }

        output.flush()?;
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

fn write_escaped_path(output: &mut impl Write, path: &Path) -> io::Result<()> {
    if let Some(path) = path.to_str() {
        for character in path.chars() {
            match character {
                '\\' => output.write_all(b"\\\\")?,
                '\t' => output.write_all(b"\\t")?,
                '\n' => output.write_all(b"\\n")?,
                '\r' => output.write_all(b"\\r")?,
                character if character.is_control() => {
                    write!(output, "{}", character.escape_default())?;
                }
                character => write!(output, "{character}")?,
            }
        }
        return Ok(());
    }

    for byte in path.as_os_str().as_bytes() {
        match byte {
            b'\\' => output.write_all(b"\\\\")?,
            b' '..=b'~' => output.write_all(std::slice::from_ref(byte))?,
            byte => write!(output, "\\x{byte:02x}")?,
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;
    use std::path::PathBuf;

    use super::*;

    fn result(path: PathBuf) -> SearchResult {
        SearchResult { score: 0.5, path }
    }

    #[test]
    fn text_output_escapes_record_separators() {
        let terminal = Terminal::new(OutputFormat::Text, true);
        let mut output = Vec::new();

        terminal
            .write_results_to(&mut output, &[result(PathBuf::from("a\tb\nc.jpg"))])
            .unwrap();

        assert_eq!(output, b"0.500\ta\\tb\\nc.jpg\n");
    }

    #[test]
    fn null_output_preserves_exact_non_utf8_paths() {
        let terminal = Terminal::new(OutputFormat::PathsNull, true);
        let path = PathBuf::from(OsString::from_vec(vec![b'a', 0xff, b'.', b'j']));
        let mut output = Vec::new();

        terminal
            .write_results_to(&mut output, &[result(path)])
            .unwrap();

        assert_eq!(output, [b'a', 0xff, b'.', b'j', 0]);
    }

    #[test]
    fn json_rejects_non_utf8_paths_explicitly() {
        let terminal = Terminal::new(OutputFormat::Json, true);
        let path = PathBuf::from(OsString::from_vec(vec![0xff]));
        let mut output = Vec::new();

        let error = terminal
            .write_results_to(&mut output, &[result(path)])
            .unwrap_err();

        assert!(matches!(error, VisionGrepError::NonUtf8JsonPath { .. }));
    }

    #[test]
    fn broken_pipe_is_reported_as_normal_termination() {
        struct BrokenPipeWriter;

        impl Write for BrokenPipeWriter {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                Ok(bytes.len())
            }

            fn flush(&mut self) -> io::Result<()> {
                Err(io::Error::from(io::ErrorKind::BrokenPipe))
            }
        }

        let terminal = Terminal::new(OutputFormat::PathsOnly, true);
        let error = terminal
            .write_results_to(&mut BrokenPipeWriter, &[result(PathBuf::from("image.jpg"))])
            .unwrap_err();

        assert!(error.is_broken_pipe());
    }
}
