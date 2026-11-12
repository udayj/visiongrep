use std::path::Path;

use super::terminal::{OutputFormat, Terminal};
use crate::application::prepare_index;
use crate::error::VisionGrepError;

/// Runs `visiongrep index`, reporting progress and the final image count on stderr.
pub(crate) fn index(
    path: &Path,
    index_path: Option<&Path>,
    quiet: bool,
) -> Result<(), VisionGrepError> {
    let mut terminal = Terminal::new(OutputFormat::Text, quiet);
    let summary = prepare_index(path, index_path, &mut |event| terminal.handle_event(event))?;
    if !quiet {
        eprintln!(
            "Indexed {} images in {}; searches are ready",
            summary.images,
            path.display()
        );
    }
    Ok(())
}
