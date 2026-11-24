use std::path::Path;

use super::terminal::{OutputFormat, Terminal};
use crate::application::prepare_index;
use crate::error::VisionGrepError;
use crate::model::{ArtifactVerification, Models};
use crate::shared_server;

/// Runs `visiongrep index`, reporting progress and the final image count on stderr.
pub(crate) fn index(
    path: &Path,
    index_path: Option<&Path>,
    quiet: bool,
    no_server: bool,
) -> Result<(), VisionGrepError> {
    let mut terminal = Terminal::new(OutputFormat::Text, quiet);
    let remote = if no_server {
        None
    } else {
        shared_server::index(path, index_path, |event| {
            terminal.handle_server_event(event)
        })?
    };
    let images = match remote {
        Some(images) => images,
        None => {
            let models = Models::new(ArtifactVerification::Fast);
            prepare_index(path, index_path, &models, &mut |event| {
                terminal.handle_event(event)
            })?
        }
    };
    if !quiet {
        eprintln!(
            "Indexed {} images in {}; searches are ready",
            images,
            path.display()
        );
    }
    Ok(())
}
