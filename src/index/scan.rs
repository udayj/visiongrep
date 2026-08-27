use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::error::VisionGrepError;

#[derive(Debug, Clone)]
pub(crate) struct ImageFile {
    pub(super) path: PathBuf,
    pub(super) mtime_ns: i64,
    pub(super) size: i64,
}

/// Recursively discovers supported images and snapshots metadata used for cache invalidation.
///
/// Symbolic links are not followed. Results are sorted by native path for deterministic indexing
/// and output behavior independent of directory iteration order.
pub(crate) fn discover_images(root: &Path) -> Result<Vec<ImageFile>, VisionGrepError> {
    let mut files = Vec::new();

    for entry in walkdir::WalkDir::new(root).follow_links(false) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                if let Some(path) = err.path() {
                    return Err(VisionGrepError::ImageMetadata {
                        path: path.to_owned(),
                        source: err.into_io_error().unwrap_or_else(|| {
                            std::io::Error::other("failed to read directory entry")
                        }),
                    });
                }
                return Err(VisionGrepError::Io(std::io::Error::other(
                    "failed to read directory entry",
                )));
            }
        };

        if !entry.file_type().is_file() || !is_supported_image(entry.path()) {
            continue;
        }

        let metadata = entry
            .metadata()
            .map_err(|source| VisionGrepError::ImageMetadata {
                path: entry.path().to_owned(),
                source: source.into(),
            })?;
        let modified = metadata
            .modified()
            .map_err(|source| VisionGrepError::ImageMetadata {
                path: entry.path().to_owned(),
                source,
            })?
            .duration_since(UNIX_EPOCH)
            .map_err(|source| {
                VisionGrepError::Io(std::io::Error::other(format!(
                    "image mtime is before Unix epoch: {source}"
                )))
            })?;
        let mtime_ns = i64::try_from(modified.as_nanos()).map_err(|_| {
            VisionGrepError::ImageTimestampOutOfRange {
                path: entry.path().to_owned(),
            }
        })?;
        let size =
            i64::try_from(metadata.len()).map_err(|_| VisionGrepError::ImageSizeOutOfRange {
                path: entry.path().to_owned(),
            })?;
        files.push(ImageFile {
            path: entry.path().to_owned(),
            mtime_ns,
            size,
        });
    }

    files.sort_by(|left, right| left.path.cmp(&right.path));
    Ok(files)
}

fn is_supported_image(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .map(|extension| {
            matches!(
                extension.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png" | "webp" | "bmp"
            )
        })
        .unwrap_or(false)
}
