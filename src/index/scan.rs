use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use crate::error::VisionGrepError;

#[derive(Debug)]
pub(crate) struct SearchRoot {
    filesystem_path: PathBuf,
    display_path: PathBuf,
}

impl SearchRoot {
    pub(crate) fn resolve(path: &Path) -> Result<Self, VisionGrepError> {
        let filesystem_path =
            path.canonicalize()
                .map_err(|source| VisionGrepError::SearchPathResolve {
                    path: path.to_owned(),
                    source,
                })?;
        Ok(Self {
            filesystem_path,
            display_path: path.to_owned(),
        })
    }

    pub(crate) fn filesystem_path(&self) -> &Path {
        &self.filesystem_path
    }

    pub(crate) fn display_path(&self) -> &Path {
        &self.display_path
    }

    pub(crate) fn image_path(&self, relative_path: &Path) -> PathBuf {
        self.filesystem_path.join(relative_path)
    }

    pub(crate) fn display_image_path(&self, relative_path: &Path) -> PathBuf {
        self.display_path.join(relative_path)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ImageFile {
    pub(super) relative_path: PathBuf,
    pub(super) mtime_ns: i64,
    pub(super) size: i64,
}

/// Recursively discovers supported images and snapshots metadata used for cache invalidation.
///
/// Symbolic links are not followed. Results are sorted by native path for deterministic indexing
/// and output behavior independent of directory iteration order.
pub(crate) fn discover_images(root: &SearchRoot) -> Result<Vec<ImageFile>, VisionGrepError> {
    let mut files = Vec::new();

    for entry in walkdir::WalkDir::new(root.filesystem_path()).follow_links(false) {
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
        let relative_path = entry
            .path()
            .strip_prefix(root.filesystem_path())
            .map_err(|_| VisionGrepError::ImageOutsideSearchRoot {
                path: entry.path().to_owned(),
                root: root.filesystem_path().to_owned(),
            })?
            .to_owned();
        files.push(ImageFile {
            relative_path,
            mtime_ns,
            size,
        });
    }

    files.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
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

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn discovery_stores_paths_relative_to_the_search_root() {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir(directory.path().join("nested")).unwrap();
        fs::write(directory.path().join("nested/image.jpg"), []).unwrap();
        let root = SearchRoot::resolve(&directory.path().join(".")).unwrap();

        let images = discover_images(&root).unwrap();

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].relative_path, Path::new("nested/image.jpg"));
        assert_eq!(
            root.image_path(&images[0].relative_path),
            directory
                .path()
                .canonicalize()
                .unwrap()
                .join("nested/image.jpg")
        );
    }
}
