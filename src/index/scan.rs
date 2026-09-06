use std::os::unix::ffi::OsStrExt;
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
/// Symbolic links are not followed. Results are sorted by exact native Unix path bytes to match
/// SQLite's BLOB ordering for incremental reconciliation.
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

    files.sort_by(|left, right| {
        left.relative_path
            .as_os_str()
            .as_bytes()
            .cmp(right.relative_path.as_os_str().as_bytes())
    });
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
    use std::io::ErrorKind;
    use std::time::Duration;

    use super::*;

    #[test]
    fn resolving_a_missing_root_preserves_the_path_and_io_error() {
        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing");

        let error = SearchRoot::resolve(&missing).unwrap_err();

        assert!(
            matches!(error, VisionGrepError::SearchPathResolve { path, source }
            if path == missing && source.kind() == ErrorKind::NotFound)
        );
    }

    #[test]
    fn discovery_reports_a_root_removed_after_resolution() {
        let directory = tempfile::tempdir().unwrap();
        let root = SearchRoot::resolve(directory.path()).unwrap();
        directory.close().unwrap();

        let error = discover_images(&root).unwrap_err();

        assert!(
            matches!(error, VisionGrepError::ImageMetadata { path, source }
            if path == root.filesystem_path() && source.kind() == ErrorKind::NotFound)
        );
    }

    #[cfg(unix)]
    #[test]
    fn discovery_reports_an_unreadable_nested_directory() {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempfile::tempdir().unwrap();
        let root = SearchRoot::resolve(directory.path()).unwrap();
        let nested = root.image_path(Path::new("nested"));
        fs::create_dir(&nested).unwrap();
        fs::write(nested.join("hidden.png"), []).unwrap();
        fs::write(root.image_path(Path::new("visible.png")), []).unwrap();
        let original_permissions = fs::metadata(&nested).unwrap().permissions();
        fs::set_permissions(&nested, fs::Permissions::from_mode(0o000)).unwrap();
        // Root and some filesystems bypass mode bits. Restore permissions before any assertion.
        let access = fs::read_dir(&nested);
        let result = discover_images(&root);
        fs::set_permissions(&nested, original_permissions).unwrap();
        if access.is_ok() {
            eprintln!(
                "skipping permission assertion: this environment can read mode-000 directories"
            );
            return;
        }
        assert_eq!(access.unwrap_err().kind(), ErrorKind::PermissionDenied);
        assert!(
            matches!(result.unwrap_err(), VisionGrepError::ImageMetadata { path, source }
            if path == nested && source.kind() == ErrorKind::PermissionDenied)
        );
    }

    #[test]
    fn discovery_rejects_modification_times_before_the_unix_epoch() {
        let directory = tempfile::tempdir().unwrap();
        let root = SearchRoot::resolve(directory.path()).unwrap();
        let path = directory.path().join("old.png");
        let file = fs::File::create(&path).unwrap();
        let modified = UNIX_EPOCH - Duration::from_secs(1);
        if let Err(error) = file.set_times(fs::FileTimes::new().set_modified(modified)) {
            if matches!(
                error.kind(),
                ErrorKind::Unsupported | ErrorKind::InvalidInput
            ) {
                eprintln!(
                    "skipping timestamp assertion: filesystem rejects pre-epoch times: {error}"
                );
                return;
            }
            panic!("failed to set modification time: {error}");
        }
        if file.metadata().unwrap().modified().unwrap() != modified {
            eprintln!("skipping timestamp assertion: filesystem does not preserve pre-epoch times");
            return;
        }

        assert!(matches!(
            discover_images(&root),
            Err(VisionGrepError::Io(_))
        ));
    }

    #[test]
    fn discovery_accepts_supported_extensions_case_insensitively() {
        let directory = tempfile::tempdir().unwrap();
        let supported = ["a.jpg", "b.JPEG", "c.PnG", "d.WeBp", "e.BMP"];
        for name in
            supported
                .into_iter()
                .chain(["notes.txt", "image.gif", "png", ".png", "a.png.bak"])
        {
            fs::write(directory.path().join(name), []).unwrap();
        }
        fs::create_dir(directory.path().join("directory.png")).unwrap();
        let root = SearchRoot::resolve(directory.path()).unwrap();

        let paths: Vec<_> = discover_images(&root)
            .unwrap()
            .into_iter()
            .map(|image| image.relative_path)
            .collect();

        assert_eq!(paths, supported.map(PathBuf::from));
    }

    #[cfg(unix)]
    #[test]
    fn discovery_ignores_file_directory_broken_and_cyclic_symlinks() {
        use std::os::unix::fs::symlink;

        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("outside.png"), []).unwrap();
        fs::write(directory.path().join("real.png"), []).unwrap();
        symlink(
            directory.path().join("real.png"),
            directory.path().join("alias.png"),
        )
        .unwrap();
        symlink(
            outside.path().join("outside.png"),
            directory.path().join("external.png"),
        )
        .unwrap();
        symlink(outside.path(), directory.path().join("external-directory")).unwrap();
        symlink(
            directory.path().join("missing.png"),
            directory.path().join("broken.png"),
        )
        .unwrap();
        symlink(directory.path(), directory.path().join("cycle")).unwrap();
        let root = SearchRoot::resolve(directory.path()).unwrap();

        let images = discover_images(&root).unwrap();

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].relative_path, Path::new("real.png"));
    }

    #[test]
    fn discovery_sorts_nested_paths_independently_of_creation_order() {
        let directory = tempfile::tempdir().unwrap();
        fs::create_dir(directory.path().join("nested")).unwrap();
        for name in ["z.png", "nested/b.png", "a.png", "nested/a.png"] {
            fs::write(directory.path().join(name), []).unwrap();
        }
        let root = SearchRoot::resolve(directory.path()).unwrap();

        let paths: Vec<_> = discover_images(&root)
            .unwrap()
            .into_iter()
            .map(|image| image.relative_path)
            .collect();

        assert_eq!(
            paths,
            ["a.png", "nested/a.png", "nested/b.png", "z.png"].map(PathBuf::from)
        );
    }

    #[test]
    fn discovery_snapshots_file_size_and_modification_time() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("image.png");
        fs::write(&path, b"image bytes").unwrap();
        let modified = UNIX_EPOCH + Duration::new(1_700_000_000, 123_456_789);
        fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_times(fs::FileTimes::new().set_modified(modified))
            .unwrap();
        // Compare the filesystem's stored timestamp, which may have coarser precision.
        let metadata = fs::metadata(&path).unwrap();
        let expected_ns = i64::try_from(
            metadata
                .modified()
                .unwrap()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        )
        .unwrap();
        let root = SearchRoot::resolve(directory.path()).unwrap();

        let images = discover_images(&root).unwrap();

        assert_eq!(images.len(), 1);
        assert_eq!(images[0].relative_path, Path::new("image.png"));
        assert_eq!(images[0].size, 11);
        assert_eq!(images[0].mtime_ns, expected_ns);
    }

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
