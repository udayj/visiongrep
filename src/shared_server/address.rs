use std::fs::{self, File};
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::VisionGrepError;

pub(super) fn socket_path(executable: &Path) -> Result<PathBuf, VisionGrepError> {
    let metadata = fs::metadata(executable)?;
    let mut hash = Sha256::new();
    hash.update(executable.as_os_str().as_bytes());
    hash.update(metadata.len().to_le_bytes());
    hash.update(metadata.mtime().to_le_bytes());
    hash.update(metadata.mtime_nsec().to_le_bytes());
    let digest = hash.finalize();
    let name = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(crate::model::cache_dir()?
        .join("run")
        .join(format!("{name}.sock")))
}

pub(super) struct SocketGuard {
    path: PathBuf,
    directory: File,
    metadata: fs::Metadata,
}

impl Drop for SocketGuard {
    fn drop(&mut self) {
        // A new server may already have replaced the socket while this server drained its requests.
        if self.directory.lock().is_ok()
            && let Ok(current) = fs::symlink_metadata(&self.path)
            && current.dev() == self.metadata.dev()
            && current.ino() == self.metadata.ino()
            && current.ctime() == self.metadata.ctime()
            && current.ctime_nsec() == self.metadata.ctime_nsec()
        {
            let _ = fs::remove_file(&self.path);
        }
    }
}

pub(super) fn bind(path: &Path) -> io::Result<Option<(UnixListener, SocketGuard)>> {
    let directory = path
        .parent()
        .ok_or_else(|| io::Error::other("socket path has no parent"))?;
    fs::DirBuilder::new()
        .recursive(true)
        .mode(0o700)
        .create(directory)?;
    fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
    let directory = File::open(directory)?;
    // Serialize stale-socket replacement without creating another lock file.
    directory.lock()?;
    let listener = match UnixListener::bind(path) {
        Ok(listener) => listener,
        Err(error) if error.kind() == io::ErrorKind::AddrInUse => match UnixStream::connect(path) {
            Ok(_) => return Ok(None),
            Err(error)
                if error.kind() == io::ErrorKind::ConnectionRefused
                    && fs::symlink_metadata(path)?.file_type().is_socket() =>
            {
                fs::remove_file(path)?;
                UnixListener::bind(path)?
            }
            Err(error) => return Err(error),
        },
        Err(error) => return Err(error),
    };
    let metadata = fs::symlink_metadata(path)?;
    directory.unlock()?;
    Ok(Some((
        listener,
        SocketGuard {
            path: path.to_owned(),
            directory,
            metadata,
        },
    )))
}
