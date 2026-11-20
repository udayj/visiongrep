use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::{DirBuilderExt, FileTypeExt, MetadataExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Condvar, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::terminal::Terminal;
use crate::application::{
    CacheMode, Query, SearchEvent, SearchRequest, prepare_index, resolve_root, search_with_models,
};
use crate::error::VisionGrepError;
use crate::index::{IndexLocation, IngestEvent, SearchRoot};
use crate::model::{ArtifactEvent, ArtifactVerification, Models, SessionLifetime};
use crate::ranking::SearchResult;
use crate::timing::TimingRecorder;

const MAX_REQUEST_BYTES: u64 = 1024 * 1024;
const MAX_CONNECTIONS: usize = 8;
const IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(2);

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
enum Request {
    Search {
        target: Target,
        query: SocketQuery,
        top: usize,
        threshold: f32,
        reindex: bool,
    },
    Index(Target),
    Stop,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Target {
    root: Vec<u8>,
    index: Vec<u8>,
}

impl Target {
    fn resolve(path: &Path, index: Option<&Path>) -> Result<(SearchRoot, Self), VisionGrepError> {
        let root = resolve_root(path)?;
        let index = IndexLocation::resolve(root.filesystem_path(), index)?;
        let target = Self {
            root: root.filesystem_path().as_os_str().as_bytes().to_vec(),
            index: index.path().as_os_str().as_bytes().to_vec(),
        };
        Ok((root, target))
    }

    fn into_paths(self) -> io::Result<(PathBuf, PathBuf)> {
        Ok((absolute_path(self.root)?, absolute_path(self.index)?))
    }
}

fn absolute_path(bytes: Vec<u8>) -> io::Result<PathBuf> {
    let path = PathBuf::from(OsString::from_vec(bytes));
    if path.is_absolute() && !path.as_os_str().as_bytes().contains(&0) {
        return Ok(path);
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "expected an absolute path",
    ))
}

#[derive(Serialize, Deserialize)]
enum SocketQuery {
    Text(String),
    Image(Vec<u8>),
}

#[derive(Serialize, Deserialize)]
enum Response {
    Event(Event),
    Results(Vec<ResultItem>),
    Indexed(usize),
    Stopped,
    Error(String),
}

#[derive(Serialize, Deserialize)]
struct ResultItem {
    path: Vec<u8>,
    score: f32,
}

#[derive(Serialize, Deserialize)]
pub(super) enum Event {
    WaitingForIndex,
    IndexStarted { total: u64 },
    ImageProcessed,
    ImageSkipped(String),
    IndexFinished,
    DownloadStarted { artifact: String },
    ContentLength { bytes: u64 },
    BytesRead { bytes: u64 },
    DownloadFinished,
}

impl From<SearchEvent> for Event {
    fn from(event: SearchEvent) -> Self {
        match event {
            SearchEvent::WaitingForIndex => Self::WaitingForIndex,
            SearchEvent::Index(IngestEvent::Started { total }) => Self::IndexStarted { total },
            SearchEvent::Index(IngestEvent::ImageProcessed) => Self::ImageProcessed,
            SearchEvent::Index(IngestEvent::ImageSkipped(error)) => {
                Self::ImageSkipped(error.to_string())
            }
            SearchEvent::Index(IngestEvent::Finished) => Self::IndexFinished,
            SearchEvent::Index(IngestEvent::Artifact(event)) | SearchEvent::Artifact(event) => {
                event.into()
            }
        }
    }
}

impl From<ArtifactEvent> for Event {
    fn from(event: ArtifactEvent) -> Self {
        match event {
            ArtifactEvent::DownloadStarted { artifact } => Self::DownloadStarted {
                artifact: artifact.to_owned(),
            },
            ArtifactEvent::ContentLength { bytes } => Self::ContentLength { bytes },
            ArtifactEvent::BytesRead { bytes } => Self::BytesRead { bytes },
            ArtifactEvent::DownloadFinished => Self::DownloadFinished,
        }
    }
}

fn socket_path(executable: &Path) -> Result<PathBuf, VisionGrepError> {
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

struct SocketGuard {
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

fn bind(path: &Path) -> io::Result<Option<(UnixListener, SocketGuard)>> {
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

struct Connections {
    state: Mutex<ConnectionState>,
    changed: Condvar,
}

struct ConnectionState {
    active: usize,
    idle_since: Instant,
    stopping: bool,
}

struct Connection<'a>(&'a Connections);

impl Drop for Connection<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.active -= 1;
        if state.active == 0 {
            state.idle_since = Instant::now();
        }
        self.0.changed.notify_all();
    }
}

impl Connections {
    fn new() -> Self {
        Self {
            state: Mutex::new(ConnectionState {
                active: 0,
                idle_since: Instant::now(),
                stopping: false,
            }),
            changed: Condvar::new(),
        }
    }

    fn wait_for_slot(&self) -> bool {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let state = self
            .changed
            .wait_while(state, |state| {
                !state.stopping && state.active == MAX_CONNECTIONS
            })
            .unwrap_or_else(PoisonError::into_inner);
        !state.stopping
    }

    fn accepted(&self) -> Connection<'_> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .active += 1;
        self.changed.notify_all();
        Connection(self)
    }

    fn stop(&self) {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stopping = true;
        self.changed.notify_all();
    }

    fn watch(&self, path: &Path, timeout: Duration) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        while !state.stopping {
            if state.active != 0 {
                state = self
                    .changed
                    .wait(state)
                    .unwrap_or_else(PoisonError::into_inner);
                continue;
            }
            let remaining = timeout.saturating_sub(state.idle_since.elapsed());
            if remaining.is_zero() {
                state.stopping = true;
                drop(state);
                let _ = UnixStream::connect(path);
                return;
            }
            (state, _) = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

pub(crate) fn serve() -> Result<(), VisionGrepError> {
    serve_at(&socket_path(&std::env::current_exe()?)?, IDLE_TIMEOUT)
}

fn serve_at(path: &Path, idle_timeout: Duration) -> Result<(), VisionGrepError> {
    let Some((listener, _socket)) = bind(path)? else {
        return Ok(());
    };
    let models = Models::with_lifetime(ArtifactVerification::Fast, SessionLifetime::Process);
    let connections = Connections::new();
    let connections = &connections;
    let models = &models;
    thread::scope(|scope| {
        let _ = thread::Builder::new().spawn_scoped(scope, || models.preload_text());
        thread::Builder::new()
            .spawn_scoped(scope, move || connections.watch(path, idle_timeout))?;
        let result = (|| {
            while connections.wait_for_slot() {
                let stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error),
                };
                let active = connections.accepted();
                thread::Builder::new().spawn_scoped(scope, move || {
                    let _active = active;
                    if let Err(error) = handle_connection(stream, models, connections, path) {
                        eprintln!("visiongrep: {error}");
                    }
                })?;
            }
            Ok(())
        })();
        drop(listener);
        connections.stop();
        result
    })?;
    Ok(())
}

fn handle_connection(
    mut stream: UnixStream,
    models: &Models,
    connections: &Connections,
    path: &Path,
) -> Result<(), VisionGrepError> {
    let timeout = Some(Duration::from_secs(30));
    stream.set_read_timeout(timeout)?;
    stream.set_write_timeout(timeout)?;
    let mut line = Vec::new();
    let count = BufReader::new(&stream)
        .take(MAX_REQUEST_BYTES + 1)
        .read_until(b'\n', &mut line)?;
    if count == 0 {
        return Ok(());
    }
    let request = if count as u64 <= MAX_REQUEST_BYTES {
        serde_json::from_slice(&line).ok()
    } else {
        None
    };
    let Some(request) = request else {
        return write_line(
            &mut stream,
            &Response::Error("invalid socket request".to_owned()),
        );
    };
    let mut write_error = None;
    let result = execute(request, models, connections, path, &mut |event| {
        if write_error.is_none() {
            write_error = write_line(&mut stream, &Response::Event(event.into())).err();
        }
    });
    if let Some(error) = write_error {
        return Err(error);
    }
    let response = result.unwrap_or_else(|error| Response::Error(error.to_string()));
    write_line(&mut stream, &response)
}

fn execute(
    request: Request,
    models: &Models,
    connections: &Connections,
    path: &Path,
    on_event: &mut impl FnMut(SearchEvent),
) -> Result<Response, VisionGrepError> {
    match request {
        Request::Search {
            target,
            query,
            top,
            threshold,
            reindex,
        } => {
            if top == 0 || !(-1.0..=1.0).contains(&threshold) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "invalid top or threshold",
                )
                .into());
            }
            let query = match query {
                SocketQuery::Text(text) if !text.trim().is_empty() => Query::Text(text),
                SocketQuery::Text(_) => {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "blank query").into());
                }
                SocketQuery::Image(bytes) => Query::Image(absolute_path(bytes)?),
            };
            let (root, index) = target.into_paths()?;
            let request = SearchRequest::new(
                query,
                root,
                top,
                threshold,
                if reindex {
                    CacheMode::Reindex
                } else {
                    CacheMode::Use
                },
                Some(index),
                ArtifactVerification::Fast,
            );
            let mut timing = TimingRecorder::new(false, crate::model::timing_metadata());
            let results = search_with_models(&request, models, on_event, &mut timing)?;
            let results = results
                .into_iter()
                .map(|result| {
                    let path = result
                        .path
                        .strip_prefix(request.path())
                        .map_err(|_| io::Error::other("search result outside server root"))?;
                    Ok(ResultItem {
                        path: path.as_os_str().as_bytes().to_vec(),
                        score: result.score,
                    })
                })
                .collect::<Result<_, io::Error>>()?;
            Ok(Response::Results(results))
        }
        Request::Index(target) => {
            let (root, index) = target.into_paths()?;
            Ok(Response::Indexed(prepare_index(
                &root,
                Some(&index),
                models,
                on_event,
            )?))
        }
        Request::Stop => {
            connections.stop();
            let _ = UnixStream::connect(path);
            Ok(Response::Stopped)
        }
    }
}

fn write_line(output: &mut impl Write, value: &impl Serialize) -> Result<(), VisionGrepError> {
    let mut bytes =
        serde_json::to_vec(value).map_err(|source| VisionGrepError::JsonOutput { source })?;
    bytes.push(b'\n');
    output.write_all(&bytes)?;
    Ok(())
}

pub(crate) fn try_search(
    request: &SearchRequest,
    terminal: &mut Terminal,
) -> Result<Option<Vec<SearchResult>>, VisionGrepError> {
    let (root, target) = Target::resolve(request.path(), request.index_path())?;
    let Some(stream) = connect_or_start() else {
        return Ok(None);
    };
    let query = match request.query() {
        Query::Text(text) => SocketQuery::Text(text.clone()),
        Query::Image(path) => {
            let absolute =
                path.canonicalize()
                    .map_err(|source| VisionGrepError::QueryImageOpen {
                        path: path.clone(),
                        source,
                    })?;
            SocketQuery::Image(absolute.as_os_str().as_bytes().to_vec())
        }
    };
    let request = Request::Search {
        target,
        query,
        top: request.top(),
        threshold: request.threshold(),
        reindex: request.is_reindex(),
    };
    let results = match exchange(stream, &request, |event| {
        terminal.handle_socket_event(event)
    })? {
        Some(Response::Results(results)) => results,
        None => return Ok(None),
        _ => return Err(invalid_response()),
    };
    Ok(Some(
        results
            .into_iter()
            .map(|result| SearchResult {
                path: root.display_path().join(OsString::from_vec(result.path)),
                score: result.score,
            })
            .collect(),
    ))
}

pub(super) fn try_index(
    path: &Path,
    index: Option<&Path>,
    terminal: &mut Terminal,
) -> Result<Option<usize>, VisionGrepError> {
    let (_, target) = Target::resolve(path, index)?;
    let Some(stream) = connect_or_start() else {
        return Ok(None);
    };
    match exchange(stream, &Request::Index(target), |event| {
        terminal.handle_socket_event(event)
    })? {
        Some(Response::Indexed(count)) => Ok(Some(count)),
        None => Ok(None),
        _ => Err(invalid_response()),
    }
}

pub(crate) fn stop() -> Result<(), VisionGrepError> {
    let path = socket_path(&std::env::current_exe()?)?;
    let stream = match UnixStream::connect(path) {
        Ok(stream) => stream,
        Err(error) if no_listener(&error) => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    match exchange(stream, &Request::Stop, |_| {})? {
        Some(Response::Stopped) | None => Ok(()),
        _ => Err(invalid_response()),
    }
}

fn exchange(
    mut stream: UnixStream,
    request: &Request,
    mut on_event: impl FnMut(Event),
) -> Result<Option<Response>, VisionGrepError> {
    // A queued connection can close during the write. Read any response already sent.
    match write_line(&mut stream, request) {
        Ok(()) => {}
        Err(VisionGrepError::Io(error)) if peer_closed(&error) => {}
        Err(error) => return Err(error),
    }
    let mut input = BufReader::new(stream);
    let mut line = Vec::new();
    let mut received_event = false;
    loop {
        line.clear();
        let count = match input.read_until(b'\n', &mut line) {
            Ok(count) => count,
            Err(error) if peer_closed(&error) => 0,
            Err(error) => return Err(error.into()),
        };
        if count == 0 {
            if !received_event && line.is_empty() {
                return Ok(None);
            }
            return Err(VisionGrepError::SocketRemote {
                message: "server closed the connection".to_owned(),
            });
        }
        match serde_json::from_slice(&line).map_err(|_| invalid_response())? {
            Response::Event(event) => {
                received_event = true;
                on_event(event);
            }
            Response::Error(message) => return Err(VisionGrepError::SocketRemote { message }),
            response => return Ok(Some(response)),
        }
    }
}

fn peer_closed(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::BrokenPipe | io::ErrorKind::ConnectionReset
    )
}

fn invalid_response() -> VisionGrepError {
    VisionGrepError::SocketRemote {
        message: "invalid server response".to_owned(),
    }
}

fn no_listener(error: &io::Error) -> bool {
    matches!(
        error.kind(),
        io::ErrorKind::NotFound | io::ErrorKind::ConnectionRefused
    )
}

fn connect_or_start() -> Option<UnixStream> {
    let executable = std::env::current_exe().ok()?;
    let path = socket_path(&executable).ok()?;
    match UnixStream::connect(&path) {
        Ok(stream) => return Some(stream),
        Err(error) if no_listener(&error) => {}
        Err(_) => return None,
    }
    let mut child = Command::new(executable)
        .args(["server", "run"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .process_group(0)
        .spawn()
        .ok()?;
    let deadline = Instant::now() + STARTUP_TIMEOUT;
    loop {
        match UnixStream::connect(&path) {
            Ok(stream) => return Some(stream),
            Err(error) if no_listener(&error) => {}
            Err(_) => return None,
        }
        if Instant::now() >= deadline
            || child
                .try_wait()
                .ok()?
                .is_some_and(|status| !status.success())
        {
            return None;
        }
        thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;

    use super::*;

    #[test]
    fn connection_close_allows_fallback_only_before_a_response() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("exchange.sock");
        let listener = UnixListener::bind(&path).unwrap();
        for event in [None, Some(Event::IndexFinished)] {
            let expect_event = event.is_some();
            let mut received = false;
            let response = thread::scope(|scope| {
                scope.spawn(|| {
                    let (mut stream, _) = listener.accept().unwrap();
                    if let Some(event) = event {
                        let mut request = String::new();
                        BufReader::new(&stream).read_line(&mut request).unwrap();
                        write_line(&mut stream, &Response::Event(event)).unwrap();
                    }
                });
                exchange(UnixStream::connect(&path).unwrap(), &Request::Stop, |_| {
                    received = true;
                })
            });
            assert_eq!(received, expect_event);
            if expect_event {
                assert!(
                    matches!(response, Err(VisionGrepError::SocketRemote { message })
                    if message == "server closed the connection")
                );
            } else {
                assert!(matches!(response, Ok(None)));
            }
        }
    }

    #[test]
    fn validates_native_path_bytes() {
        let path = b"/tmp/\xff.png".to_vec();
        let target = Target {
            root: path.clone(),
            index: path.clone(),
        };
        let bytes = serde_json::to_vec(&target).unwrap();
        let target: Target = serde_json::from_slice(&bytes).unwrap();
        let (root, index) = target.into_paths().unwrap();
        assert_eq!(root.as_os_str().as_bytes(), path);
        assert_eq!(index, root);
        assert!(absolute_path(b"relative".to_vec()).is_err());
        assert!(absolute_path(b"/tmp/\0.png".to_vec()).is_err());
    }

    #[test]
    fn idle_server_removes_its_socket() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("idle.sock");
        serve_at(&path, Duration::from_millis(20)).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn watchdog_waits_for_the_last_connection() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("watch.sock");
        let listener = UnixListener::bind(&path).unwrap();
        let connections = Connections::new();
        let active = connections.accepted();
        let (woke, wakeup) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(|| connections.watch(&path, Duration::from_millis(10)));
            scope.spawn(|| {
                listener.accept().unwrap();
                woke.send(()).unwrap();
            });
            let while_active = wakeup.recv_timeout(Duration::from_millis(40));
            drop(active);
            assert_eq!(while_active, Err(mpsc::RecvTimeoutError::Timeout));
            wakeup.recv().unwrap();
        });
        assert!(!connections.wait_for_slot());
    }
}
