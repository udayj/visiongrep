use std::ffi::OsString;
use std::io::{self, BufRead, BufReader};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::net::UnixStream;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use super::address::socket_path;
use super::protocol::{self, Event, Request, Response, Target, write_line};
use crate::application::{Query, SearchRequest, resolve_root};
use crate::error::VisionGrepError;
use crate::index::{IndexLocation, SearchRoot};
use crate::ranking::SearchResult;

const STARTUP_TIMEOUT: Duration = Duration::from_secs(2);

pub(crate) fn search(
    request: &SearchRequest,
    on_event: impl FnMut(Event),
) -> Result<Option<Vec<SearchResult>>, VisionGrepError> {
    let (root, target) = resolve_target(request.path(), request.index_path())?;
    let Some(stream) = connect_or_start() else {
        return Ok(None);
    };
    let query = match request.query() {
        Query::Text(text) => protocol::Query::Text(text.clone()),
        Query::Image(path) => {
            let absolute =
                path.canonicalize()
                    .map_err(|source| VisionGrepError::QueryImageOpen {
                        path: path.clone(),
                        source,
                    })?;
            protocol::Query::Image(absolute.as_os_str().as_bytes().to_vec())
        }
    };
    let request = Request::Search {
        target,
        query,
        top: request.top(),
        threshold: request.threshold(),
        reindex: request.is_reindex(),
    };
    let results = match exchange(stream, &request, on_event)? {
        Reply::Response(Response::Results(results)) => results,
        Reply::Unanswered => return Ok(None),
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

pub(crate) fn index(
    path: &Path,
    index: Option<&Path>,
    on_event: impl FnMut(Event),
) -> Result<Option<usize>, VisionGrepError> {
    let (_, target) = resolve_target(path, index)?;
    let Some(stream) = connect_or_start() else {
        return Ok(None);
    };
    match exchange(stream, &Request::Index(target), on_event)? {
        Reply::Response(Response::Indexed(count)) => Ok(Some(count)),
        Reply::Unanswered => Ok(None),
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
        Reply::Response(Response::Stopped) | Reply::Unanswered => Ok(()),
        _ => Err(invalid_response()),
    }
}

fn resolve_target(
    path: &Path,
    index: Option<&Path>,
) -> Result<(SearchRoot, Target), VisionGrepError> {
    let root = resolve_root(path)?;
    let index = IndexLocation::resolve(root.filesystem_path(), index)?;
    let target = Target::new(root.filesystem_path(), index.path());
    Ok((root, target))
}

enum Reply {
    Response(Response),
    /// The server closed the connection before sending anything; the caller falls back
    /// to an in-process run.
    Unanswered,
}

fn exchange(
    mut stream: UnixStream,
    request: &Request,
    mut on_event: impl FnMut(Event),
) -> Result<Reply, VisionGrepError> {
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
                return Ok(Reply::Unanswered);
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
            response => return Ok(Reply::Response(response)),
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
    use std::os::unix::net::UnixListener;

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
                assert!(matches!(response, Ok(Reply::Unanswered)));
            }
        }
    }
}
