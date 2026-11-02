use std::ffi::OsString;
use std::fs;
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::terminal::Terminal;
use crate::application::{Query, SearchEvent, SearchRequest, SearchService};
use crate::error::VisionGrepError;
use crate::index::{IndexLocation, IngestEvent, SearchRoot};
use crate::model::ArtifactEvent;
use crate::ranking::SearchResult;
use crate::timing::TimingRecorder;

const MAX_REQUEST_BYTES: u64 = 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    query: SocketQuery,
    top: usize,
    threshold: f32,
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
    Error(String),
}

#[derive(Serialize, Deserialize)]
struct ResultItem {
    path: Vec<u8>,
    score: f32,
}

#[derive(Serialize, Deserialize)]
pub(super) enum Event {
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
            SearchEvent::Index(IngestEvent::Started { total }) => Self::IndexStarted { total },
            SearchEvent::Index(IngestEvent::ImageProcessed) => Self::ImageProcessed,
            SearchEvent::Index(IngestEvent::ImageSkipped(error)) => {
                Self::ImageSkipped(error.to_string())
            }
            SearchEvent::Index(IngestEvent::Finished) => Self::IndexFinished,
            SearchEvent::Artifact(ArtifactEvent::DownloadStarted { artifact }) => {
                Self::DownloadStarted {
                    artifact: artifact.to_owned(),
                }
            }
            SearchEvent::Artifact(ArtifactEvent::ContentLength { bytes }) => {
                Self::ContentLength { bytes }
            }
            SearchEvent::Artifact(ArtifactEvent::BytesRead { bytes }) => Self::BytesRead { bytes },
            SearchEvent::Artifact(ArtifactEvent::DownloadFinished) => Self::DownloadFinished,
        }
    }
}

fn socket_path(root: &Path, index_path: Option<&Path>) -> Result<PathBuf, VisionGrepError> {
    let location = IndexLocation::resolve(root, index_path)?;
    let mut hash = Sha256::new();
    let root_bytes = root.as_os_str().as_bytes();
    hash.update((root_bytes.len() as u64).to_le_bytes());
    hash.update(root_bytes);
    hash.update(location.path().as_os_str().as_bytes());
    let digest = hash.finalize();
    let name = digest[..8]
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    Ok(crate::model::cache_dir()?
        .join("run")
        .join(format!("{name}.sock")))
}

struct SocketGuard(PathBuf);

impl Drop for SocketGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

fn bind(path: &Path, root: &Path) -> Result<(UnixListener, SocketGuard), VisionGrepError> {
    let directory = path
        .parent()
        .ok_or_else(|| io::Error::other("socket path has no parent"))?;
    fs::create_dir_all(directory)?;
    fs::set_permissions(directory, fs::Permissions::from_mode(0o700))?;
    match UnixStream::connect(path) {
        Ok(_) => {
            return Err(VisionGrepError::SocketServerRunning {
                path: root.to_owned(),
            });
        }
        Err(error) if error.kind() == io::ErrorKind::ConnectionRefused => fs::remove_file(path)?,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => return Err(error.into()),
    }
    let listener = UnixListener::bind(path)?;
    Ok((listener, SocketGuard(path.to_owned())))
}

pub(crate) fn serve(path: &Path, index_path: Option<&Path>) -> Result<(), VisionGrepError> {
    let root = SearchRoot::resolve(path)?;
    let path = socket_path(root.filesystem_path(), index_path)?;
    let (listener, _guard) = bind(&path, root.filesystem_path())?;
    let mut service = SearchService::open(root.filesystem_path(), index_path)?;
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(stream) => stream,
            Err(error) => {
                eprintln!("visiongrep: {error}");
                continue;
            }
        };
        // A disconnected client must not stop the server or affect later searches.
        let result = (|| {
            stream.set_read_timeout(Some(Duration::from_secs(30)))?;
            let reader = stream.try_clone()?;
            serve_requests(
                &mut service,
                root.filesystem_path(),
                &mut BufReader::new(reader),
                &mut stream,
            )
        })();
        if let Err(error) = result {
            eprintln!("visiongrep: {error}");
        }
    }
    Ok(())
}

fn serve_requests(
    service: &mut SearchService,
    root: &Path,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<(), VisionGrepError> {
    let mut line = Vec::new();
    loop {
        line.clear();
        if input
            .take(MAX_REQUEST_BYTES + 1)
            .read_until(b'\n', &mut line)?
            == 0
        {
            return Ok(());
        }
        if line.len() as u64 > MAX_REQUEST_BYTES {
            write_response(
                output,
                &Response::Error("invalid request; restart the server".to_owned()),
            )?;
            return Ok(());
        }
        let request: Request = match serde_json::from_slice(&line) {
            Ok(request) => request,
            Err(_) => {
                write_response(
                    output,
                    &Response::Error("invalid request; restart the server".to_owned()),
                )?;
                continue;
            }
        };
        let query = match request.query {
            SocketQuery::Text(text) => Query::Text(text),
            SocketQuery::Image(bytes) => Query::Image(PathBuf::from(OsString::from_vec(bytes))),
        };
        let mut write_error = None;
        let mut timing = TimingRecorder::new(false, crate::model::timing_metadata());
        let result = service.search(
            &query,
            request.top,
            request.threshold,
            &mut |event| {
                if write_error.is_none() {
                    write_error = write_response(output, &Response::Event(event.into())).err();
                }
            },
            &mut timing,
        );
        if let Some(error) = write_error {
            return Err(error);
        }
        let response = match result {
            Ok(results) => {
                let results = results
                    .into_iter()
                    .map(|result| {
                        let path = result
                            .path
                            .strip_prefix(root)
                            .map_err(|_| io::Error::other("search result outside server root"))?;
                        Ok(ResultItem {
                            path: path.as_os_str().as_bytes().to_vec(),
                            score: result.score,
                        })
                    })
                    .collect::<Result<Vec<_>, io::Error>>()?;
                Response::Results(results)
            }
            Err(error) => Response::Error(error.to_string()),
        };
        write_response(output, &response)?;
    }
}

fn write_response(output: &mut impl Write, response: &Response) -> Result<(), VisionGrepError> {
    let mut bytes =
        serde_json::to_vec(response).map_err(|source| VisionGrepError::JsonOutput { source })?;
    bytes.push(b'\n');
    output.write_all(&bytes)?;
    output.flush()?;
    Ok(())
}

pub(crate) fn try_search(
    request: &SearchRequest,
    terminal: &mut Terminal,
) -> Result<Option<Vec<SearchResult>>, VisionGrepError> {
    let Some((root, mut stream)) = connect(request) else {
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
    let request = Request {
        query,
        top: request.top(),
        threshold: request.threshold(),
    };
    let mut bytes =
        serde_json::to_vec(&request).map_err(|source| VisionGrepError::JsonOutput { source })?;
    bytes.push(b'\n');
    stream.write_all(&bytes)?;
    let mut input = BufReader::new(stream);
    loop {
        let mut line = Vec::new();
        if input.read_until(b'\n', &mut line)? == 0 {
            return Err(VisionGrepError::SocketRemote {
                message: "server closed the connection; restart the server".to_owned(),
            });
        }
        let response: Response =
            serde_json::from_slice(&line).map_err(|_| VisionGrepError::SocketRemote {
                message: "invalid server response; restart the server".to_owned(),
            })?;
        match response {
            Response::Event(event) => terminal.handle_socket_event(event),
            Response::Results(results) => {
                return Ok(Some(
                    results
                        .into_iter()
                        .map(|result| SearchResult {
                            path: root.display_path().join(OsString::from_vec(result.path)),
                            score: result.score,
                        })
                        .collect(),
                ));
            }
            Response::Error(message) => return Err(VisionGrepError::SocketRemote { message }),
        }
    }
}

pub(crate) fn server_running(request: &SearchRequest) -> bool {
    connect(request).is_some()
}

fn connect(request: &SearchRequest) -> Option<(SearchRoot, UnixStream)> {
    // Leave path and index validation to the ordinary search if no server can be found.
    let root = SearchRoot::resolve(request.path()).ok()?;
    let path = socket_path(root.filesystem_path(), request.index_path()).ok()?;
    let stream = UnixStream::connect(path).ok()?;
    Some((root, stream))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_request_does_not_end_stream() {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let mut service = SearchService::open(&root, None).unwrap();
        let mut output = Vec::new();
        let input = b"invalid\n{\"query\":{\"Text\":\"cat\"},\"top\":5,\"threshold\":0.25}\n";
        serve_requests(&mut service, &root, &mut &input[..], &mut output).unwrap();
        let responses: Vec<Response> = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert!(
            matches!(&responses[..], [Response::Error(message), Response::Results(results)] if message.contains("restart") && results.is_empty())
        );
    }

    #[test]
    fn native_path_bytes_survive_protocol_round_trip() {
        let path = b"image-\xff.png".to_vec();
        let request = Request {
            query: SocketQuery::Image(path.clone()),
            top: 5,
            threshold: 0.25,
        };
        let encoded = serde_json::to_vec(&request).unwrap();
        let decoded: Request = serde_json::from_slice(&encoded).unwrap();
        assert!(matches!(decoded.query, SocketQuery::Image(bytes) if bytes == path));

        let response = Response::Results(vec![ResultItem {
            path: path.clone(),
            score: 0.5,
        }]);
        let encoded = serde_json::to_vec(&response).unwrap();
        let decoded: Response = serde_json::from_slice(&encoded).unwrap();
        assert!(matches!(decoded, Response::Results(results) if results[0].path == path));
    }

    #[test]
    fn stale_socket_is_replaced_and_second_server_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("run/test.sock");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        drop(UnixListener::bind(&path).unwrap());
        assert_eq!(
            UnixStream::connect(&path).unwrap_err().kind(),
            io::ErrorKind::ConnectionRefused
        );
        let (listener, guard) = bind(&path, directory.path()).unwrap();
        assert!(matches!(
            bind(&path, directory.path()),
            Err(VisionGrepError::SocketServerRunning { .. })
        ));
        drop(listener);
        drop(guard);
        assert!(!path.exists());
    }
}
