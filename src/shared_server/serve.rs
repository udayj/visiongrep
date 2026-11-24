use std::io::{self, BufRead, BufReader, Read};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::sync::{Condvar, Mutex, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use super::address::{bind, socket_path};
use super::protocol::{self, Request, Response, ResultItem, absolute_path, write_line};
use crate::application::{
    CacheMode, Query, SearchEvent, SearchRequest, prepare_index, search_with_models,
};
use crate::error::VisionGrepError;
use crate::model::{ArtifactVerification, Models, SessionLifetime};
use crate::timing::TimingRecorder;

const MAX_REQUEST_BYTES: u64 = 1024 * 1024;
const MAX_CONNECTIONS: usize = 8;
const IDLE_TIMEOUT: Duration = Duration::from_secs(10 * 60);

enum Admission {
    Accept,
    Stopping,
}

struct Activity {
    state: Mutex<ActivityState>,
    changed: Condvar,
}

struct ActivityState {
    active: usize,
    idle_since: Instant,
    stopping: bool,
}

struct ActiveRequest<'a>(&'a Activity);

impl Drop for ActiveRequest<'_> {
    fn drop(&mut self) {
        let mut state = self.0.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.active -= 1;
        if state.active == 0 {
            state.idle_since = Instant::now();
        }
        self.0.changed.notify_all();
    }
}

impl Activity {
    fn new() -> Self {
        Self {
            state: Mutex::new(ActivityState {
                active: 0,
                idle_since: Instant::now(),
                stopping: false,
            }),
            changed: Condvar::new(),
        }
    }

    fn wait_for_slot(&self) -> Admission {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let state = self
            .changed
            .wait_while(state, |state| {
                !state.stopping && state.active == MAX_CONNECTIONS
            })
            .unwrap_or_else(PoisonError::into_inner);
        if state.stopping {
            Admission::Stopping
        } else {
            Admission::Accept
        }
    }

    fn begin(&self) -> ActiveRequest<'_> {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .active += 1;
        self.changed.notify_all();
        ActiveRequest(self)
    }

    fn stop(&self) {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .stopping = true;
        self.changed.notify_all();
    }

    /// Marks the server as stopping and wakes the accept loop, which is blocked in `accept()`.
    fn shut_down(&self, path: &Path) {
        self.stop();
        let _ = UnixStream::connect(path);
    }

    fn stop_when_idle(&self, path: &Path, timeout: Duration) {
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
                self.shut_down(path);
                return;
            }
            (state, _) = self
                .changed
                .wait_timeout(state, remaining)
                .unwrap_or_else(PoisonError::into_inner);
        }
    }
}

pub(crate) fn run() -> Result<(), VisionGrepError> {
    serve_at(&socket_path(&std::env::current_exe()?)?, IDLE_TIMEOUT)
}

fn serve_at(path: &Path, idle_timeout: Duration) -> Result<(), VisionGrepError> {
    let Some((listener, _socket)) = bind(path)? else {
        return Ok(());
    };
    let models = Models::with_lifetime(ArtifactVerification::Fast, SessionLifetime::Process);
    let activity = Activity::new();
    let activity = &activity;
    let models = &models;
    thread::scope(|scope| {
        let _ = thread::Builder::new().spawn_scoped(scope, || models.preload_text());
        thread::Builder::new()
            .spawn_scoped(scope, move || activity.stop_when_idle(path, idle_timeout))?;
        let result = (|| {
            while let Admission::Accept = activity.wait_for_slot() {
                let stream = match listener.accept() {
                    Ok((stream, _)) => stream,
                    Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                    Err(error) => return Err(error),
                };
                let active = activity.begin();
                thread::Builder::new().spawn_scoped(scope, move || {
                    let _active = active;
                    if let Err(error) = handle_connection(stream, models, activity, path) {
                        eprintln!("visiongrep: {error}");
                    }
                })?;
            }
            Ok(())
        })();
        drop(listener);
        activity.stop();
        result
    })?;
    Ok(())
}

fn handle_connection(
    mut stream: UnixStream,
    models: &Models,
    activity: &Activity,
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
    let result = execute(request, models, activity, path, &mut |event| {
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
    activity: &Activity,
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
            let query = match query {
                protocol::Query::Text(text) if !text.trim().is_empty() => Query::Text(text),
                protocol::Query::Text(_) => {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "blank query").into());
                }
                protocol::Query::Image(bytes) => Query::Image(absolute_path(bytes)?),
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
            activity.shut_down(path);
            Ok(Response::Stopped)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixListener;
    use std::sync::mpsc;

    use super::*;

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
        let activity = Activity::new();
        let active = activity.begin();
        let (woke, wakeup) = mpsc::channel();
        thread::scope(|scope| {
            scope.spawn(|| activity.stop_when_idle(&path, Duration::from_millis(10)));
            scope.spawn(|| {
                listener.accept().unwrap();
                woke.send(()).unwrap();
            });
            let while_active = wakeup.recv_timeout(Duration::from_millis(40));
            drop(active);
            assert_eq!(while_active, Err(mpsc::RecvTimeoutError::Timeout));
            wakeup.recv().unwrap();
        });
        assert!(matches!(activity.wait_for_slot(), Admission::Stopping));
    }
}
