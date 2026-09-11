use std::io::{self, BufRead, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::terminal::{OutputFormat, Terminal};
use crate::application::{Query, SearchService};
use crate::error::VisionGrepError;
use crate::ranking::{DEFAULT_SIMILARITY_THRESHOLD, SearchResult};
use crate::timing::TimingRecorder;

const MAX_REQUEST_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
enum RequestId {
    Number(u64),
    Text(String),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: Option<RequestId>,
    query: Option<String>,
    image: Option<PathBuf>,
    #[serde(default = "default_top")]
    top: usize,
    #[serde(default = "default_threshold")]
    threshold: f32,
}

fn default_top() -> usize {
    5
}

fn default_threshold() -> f32 {
    DEFAULT_SIMILARITY_THRESHOLD
}

#[derive(Debug, thiserror::Error)]
enum RequestError {
    #[error("provide exactly one non-blank query or image path")]
    Query,
    #[error("top must be at least 1")]
    Top,
    #[error("threshold must be between -1.0 and 1.0")]
    Threshold,
    #[error(transparent)]
    Search(#[from] VisionGrepError),
}

#[derive(Serialize)]
#[serde(untagged)]
enum Response<'a> {
    Results {
        id: &'a Option<RequestId>,
        results: Vec<SearchResult>,
    },
    Error {
        id: &'a Option<RequestId>,
        error: String,
    },
}

impl Request {
    fn execute(
        &mut self,
        service: &mut SearchService,
        terminal: &mut Terminal,
    ) -> Result<Vec<SearchResult>, RequestError> {
        if self.top == 0 {
            return Err(RequestError::Top);
        }
        if !(-1.0..=1.0).contains(&self.threshold) {
            return Err(RequestError::Threshold);
        }
        let query = match (self.query.take(), self.image.take()) {
            (Some(text), None) if !text.trim().is_empty() => Query::Text(text),
            (None, Some(path)) if !path.as_os_str().is_empty() => Query::Image(path),
            _ => return Err(RequestError::Query),
        };
        let mut timing = TimingRecorder::new(false, crate::model::timing_metadata());
        let results = service.search(
            &query,
            self.top,
            self.threshold,
            &mut |event| terminal.handle_event(event),
            &mut timing,
        )?;
        if let Some(result) = results.iter().find(|result| result.path.to_str().is_none()) {
            return Err(VisionGrepError::NonUtf8JsonPath {
                path: result.path.clone(),
            }
            .into());
        }
        Ok(results)
    }
}

pub(crate) fn serve(path: &Path, index_path: Option<&Path>) -> Result<(), VisionGrepError> {
    let mut service = SearchService::open(path, index_path)?;
    serve_requests(
        &mut service,
        &mut io::stdin().lock(),
        &mut io::stdout().lock(),
    )
}

fn serve_requests(
    service: &mut SearchService,
    input: &mut impl BufRead,
    output: &mut impl Write,
) -> Result<(), VisionGrepError> {
    let mut line = Vec::new();
    let mut terminal = Terminal::new(OutputFormat::Json, false);
    loop {
        line.clear();
        let count = input
            .take(MAX_REQUEST_BYTES + 1)
            .read_until(b'\n', &mut line)?;
        if count == 0 {
            return Ok(());
        }
        if line.len() as u64 > MAX_REQUEST_BYTES {
            return Err(VisionGrepError::StdioRequestTooLarge {
                limit: MAX_REQUEST_BYTES,
            });
        }
        let mut request = match serde_json::from_slice::<Request>(&line) {
            Ok(request) => request,
            Err(error) => {
                write_response(
                    output,
                    &Response::Error {
                        id: &None,
                        error: error.to_string(),
                    },
                )?;
                continue;
            }
        };
        let response = match request.execute(service, &mut terminal) {
            Ok(results) => Response::Results {
                id: &request.id,
                results,
            },
            Err(error) => Response::Error {
                id: &request.id,
                error: error.to_string(),
            },
        };
        write_response(output, &response)?;
    }
}

fn write_response(output: &mut impl Write, response: &Response<'_>) -> Result<(), VisionGrepError> {
    // Serialize first so a serialization failure cannot leave a partial JSON line.
    let mut bytes =
        serde_json::to_vec(response).map_err(|source| VisionGrepError::JsonOutput { source })?;
    bytes.push(b'\n');
    output.write_all(&bytes)?;
    output.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_requests_do_not_end_the_stream() {
        let directory = tempfile::tempdir().unwrap();
        let mut service = SearchService::open(directory.path(), None).unwrap();
        let input = b"not json\n{\"id\":1,\"query\":\"dog\",\"image\":\"dog.png\"}\n{\"id\":2,\"query\":\"dog\",\"top\":0}\n{\"id\":3,\"query\":\"dog\",\"threshold\":2}\n{\"id\":\"ok\",\"query\":\"dog\"}";
        let mut output = Vec::new();
        serve_requests(&mut service, &mut &input[..], &mut output).unwrap();
        let responses: Vec<serde_json::Value> = output
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice(line).unwrap())
            .collect();
        assert_eq!(responses.len(), 5);
        assert_eq!(responses[0]["id"], serde_json::Value::Null);
        for (response, id) in responses[1..4].iter().zip(1..=3) {
            assert_eq!(response["id"], id);
            assert!(response["error"].is_string());
        }
        assert_eq!(responses[4], serde_json::json!({"id":"ok", "results":[]}));
    }

    #[test]
    fn oversized_request_is_a_fatal_error() {
        let directory = tempfile::tempdir().unwrap();
        let mut service = SearchService::open(directory.path(), None).unwrap();
        let input = vec![b' '; MAX_REQUEST_BYTES as usize + 1];
        let mut output = Vec::new();
        assert!(matches!(
            serve_requests(&mut service, &mut &input[..], &mut output),
            Err(VisionGrepError::StdioRequestTooLarge { .. })
        ));
        assert!(output.is_empty());
    }
}
