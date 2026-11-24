use std::ffi::OsString;
use std::io::{self, Write};
use std::num::NonZeroUsize;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::application::SearchEvent;
use crate::error::VisionGrepError;
use crate::index::IngestEvent;
use crate::model::ArtifactEvent;
use crate::ranking::SimilarityThreshold;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) enum Request {
    Search {
        target: Target,
        query: Query,
        top: NonZeroUsize,
        threshold: SimilarityThreshold,
        reindex: bool,
    },
    Index(Target),
    Stop,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Target {
    root: Vec<u8>,
    index: Vec<u8>,
}

impl Target {
    pub(super) fn new(root: &Path, index: &Path) -> Self {
        Self {
            root: root.as_os_str().as_bytes().to_vec(),
            index: index.as_os_str().as_bytes().to_vec(),
        }
    }

    pub(super) fn into_paths(self) -> io::Result<(PathBuf, PathBuf)> {
        Ok((absolute_path(self.root)?, absolute_path(self.index)?))
    }
}

pub(super) fn absolute_path(bytes: Vec<u8>) -> io::Result<PathBuf> {
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
pub(super) enum Query {
    Text(String),
    Image(Vec<u8>),
}

#[derive(Serialize, Deserialize)]
pub(super) enum Response {
    Event(Event),
    Results(Vec<ResultItem>),
    Indexed(usize),
    Stopped,
    Error(String),
}

#[derive(Serialize, Deserialize)]
pub(super) struct ResultItem {
    pub(super) path: Vec<u8>,
    pub(super) score: f32,
}

#[derive(Serialize, Deserialize)]
pub(crate) enum Event {
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

pub(super) fn write_line(
    output: &mut impl Write,
    value: &impl Serialize,
) -> Result<(), VisionGrepError> {
    let mut bytes =
        serde_json::to_vec(value).map_err(|source| VisionGrepError::JsonOutput { source })?;
    bytes.push(b'\n');
    output.write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn search_parameters_use_json_numbers_and_reject_out_of_range_values() {
        let valid = json!({
            "Search": {
                "target": {"root": [47], "index": [47, 255]},
                "query": {"Image": [47, 255]},
                "top": 5,
                "threshold": -1.0,
                "reindex": false,
            }
        });
        let request: Request = serde_json::from_value(valid.clone()).unwrap();
        assert_eq!(serde_json::to_value(request).unwrap(), valid);

        for (field, value) in [("top", json!(0)), ("threshold", json!(1.01))] {
            let mut invalid = valid.clone();
            invalid["Search"][field] = value;
            assert!(serde_json::from_value::<Request>(invalid).is_err());
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
}
