use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

use crate::error::VisionGrepError;

const VISION_MODEL_URL: &str = "https://huggingface.co/Qdrant/clip-ViT-B-32-vision/resolve/a636590e595dbbd798647c9dd4550d5652fba969/model.onnx";
const TEXT_MODEL_URL: &str = "https://huggingface.co/Qdrant/clip-ViT-B-32-text/resolve/48ca1db27cb4063eb311ec2aa7f087a808112876/model.onnx";
const TOKENIZER_URL: &str = "https://huggingface.co/Qdrant/clip-ViT-B-32-text/resolve/48ca1db27cb4063eb311ec2aa7f087a808112876/tokenizer.json";

const VISION_MODEL_SHA256: &str =
    "c68d3d9a200ddd2a8c8a5510b576d4c94d1ae383bf8b36dd8c084f94e1fb4d63";
const TEXT_MODEL_SHA256: &str = "4dbe762b11e36488304471e439cde89da053ad7acaddbf9e096745d142ec8d8b";
const TOKENIZER_SHA256: &str = "b68d571997a1f81bf521fb73806740ddb91e4ed6666cb6e996c066bb289cf55b";

#[derive(Debug, Clone)]
pub(crate) struct ModelPaths {
    pub(crate) vision_model: PathBuf,
    pub(crate) text_model: PathBuf,
    pub(crate) tokenizer: PathBuf,
}

pub(crate) enum ArtifactEvent {
    DownloadStarted { artifact: &'static str },
    ContentLength { bytes: u64 },
    BytesRead { bytes: u64 },
    DownloadFinished,
}

pub(crate) fn model_paths() -> Result<ModelPaths, VisionGrepError> {
    let model_dir = cache_dir()?.join("models");
    fs::create_dir_all(&model_dir).map_err(|source| VisionGrepError::ArtifactFile {
        operation: "creating model cache directory",
        path: model_dir.clone(),
        source,
    })?;

    Ok(ModelPaths {
        vision_model: model_dir.join("clip_vision.onnx"),
        text_model: model_dir.join("clip_text.onnx"),
        tokenizer: model_dir.join("tokenizer.json"),
    })
}

pub(crate) fn ensure_vision_artifacts(
    paths: &ModelPaths,
    on_event: &mut impl FnMut(ArtifactEvent),
) -> Result<(), VisionGrepError> {
    ensure_artifact(
        "CLIP vision model",
        VISION_MODEL_URL,
        &paths.vision_model,
        VISION_MODEL_SHA256,
        on_event,
    )
}

pub(crate) fn ensure_text_artifacts(
    paths: &ModelPaths,
    on_event: &mut impl FnMut(ArtifactEvent),
) -> Result<(), VisionGrepError> {
    ensure_artifact(
        "CLIP text model",
        TEXT_MODEL_URL,
        &paths.text_model,
        TEXT_MODEL_SHA256,
        on_event,
    )?;
    ensure_artifact(
        "CLIP tokenizer",
        TOKENIZER_URL,
        &paths.tokenizer,
        TOKENIZER_SHA256,
        on_event,
    )?;

    Ok(())
}

fn cache_dir() -> Result<PathBuf, VisionGrepError> {
    let home = std::env::var_os("HOME").ok_or(VisionGrepError::HomeDirectory)?;
    Ok(PathBuf::from(home).join(".cache").join("visiongrep"))
}

/// Ensures an artifact matches its pinned SHA-256, replacing invalid files atomically.
fn ensure_artifact(
    artifact: &'static str,
    url: &str,
    destination: &Path,
    expected_sha256: &str,
    on_event: &mut impl FnMut(ArtifactEvent),
) -> Result<(), VisionGrepError> {
    if destination.exists() && sha256_file(destination)? == expected_sha256 {
        return Ok(());
    }

    download_artifact(artifact, url, destination, expected_sha256, on_event)
}

/// Streams an artifact into a sibling temporary file before replacing the final destination.
///
/// Transfer events keep installation independent of terminal presentation. The destination remains
/// untouched until the complete response is written, so a partial download cannot be mistaken for
/// a usable model on the next run.
fn download_artifact(
    artifact: &'static str,
    url: &str,
    destination: &Path,
    expected_sha256: &str,
    on_event: &mut impl FnMut(ArtifactEvent),
) -> Result<(), VisionGrepError> {
    on_event(ArtifactEvent::DownloadStarted { artifact });
    let mut response = reqwest::blocking::get(url)
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|source| VisionGrepError::DownloadRequest {
            artifact,
            url: url.to_owned(),
            source,
        })?;
    let content_length = response.content_length();
    install_download(
        artifact,
        url,
        &mut response,
        content_length,
        destination,
        expected_sha256,
        on_event,
    )
}

fn install_download(
    artifact: &'static str,
    url: &str,
    response: &mut impl Read,
    content_length: Option<u64>,
    destination: &Path,
    expected_sha256: &str,
    on_event: &mut impl FnMut(ArtifactEvent),
) -> Result<(), VisionGrepError> {
    if let Some(bytes) = content_length {
        on_event(ArtifactEvent::ContentLength { bytes });
    }

    let parent =
        destination
            .parent()
            .ok_or_else(|| VisionGrepError::ArtifactDestinationWithoutParent {
                path: destination.to_owned(),
            })?;
    let mut temporary =
        NamedTempFile::new_in(parent).map_err(|source| VisionGrepError::ArtifactFile {
            operation: "creating temporary download",
            path: parent.to_owned(),
            source,
        })?;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|source| VisionGrepError::DownloadRead {
                artifact,
                url: url.to_owned(),
                source,
            })?;
        if read == 0 {
            break;
        }
        temporary
            .write_all(&buffer[..read])
            .map_err(|source| VisionGrepError::ArtifactFile {
                operation: "writing temporary download",
                path: temporary.path().to_owned(),
                source,
            })?;
        let bytes = u64::try_from(read).map_err(|source| VisionGrepError::NumericConversion {
            context: "updating download progress",
            source,
        })?;
        on_event(ArtifactEvent::BytesRead { bytes });
    }
    temporary
        .flush()
        .map_err(|source| VisionGrepError::ArtifactFile {
            operation: "flushing temporary download",
            path: temporary.path().to_owned(),
            source,
        })?;

    let actual = sha256_file(temporary.path())?;
    if actual != expected_sha256 {
        return Err(VisionGrepError::Checksum {
            file: destination.to_owned(),
            expected: expected_sha256.to_owned(),
            actual,
        });
    }

    temporary
        .persist(destination)
        .map_err(|error| VisionGrepError::ArtifactFile {
            operation: "installing verified download",
            path: destination.to_owned(),
            source: error.error,
        })?;
    on_event(ArtifactEvent::DownloadFinished);

    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, VisionGrepError> {
    let mut file = File::open(path).map_err(|source| VisionGrepError::ArtifactFile {
        operation: "opening artifact for checksum verification",
        path: path.to_owned(),
        source,
    })?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| VisionGrepError::ArtifactFile {
                operation: "reading artifact for checksum verification",
                path: path.to_owned(),
                source,
            })?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn sha256(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn download_accepts_a_response_without_content_length() {
        const BODY: &[u8] = b"verified artifact";
        let mut response = Cursor::new(BODY);
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        let mut events = Vec::new();

        install_download(
            "test artifact",
            "https://example.invalid/artifact",
            &mut response,
            None,
            &destination,
            &sha256(BODY),
            &mut |event| events.push(event),
        )
        .unwrap();

        assert_eq!(fs::read(destination).unwrap(), BODY);
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, ArtifactEvent::ContentLength { .. }))
        );
        assert!(matches!(
            events.last(),
            Some(ArtifactEvent::DownloadFinished)
        ));
    }

    #[test]
    fn existing_verified_artifact_does_not_download_again() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        fs::write(&destination, b"verified artifact").unwrap();
        let mut events = Vec::new();

        ensure_artifact(
            "test artifact",
            "https://example.invalid/artifact",
            &destination,
            &sha256(b"verified artifact"),
            &mut |event| events.push(event),
        )
        .unwrap();

        assert!(events.is_empty());
    }

    #[test]
    fn checksum_failure_preserves_the_existing_artifact() {
        let mut response = Cursor::new(b"bad");
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        fs::write(&destination, b"previous artifact").unwrap();

        let error = install_download(
            "test artifact",
            "https://example.invalid/artifact",
            &mut response,
            Some(3),
            &destination,
            &sha256(b"good"),
            &mut |_| {},
        )
        .unwrap_err();

        assert!(matches!(error, VisionGrepError::Checksum { .. }));
        assert_eq!(fs::read(destination).unwrap(), b"previous artifact");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }
}
