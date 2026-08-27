use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::error::VisionGrepError;

const VISION_MODEL_URL: &str =
    "https://huggingface.co/Qdrant/clip-ViT-B-32-vision/resolve/main/model.onnx";
const TEXT_MODEL_URL: &str =
    "https://huggingface.co/Qdrant/clip-ViT-B-32-text/resolve/main/model.onnx";
const TOKENIZER_URL: &str =
    "https://huggingface.co/Qdrant/clip-ViT-B-32-text/resolve/main/tokenizer.json";

const VISION_MODEL_SHA256: &str =
    "c68d3d9a200ddd2a8c8a5510b576d4c94d1ae383bf8b36dd8c084f94e1fb4d63";
const TEXT_MODEL_SHA256: &str = "4dbe762b11e36488304471e439cde89da053ad7acaddbf9e096745d142ec8d8b";

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
    fs::create_dir_all(&model_dir)?;

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
        Some(VISION_MODEL_SHA256),
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
        Some(TEXT_MODEL_SHA256),
        on_event,
    )?;
    ensure_artifact(
        "CLIP tokenizer",
        TOKENIZER_URL,
        &paths.tokenizer,
        None,
        on_event,
    )?;

    Ok(())
}

fn cache_dir() -> Result<PathBuf, VisionGrepError> {
    let home = std::env::var_os("HOME").ok_or(VisionGrepError::HomeDirectory)?;
    Ok(PathBuf::from(home).join(".cache").join("visiongrep"))
}

/// Ensures an artifact is usable, replacing an existing checksummed file when validation fails.
///
/// Checksummed models are trusted only after SHA-256 validation. The tokenizer currently has no
/// published checksum, so an existing tokenizer is accepted as-is.
fn ensure_artifact(
    artifact: &'static str,
    url: &str,
    destination: &Path,
    expected_sha256: Option<&'static str>,
    on_event: &mut impl FnMut(ArtifactEvent),
) -> Result<(), VisionGrepError> {
    if destination.exists() {
        if let Some(expected) = expected_sha256 {
            if sha256_file(destination)? == expected {
                return Ok(());
            }
            fs::remove_file(destination)?;
        } else {
            return Ok(());
        }
    }

    download_artifact(artifact, url, destination, on_event)?;

    if let Some(expected) = expected_sha256 {
        let actual = sha256_file(destination)?;
        if actual != expected {
            fs::remove_file(destination)?;
            return Err(VisionGrepError::Checksum {
                file: destination.to_owned(),
                expected,
                actual,
            });
        }
    }

    Ok(())
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
    on_event: &mut impl FnMut(ArtifactEvent),
) -> Result<(), VisionGrepError> {
    on_event(ArtifactEvent::DownloadStarted { artifact });
    let mut response =
        reqwest::blocking::get(url).map_err(|source| VisionGrepError::DownloadRequest {
            artifact,
            url: url.to_owned(),
            source,
        })?;
    let total = response
        .content_length()
        .ok_or_else(|| VisionGrepError::MissingContentLength {
            artifact,
            url: url.to_owned(),
        })?;

    let tmp_path = destination.with_extension("tmp");
    let mut file = File::create(&tmp_path)?;
    on_event(ArtifactEvent::ContentLength { bytes: total });
    let stream_result: Result<(), VisionGrepError> = (|| {
        let mut buffer = [0_u8; 64 * 1024];
        loop {
            let read =
                response
                    .read(&mut buffer)
                    .map_err(|source| VisionGrepError::DownloadRead {
                        artifact,
                        url: url.to_owned(),
                        source,
                    })?;
            if read == 0 {
                break;
            }
            file.write_all(&buffer[..read])?;
            let bytes =
                u64::try_from(read).map_err(|source| VisionGrepError::NumericConversion {
                    context: "updating download progress",
                    source,
                })?;
            on_event(ArtifactEvent::BytesRead { bytes });
        }
        Ok(())
    })();
    on_event(ArtifactEvent::DownloadFinished);
    stream_result?;
    fs::rename(tmp_path, destination)?;

    Ok(())
}

fn sha256_file(path: &Path) -> Result<String, VisionGrepError> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(format!("{:x}", hasher.finalize()))
}
