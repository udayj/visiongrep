use std::ffi::OsString;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;

use crate::application::ArtifactVerification;
use crate::embedding::EmbeddingContract;
use crate::error::VisionGrepError;
use crate::timing::{ModelMetadata, Phase, TimingRecorder};

const VERIFIED_MANIFEST_VERSION: u32 = 1;
const VERIFIED_MANIFEST_SUFFIX: &str = ".verified.json";
const MODEL_CONTRACT: &str = "qdrant-clip-vit-b-32-224-v1";
const IMAGE_EMBEDDING_CONTRACT: &str = "qdrant-clip-vit-b-32-224-v1;vision=a636590e595dbbd798647c9dd4550d5652fba969;sha256=c68d3d9a200ddd2a8c8a5510b576d4c94d1ae383bf8b36dd8c084f94e1fb4d63;preprocess=rgb-center-crop-224-catmullrom-openai-clip-normalization";
const QUERY_EMBEDDING_CONTRACT: &str = "qdrant-clip-vit-b-32-224-v1;text=48ca1db27cb4063eb311ec2aa7f087a808112876;sha256=4dbe762b11e36488304471e439cde89da053ad7acaddbf9e096745d142ec8d8b;tokenizer_sha256=b68d571997a1f81bf521fb73806740ddb91e4ed6666cb6e996c066bb289cf55b;tokens=77-pad-1";
const VISION_MODEL_REVISION: &str = "a636590e595dbbd798647c9dd4550d5652fba969";
const TEXT_MODEL_REVISION: &str = "48ca1db27cb4063eb311ec2aa7f087a808112876";
const VISION_MODEL_URL: &str = "https://huggingface.co/Qdrant/clip-ViT-B-32-vision/resolve/a636590e595dbbd798647c9dd4550d5652fba969/model.onnx";
const TEXT_MODEL_URL: &str = "https://huggingface.co/Qdrant/clip-ViT-B-32-text/resolve/48ca1db27cb4063eb311ec2aa7f087a808112876/model.onnx";
const TOKENIZER_URL: &str = "https://huggingface.co/Qdrant/clip-ViT-B-32-text/resolve/48ca1db27cb4063eb311ec2aa7f087a808112876/tokenizer.json";

const VISION_MODEL_SHA256: &str =
    "c68d3d9a200ddd2a8c8a5510b576d4c94d1ae383bf8b36dd8c084f94e1fb4d63";
const TEXT_MODEL_SHA256: &str = "4dbe762b11e36488304471e439cde89da053ad7acaddbf9e096745d142ec8d8b";
const TOKENIZER_SHA256: &str = "b68d571997a1f81bf521fb73806740ddb91e4ed6666cb6e996c066bb289cf55b";
const VISION_MODEL_SIZE: u64 = 351_686_194;
const TEXT_MODEL_SIZE: u64 = 254_102_519;
const TOKENIZER_SIZE: u64 = 2_224_147;

#[derive(Debug, Clone, Copy)]
struct ArtifactSpec {
    name: &'static str,
    url: &'static str,
    revision: &'static str,
    sha256: &'static str,
    size: u64,
}

const VISION_MODEL: ArtifactSpec = ArtifactSpec {
    name: "CLIP vision model",
    url: VISION_MODEL_URL,
    revision: VISION_MODEL_REVISION,
    sha256: VISION_MODEL_SHA256,
    size: VISION_MODEL_SIZE,
};
const TEXT_MODEL: ArtifactSpec = ArtifactSpec {
    name: "CLIP text model",
    url: TEXT_MODEL_URL,
    revision: TEXT_MODEL_REVISION,
    sha256: TEXT_MODEL_SHA256,
    size: TEXT_MODEL_SIZE,
};
const TOKENIZER: ArtifactSpec = ArtifactSpec {
    name: "CLIP tokenizer",
    url: TOKENIZER_URL,
    revision: TEXT_MODEL_REVISION,
    sha256: TOKENIZER_SHA256,
    size: TOKENIZER_SIZE,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ArtifactIdentity {
    size: u64,
    modified_unix_ns: u128,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct VerifiedArtifactManifest {
    version: u32,
    revision: String,
    sha256: String,
    expected_size: u64,
    identity: ArtifactIdentity,
}

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

pub(crate) fn timing_metadata() -> ModelMetadata {
    ModelMetadata {
        contract: MODEL_CONTRACT,
        vision_revision: VISION_MODEL_REVISION,
        vision_sha256: VISION_MODEL_SHA256,
        text_revision: TEXT_MODEL_REVISION,
        text_sha256: TEXT_MODEL_SHA256,
        tokenizer_sha256: TOKENIZER_SHA256,
    }
}

pub(crate) fn embedding_contract() -> EmbeddingContract {
    EmbeddingContract {
        image: IMAGE_EMBEDDING_CONTRACT,
        query: QUERY_EMBEDDING_CONTRACT,
    }
}

pub(crate) fn ensure_vision_artifacts(
    paths: &ModelPaths,
    on_event: &mut impl FnMut(ArtifactEvent),
    timing: &mut TimingRecorder,
    verification: ArtifactVerification,
) -> Result<(), VisionGrepError> {
    ensure_artifact(
        VISION_MODEL,
        &paths.vision_model,
        on_event,
        timing,
        verification,
    )
}

pub(crate) fn ensure_text_artifacts(
    paths: &ModelPaths,
    on_event: &mut impl FnMut(ArtifactEvent),
    timing: &mut TimingRecorder,
    verification: ArtifactVerification,
) -> Result<(), VisionGrepError> {
    ensure_artifact(
        TEXT_MODEL,
        &paths.text_model,
        on_event,
        timing,
        verification,
    )?;
    ensure_artifact(TOKENIZER, &paths.tokenizer, on_event, timing, verification)?;

    Ok(())
}

fn cache_dir() -> Result<PathBuf, VisionGrepError> {
    if let Some(cache_home) = std::env::var_os("XDG_CACHE_HOME").filter(|path| !path.is_empty()) {
        return Ok(PathBuf::from(cache_home).join("visiongrep"));
    }
    let home = std::env::var_os("HOME").ok_or(VisionGrepError::HomeDirectory)?;
    Ok(PathBuf::from(home).join(".cache").join("visiongrep"))
}

/// Uses an identity-bound install manifest on the common path and hashes on every uncertain path.
fn ensure_artifact(
    spec: ArtifactSpec,
    destination: &Path,
    on_event: &mut impl FnMut(ArtifactEvent),
    timing: &mut TimingRecorder,
    verification: ArtifactVerification,
) -> Result<(), VisionGrepError> {
    if verification == ArtifactVerification::Fast && verified_manifest_matches(spec, destination)? {
        return Ok(());
    }

    let validation_started = timing.start();
    let valid = if destination.exists() {
        validate_artifact(spec, destination)?
    } else {
        false
    };
    timing.record(Phase::ArtifactValidation, validation_started);
    if valid {
        write_verified_manifest(spec, destination)?;
        return Ok(());
    }

    let download_started = timing.start();
    let result = download_artifact(spec, destination, on_event);
    timing.record(Phase::ArtifactDownload, download_started);
    result
}

fn validate_artifact(spec: ArtifactSpec, destination: &Path) -> Result<bool, VisionGrepError> {
    let metadata = destination
        .metadata()
        .map_err(|source| VisionGrepError::ArtifactFile {
            operation: "reading artifact metadata",
            path: destination.to_owned(),
            source,
        })?;
    if metadata.len() != spec.size {
        return Ok(false);
    }
    Ok(sha256_file(destination)? == spec.sha256)
}

fn verified_manifest_matches(
    spec: ArtifactSpec,
    destination: &Path,
) -> Result<bool, VisionGrepError> {
    if !destination.exists() {
        return Ok(false);
    }
    let Some(identity) = artifact_identity(destination)? else {
        return Ok(false);
    };
    let marker_path = verified_manifest_path(destination);
    let manifest = match File::open(&marker_path) {
        Ok(file) => serde_json::from_reader::<_, VerifiedArtifactManifest>(file).ok(),
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => None,
        Err(source) => {
            return Err(VisionGrepError::ArtifactFile {
                operation: "opening verified artifact manifest",
                path: marker_path,
                source,
            });
        }
    };

    Ok(manifest.is_some_and(|manifest| {
        manifest.version == VERIFIED_MANIFEST_VERSION
            && manifest.revision == spec.revision
            && manifest.sha256 == spec.sha256
            && manifest.expected_size == spec.size
            && manifest.identity == identity
    }))
}

fn artifact_identity(path: &Path) -> Result<Option<ArtifactIdentity>, VisionGrepError> {
    let metadata = path
        .metadata()
        .map_err(|source| VisionGrepError::ArtifactFile {
            operation: "reading artifact identity",
            path: path.to_owned(),
            source,
        })?;
    let modified = metadata
        .modified()
        .map_err(|source| VisionGrepError::ArtifactFile {
            operation: "reading artifact modification time",
            path: path.to_owned(),
            source,
        })?;
    let Ok(modified) = modified.duration_since(UNIX_EPOCH) else {
        return Ok(None);
    };
    Ok(Some(ArtifactIdentity {
        size: metadata.len(),
        modified_unix_ns: modified.as_nanos(),
    }))
}

fn write_verified_manifest(spec: ArtifactSpec, destination: &Path) -> Result<(), VisionGrepError> {
    let Some(identity) = artifact_identity(destination)? else {
        return Ok(());
    };
    let manifest = VerifiedArtifactManifest {
        version: VERIFIED_MANIFEST_VERSION,
        revision: spec.revision.to_owned(),
        sha256: spec.sha256.to_owned(),
        expected_size: spec.size,
        identity,
    };
    let marker_path = verified_manifest_path(destination);
    let parent =
        marker_path
            .parent()
            .ok_or_else(|| VisionGrepError::ArtifactDestinationWithoutParent {
                path: marker_path.clone(),
            })?;
    let mut temporary =
        NamedTempFile::new_in(parent).map_err(|source| VisionGrepError::ArtifactFile {
            operation: "creating verified artifact manifest",
            path: marker_path.clone(),
            source,
        })?;
    serde_json::to_writer(temporary.as_file_mut(), &manifest).map_err(|source| {
        VisionGrepError::ArtifactManifestSerialize {
            path: marker_path.clone(),
            source,
        }
    })?;
    temporary
        .as_file_mut()
        .flush()
        .map_err(|source| VisionGrepError::ArtifactFile {
            operation: "flushing verified artifact manifest",
            path: marker_path.clone(),
            source,
        })?;
    temporary
        .as_file()
        .sync_all()
        .map_err(|source| VisionGrepError::ArtifactFile {
            operation: "syncing verified artifact manifest",
            path: marker_path.clone(),
            source,
        })?;
    temporary
        .persist(&marker_path)
        .map_err(|error| VisionGrepError::ArtifactFile {
            operation: "installing verified artifact manifest",
            path: marker_path,
            source: error.error,
        })?;
    Ok(())
}

fn verified_manifest_path(destination: &Path) -> PathBuf {
    let mut path = OsString::from(destination.as_os_str());
    path.push(VERIFIED_MANIFEST_SUFFIX);
    PathBuf::from(path)
}

/// Streams an artifact into a sibling temporary file before replacing the final destination.
///
/// Transfer events keep installation independent of terminal presentation. The destination remains
/// untouched until the complete response is written, so a partial download cannot be mistaken for
/// a usable model on the next run.
fn download_artifact(
    spec: ArtifactSpec,
    destination: &Path,
    on_event: &mut impl FnMut(ArtifactEvent),
) -> Result<(), VisionGrepError> {
    on_event(ArtifactEvent::DownloadStarted {
        artifact: spec.name,
    });
    let mut response = reqwest::blocking::get(spec.url)
        .and_then(reqwest::blocking::Response::error_for_status)
        .map_err(|source| VisionGrepError::DownloadRequest {
            artifact: spec.name,
            url: spec.url.to_owned(),
            source,
        })?;
    let content_length = response.content_length();
    install_download(spec, &mut response, content_length, destination, on_event)
}

fn install_download(
    spec: ArtifactSpec,
    response: &mut impl Read,
    content_length: Option<u64>,
    destination: &Path,
    on_event: &mut impl FnMut(ArtifactEvent),
) -> Result<(), VisionGrepError> {
    if let Some(bytes) = content_length {
        if bytes != spec.size {
            return Err(VisionGrepError::ArtifactSize {
                file: destination.to_owned(),
                expected: spec.size,
                actual: bytes,
            });
        }
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
    let mut hasher = Sha256::new();
    let mut total_bytes = 0_u64;
    loop {
        let read = response
            .read(&mut buffer)
            .map_err(|source| VisionGrepError::DownloadRead {
                artifact: spec.name,
                url: spec.url.to_owned(),
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
        hasher.update(&buffer[..read]);
        total_bytes = total_bytes.checked_add(bytes).ok_or_else(|| {
            VisionGrepError::Io(std::io::Error::other("artifact byte count overflow"))
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

    if total_bytes != spec.size {
        return Err(VisionGrepError::ArtifactSize {
            file: destination.to_owned(),
            expected: spec.size,
            actual: total_bytes,
        });
    }

    let actual = format!("{:x}", hasher.finalize());
    if actual != spec.sha256 {
        return Err(VisionGrepError::Checksum {
            file: destination.to_owned(),
            expected: spec.sha256.to_owned(),
            actual,
        });
    }

    temporary
        .as_file()
        .sync_all()
        .map_err(|source| VisionGrepError::ArtifactFile {
            operation: "syncing verified download",
            path: temporary.path().to_owned(),
            source,
        })?;

    temporary
        .persist(destination)
        .map_err(|error| VisionGrepError::ArtifactFile {
            operation: "installing verified download",
            path: destination.to_owned(),
            source: error.error,
        })?;
    write_verified_manifest(spec, destination)?;
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
    use std::io::{Cursor, ErrorKind};

    use super::*;

    const VERIFIED_SHA256: &str =
        "2127de9293abf1503418b9f78b3d530cdd2263417064815ee46b7ecdf1215ddc";
    const GOOD_SHA256: &str = "770e607624d689265ca6c44884d0807d9b054d23c473c106c72be9de08b7376c";

    fn spec(revision: &'static str, sha256: &'static str, size: u64) -> ArtifactSpec {
        ArtifactSpec {
            name: "test artifact",
            url: "https://example.invalid/artifact",
            revision,
            sha256,
            size,
        }
    }

    #[test]
    fn download_accepts_a_response_without_content_length() {
        const BODY: &[u8] = b"verified artifact";
        let mut response = Cursor::new(BODY);
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        let mut events = Vec::new();

        install_download(
            spec("revision-a", VERIFIED_SHA256, BODY.len() as u64),
            &mut response,
            None,
            &destination,
            &mut |event| events.push(event),
        )
        .unwrap();

        assert_eq!(fs::read(&destination).unwrap(), BODY);
        assert!(
            verified_manifest_matches(
                spec("revision-a", VERIFIED_SHA256, BODY.len() as u64),
                &destination,
            )
            .unwrap()
        );
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
        let mut timing = TimingRecorder::disabled(timing_metadata());
        let spec = spec("revision-a", VERIFIED_SHA256, 17);
        write_verified_manifest(spec, &destination).unwrap();

        ensure_artifact(
            spec,
            &destination,
            &mut |event| events.push(event),
            &mut timing,
            ArtifactVerification::Fast,
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
            spec("revision-a", GOOD_SHA256, 3),
            &mut response,
            Some(3),
            &destination,
            &mut |_| {},
        )
        .unwrap_err();

        assert!(matches!(error, VisionGrepError::Checksum { .. }));
        assert_eq!(fs::read(destination).unwrap(), b"previous artifact");
        assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1);
    }

    #[test]
    fn stale_manifest_falls_back_to_full_verification_and_is_replaced() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        fs::write(&destination, b"verified artifact").unwrap();
        let old_spec = spec("old-revision", VERIFIED_SHA256, 17);
        let current_spec = spec("current-revision", VERIFIED_SHA256, 17);
        write_verified_manifest(old_spec, &destination).unwrap();
        let mut timing = TimingRecorder::disabled(timing_metadata());

        ensure_artifact(
            current_spec,
            &destination,
            &mut |_| {},
            &mut timing,
            ArtifactVerification::Fast,
        )
        .unwrap();

        assert!(verified_manifest_matches(current_spec, &destination).unwrap());
    }

    #[test]
    fn explicit_full_verification_detects_corruption_even_with_a_marker() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        let spec = spec("revision-a", VERIFIED_SHA256, 17);
        fs::write(&destination, b"verified artifact").unwrap();
        write_verified_manifest(spec, &destination).unwrap();
        fs::write(&destination, b"corrupted-artifac").unwrap();

        assert!(!validate_artifact(spec, &destination).unwrap());
    }

    #[test]
    fn malformed_manifest_is_never_trusted() {
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        let spec = spec("revision-a", VERIFIED_SHA256, 17);
        fs::write(&destination, b"verified artifact").unwrap();
        fs::write(verified_manifest_path(&destination), b"not json").unwrap();

        assert!(!verified_manifest_matches(spec, &destination).unwrap());
    }

    #[test]
    fn interrupted_install_preserves_destination_and_never_writes_marker() {
        struct InterruptedReader {
            first_read: bool,
        }

        impl Read for InterruptedReader {
            fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
                if self.first_read {
                    return Err(std::io::Error::new(
                        ErrorKind::ConnectionReset,
                        "interrupted",
                    ));
                }
                self.first_read = true;
                buffer[..2].copy_from_slice(b"pa");
                Ok(2)
            }
        }

        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("artifact.bin");
        fs::write(&destination, b"previous artifact").unwrap();
        let mut response = InterruptedReader { first_read: false };

        let error = install_download(
            spec("revision-a", GOOD_SHA256, 4),
            &mut response,
            None,
            &destination,
            &mut |_| {},
        )
        .unwrap_err();

        assert!(matches!(error, VisionGrepError::DownloadRead { .. }));
        assert_eq!(fs::read(&destination).unwrap(), b"previous artifact");
        assert!(!verified_manifest_path(&destination).exists());
    }
}
