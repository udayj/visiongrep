use std::path::PathBuf;

use ort::session::builder::SessionBuilder;

#[derive(Debug, thiserror::Error)]
pub(crate) enum EmbeddingError {
    #[error("expected {expected} values, got {actual}")]
    Dimension { expected: usize, actual: usize },

    #[error("expected {expected} bytes, got {actual}")]
    ByteLength { expected: usize, actual: usize },

    #[error("contains a non-finite value")]
    NonFinite,

    #[error("has invalid L2 norm {norm}")]
    Norm { norm: f32 },
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum VisionGrepError {
    #[error("failed to download {artifact} from {url}: {source}")]
    DownloadRequest {
        artifact: &'static str,
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("failed while reading download stream for {artifact} from {url}: {source}")]
    DownloadRead {
        artifact: &'static str,
        url: String,
        #[source]
        source: std::io::Error,
    },

    #[error("checksum mismatch for {file}: expected {expected}, got {actual}")]
    Checksum {
        file: PathBuf,
        expected: String,
        actual: String,
    },

    #[error("artifact destination has no parent directory: {path}")]
    ArtifactDestinationWithoutParent { path: PathBuf },

    #[error("failed while {operation} at {path}: {source}")]
    ArtifactFile {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("failed to decode image {path}: {source}")]
    ImageDecode {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },

    #[error(
        "image {path} is too large to process safely ({width}x{height}, {decoded_bytes} decoded bytes, approximately {estimated_working_bytes} working bytes)"
    )]
    ImageTooLarge {
        path: PathBuf,
        width: u32,
        height: u32,
        decoded_bytes: u64,
        estimated_working_bytes: u64,
    },

    #[error("image {path} has invalid dimensions {width}x{height}")]
    InvalidImageDimensions {
        path: PathBuf,
        width: u32,
        height: u32,
    },

    #[error("failed to read image metadata for {path}: {source}")]
    ImageMetadata {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("search path does not exist: {path}")]
    SearchPathMissing { path: PathBuf },

    #[error("search path is not a directory: {path}")]
    SearchPathNotDirectory { path: PathBuf },

    #[error("failed to resolve search path {path}: {source}")]
    SearchPathResolve {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("discovered image {path} is outside search root {root}")]
    ImageOutsideSearchRoot { path: PathBuf, root: PathBuf },

    #[error("model file missing: {path}")]
    ModelMissing { path: PathBuf },

    #[error("ONNX Runtime error: {0}")]
    Inference(#[from] ort::Error),

    #[error("ONNX session builder error: {source}")]
    SessionBuilder {
        #[source]
        source: Box<ort::Error<SessionBuilder>>,
    },

    #[error("model did not return an output tensor")]
    MissingModelOutput,

    #[error("model returned output shape {actual:?}; expected [1, {expected}]")]
    UnexpectedModelOutputShape { expected: usize, actual: Vec<usize> },

    #[error("index error: {0}")]
    Index(#[from] rusqlite::Error),

    #[error("image index version {found} is newer than the supported version {supported}")]
    IndexVersionTooNew { found: i64, supported: i64 },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to serialize JSON results: {source}")]
    JsonOutput {
        #[source]
        source: serde_json::Error,
    },

    #[error("cannot represent non-UTF-8 path in JSON: {path}")]
    NonUtf8JsonPath { path: PathBuf },

    #[error("failed to locate home directory")]
    HomeDirectory,

    #[error("failed to load tokenizer from {path}: {source}")]
    TokenizerLoad {
        path: PathBuf,
        #[source]
        source: tokenizers::Error,
    },

    #[error("failed to encode query with tokenizer: {source}")]
    TokenizerEncode {
        query: String,
        #[source]
        source: tokenizers::Error,
    },

    #[error("model produced an invalid {kind} embedding: {source}")]
    InvalidModelEmbedding {
        kind: &'static str,
        #[source]
        source: EmbeddingError,
    },

    #[error("cached embedding for {path} is invalid: {source}")]
    InvalidCachedImageEmbedding {
        path: PathBuf,
        #[source]
        source: EmbeddingError,
    },

    #[error("cached query embedding is invalid: {source}")]
    InvalidCachedQueryEmbedding {
        #[source]
        source: EmbeddingError,
    },

    #[error("image modification time is outside the supported range: {path}")]
    ImageTimestampOutOfRange { path: PathBuf },

    #[error("image file size is outside the supported range: {path}")]
    ImageSizeOutOfRange { path: PathBuf },

    #[error("numeric conversion failed while {context}: {source}")]
    NumericConversion {
        context: &'static str,
        #[source]
        source: std::num::TryFromIntError,
    },
}

impl VisionGrepError {
    pub(crate) fn is_broken_pipe(&self) -> bool {
        match self {
            Self::Io(source) => source.kind() == std::io::ErrorKind::BrokenPipe,
            Self::JsonOutput { source } => {
                source.io_error_kind() == Some(std::io::ErrorKind::BrokenPipe)
            }
            _ => false,
        }
    }
}
