use std::path::PathBuf;

use ort::session::builder::SessionBuilder;

#[derive(Debug, thiserror::Error)]
pub(crate) enum VisionGrepError {
    #[error("failed to download {artifact} from {url}: {source}")]
    DownloadRequest {
        artifact: &'static str,
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("downloaded {artifact} from {url} without a valid content length")]
    MissingContentLength { artifact: &'static str, url: String },

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
        expected: &'static str,
        actual: String,
    },

    #[error("failed to decode image {path}: {source}")]
    ImageDecode {
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },

    #[error(
        "image {path} is too large to decode safely ({width}x{height}, {decoded_bytes} decoded bytes)"
    )]
    ImageTooLarge {
        path: PathBuf,
        width: u32,
        height: u32,
        decoded_bytes: u64,
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

    #[error("embedding for {kind} had invalid L2 norm {norm}")]
    InvalidEmbeddingNorm { kind: &'static str, norm: f32 },

    #[error("embedding blob for {path} has invalid byte length {len}")]
    InvalidEmbeddingBlob { path: PathBuf, len: usize },

    #[error("cached embedding for query has invalid byte length {len}")]
    InvalidQueryEmbeddingBlob { len: usize },

    #[error("cached embedding for {path} contains a non-finite value")]
    NonFiniteEmbedding { path: PathBuf },

    #[error("cached query embedding contains a non-finite value")]
    NonFiniteQueryEmbedding,

    #[error(
        "embedding dimensions differ for {path}: query has {query_len} values, image has {image_len}"
    )]
    EmbeddingDimensionMismatch {
        path: PathBuf,
        query_len: usize,
        image_len: usize,
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
