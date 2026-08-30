use std::path::Path;

use ndarray::{Array2, Array4};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;
use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

use super::artifacts::ModelPaths;
use crate::embedding::EMBEDDING_DIM;
use crate::error::VisionGrepError;

const TEXT_TOKENS: usize = 77;

pub(crate) struct VisionSession {
    session: Session,
}

pub(crate) struct TextSession {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
}

impl VisionSession {
    pub(crate) fn load(paths: &ModelPaths) -> Result<Self, VisionGrepError> {
        Ok(Self {
            session: load_session(&paths.vision_model)?,
        })
    }

    pub(crate) fn run(&mut self, input: &Array4<f32>) -> Result<Vec<f32>, VisionGrepError> {
        let input = TensorRef::from_array_view(input)?;
        let outputs = self.session.run(ort::inputs![input])?;
        extract_embedding(&outputs)
    }
}

impl TextSession {
    pub(crate) fn load(paths: &ModelPaths) -> Result<Self, VisionGrepError> {
        require_file(&paths.text_model)?;
        require_file(&paths.tokenizer)?;

        let session = load_session(&paths.text_model)?;
        let tokenizer = tokenizers::Tokenizer::from_file(&paths.tokenizer).map_err(|source| {
            VisionGrepError::TokenizerLoad {
                path: paths.tokenizer.clone(),
                source,
            }
        })?;

        Ok(Self { session, tokenizer })
    }

    /// Tokenizes according to the model's fixed input contract and performs one text inference.
    pub(crate) fn run(&mut self, query: &str) -> Result<Vec<f32>, VisionGrepError> {
        let (input_ids, attention_mask) = tokenize(query, &mut self.tokenizer)?;
        let input_ids = TensorRef::from_array_view(&input_ids)?;
        let attention_mask = TensorRef::from_array_view(&attention_mask)?;
        let outputs = self.session.run(ort::inputs![input_ids, attention_mask])?;
        extract_embedding(&outputs)
    }
}

fn tokenize(
    query: &str,
    tokenizer: &mut tokenizers::Tokenizer,
) -> Result<(Array2<i64>, Array2<i64>), VisionGrepError> {
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(TEXT_TOKENS),
        ..PaddingParams::default()
    }));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: TEXT_TOKENS,
            ..TruncationParams::default()
        }))
        .map_err(|source| VisionGrepError::TokenizerEncode {
            query: query.to_owned(),
            source,
        })?;

    let encoding =
        tokenizer
            .encode(query, true)
            .map_err(|source| VisionGrepError::TokenizerEncode {
                query: query.to_owned(),
                source,
            })?;
    let ids = encoding
        .get_ids()
        .iter()
        .map(|id| i64::from(*id))
        .collect::<Vec<_>>();
    let mask = encoding
        .get_attention_mask()
        .iter()
        .map(|id| i64::from(*id))
        .collect::<Vec<_>>();

    Array2::from_shape_vec((1, TEXT_TOKENS), ids)
        .and_then(|ids| Array2::from_shape_vec((1, TEXT_TOKENS), mask).map(|mask| (ids, mask)))
        .map_err(|source| {
            VisionGrepError::Io(std::io::Error::other(format!(
                "tokenizer returned unexpected sequence length: {source}"
            )))
        })
}

fn extract_embedding(
    outputs: &ort::session::SessionOutputs<'_>,
) -> Result<Vec<f32>, VisionGrepError> {
    let output_value = outputs
        .values()
        .next()
        .ok_or(VisionGrepError::MissingModelOutput)?;
    let output = output_value.try_extract_array::<f32>()?;
    if output.shape() != [1, EMBEDDING_DIM] {
        return Err(VisionGrepError::UnexpectedModelOutputShape {
            expected: EMBEDDING_DIM,
            actual: output.shape().to_vec(),
        });
    }
    Ok(output.iter().copied().collect())
}

/// Creates an optimized ONNX Runtime session for one model artifact.
///
/// Session ownership stays with the typed vision or text wrapper so runtime resources are released
/// through ordinary RAII and are never hidden in global state.
fn load_session(path: &Path) -> Result<Session, VisionGrepError> {
    require_file(path)?;
    Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|source| VisionGrepError::SessionBuilder {
            source: Box::new(source),
        })?
        .commit_from_file(path)
        .map_err(VisionGrepError::Inference)
}

fn require_file(path: &Path) -> Result<(), VisionGrepError> {
    if path.exists() {
        Ok(())
    } else {
        Err(VisionGrepError::ModelMissing {
            path: path.to_owned(),
        })
    }
}
