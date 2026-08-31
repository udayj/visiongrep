use std::path::Path;

use ndarray::{Array2, Array4};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;
use tokenizers::{PaddingParams, PaddingStrategy, TruncationParams};

use super::artifacts::ModelPaths;
use crate::embedding::EMBEDDING_DIM;
use crate::error::VisionGrepError;
use crate::timing::{Phase, TimingRecorder};

const TEXT_TOKENS: usize = 77;
const PAD_TOKEN_ID: u32 = 1;
const PAD_TOKEN: &str = "<|endoftext|>";

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
    pub(crate) fn run(
        &mut self,
        query: &str,
        timing: &mut TimingRecorder,
    ) -> Result<Vec<f32>, VisionGrepError> {
        let tokenization_started = timing.start();
        let (input_ids, attention_mask) = tokenize(query, &mut self.tokenizer)?;
        timing.record(Phase::TextTokenization, tokenization_started);
        let input_ids = TensorRef::from_array_view(&input_ids)?;
        let attention_mask = TensorRef::from_array_view(&attention_mask)?;
        let inference_started = timing.start();
        let outputs = self.session.run(ort::inputs![input_ids, attention_mask])?;
        let embedding = extract_embedding(&outputs)?;
        timing.record(Phase::TextInference, inference_started);
        Ok(embedding)
    }
}

fn tokenize(
    query: &str,
    tokenizer: &mut tokenizers::Tokenizer,
) -> Result<(Array2<i64>, Array2<i64>), VisionGrepError> {
    configure_tokenizer(tokenizer).map_err(|source| VisionGrepError::TokenizerEncode {
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

fn configure_tokenizer(tokenizer: &mut tokenizers::Tokenizer) -> Result<(), tokenizers::Error> {
    // These values mirror the pinned Qdrant/FastEmbed contract: the model configuration supplies
    // pad ID 1, while the tokenizer configuration names the end-of-text token for padded slots.
    tokenizer.with_padding(Some(PaddingParams {
        strategy: PaddingStrategy::Fixed(TEXT_TOKENS),
        pad_id: PAD_TOKEN_ID,
        pad_token: PAD_TOKEN.to_owned(),
        ..PaddingParams::default()
    }));
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length: TEXT_TOKENS,
            ..TruncationParams::default()
        }))
        .map(|_| ())
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

#[cfg(test)]
mod tests {
    use serde::Deserialize;
    use tokenizers::{Tokenizer, models::bpe::BPE};

    use super::*;

    const GOLDEN_VECTORS: &str = include_str!("../../tests/fixtures/clip_text_golden.json");

    #[derive(Deserialize)]
    struct GoldenFixture {
        model_revision: String,
        model_sha256: String,
        tokenizer_sha256: String,
        reference: String,
        cases: Vec<GoldenCase>,
    }

    #[derive(Deserialize)]
    struct GoldenCase {
        query: String,
        embedding_le_hex: String,
    }

    fn decode_hex(value: &str) -> Vec<u8> {
        assert_eq!(value.len() % 2, 0);
        value
            .as_bytes()
            .chunks_exact(2)
            .map(|pair| {
                let pair = std::str::from_utf8(pair).unwrap();
                u8::from_str_radix(pair, 16).unwrap()
            })
            .collect()
    }

    #[test]
    fn tokenizer_configuration_matches_the_pinned_model_contract() {
        let mut tokenizer = Tokenizer::new(BPE::default());

        configure_tokenizer(&mut tokenizer).unwrap();

        let padding = tokenizer.get_padding().unwrap();
        assert!(matches!(
            padding.strategy,
            PaddingStrategy::Fixed(TEXT_TOKENS)
        ));
        assert_eq!(padding.pad_id, PAD_TOKEN_ID);
        assert_eq!(padding.pad_token, PAD_TOKEN);
        assert_eq!(tokenizer.get_truncation().unwrap().max_length, TEXT_TOKENS);
    }

    /// Run explicitly after installing the pinned model artifacts; ordinary unit tests stay fast
    /// and never initiate a 250 MB download.
    #[test]
    #[ignore = "requires the pinned CLIP text model and tokenizer in the visiongrep cache"]
    fn text_embeddings_match_qdrant_fastembed_golden_vectors() {
        let fixture: GoldenFixture = serde_json::from_str(GOLDEN_VECTORS).unwrap();
        assert_eq!(
            fixture.model_revision,
            "48ca1db27cb4063eb311ec2aa7f087a808112876"
        );
        assert_eq!(
            fixture.model_sha256,
            "4dbe762b11e36488304471e439cde89da053ad7acaddbf9e096745d142ec8d8b"
        );
        assert_eq!(
            fixture.tokenizer_sha256,
            "b68d571997a1f81bf521fb73806740ddb91e4ed6666cb6e996c066bb289cf55b"
        );
        assert!(fixture.reference.contains("Qdrant/FastEmbed"));

        let paths = crate::model::model_paths().unwrap();
        let mut session = TextSession::load(&paths).unwrap();
        let mut timing = crate::timing::TimingRecorder::disabled(crate::model::timing_metadata());
        for case in fixture.cases {
            let actual =
                crate::embedding::embed_text(&case.query, &mut session, &mut timing).unwrap();
            let expected = crate::embedding::NormalizedEmbedding::from_le_bytes(&decode_hex(
                &case.embedding_le_hex,
            ))
            .unwrap();
            let maximum_error = actual
                .as_slice()
                .iter()
                .zip(expected.as_slice())
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);

            assert!(
                maximum_error <= 1e-5,
                "query {:?} exceeded the reference tolerance: {maximum_error}",
                case.query
            );
        }
    }
}
