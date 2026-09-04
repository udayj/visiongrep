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
const PAD_TOKEN_ID: u32 = 0;
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

    #[cfg(test)]
    pub(crate) fn run(&mut self, input: &Array4<f32>) -> Result<Vec<f32>, VisionGrepError> {
        let mut embeddings = self.run_batch(input)?;
        if embeddings.len() != 1 {
            return Err(VisionGrepError::UnexpectedModelOutputShape {
                expected: vec![1, EMBEDDING_DIM],
                actual: vec![embeddings.len(), EMBEDDING_DIM],
            });
        }
        Ok(embeddings.remove(0))
    }

    pub(crate) fn run_batch(
        &mut self,
        input: &Array4<f32>,
    ) -> Result<Vec<Vec<f32>>, VisionGrepError> {
        let batch_size = input.shape()[0];
        let input = TensorRef::from_array_view(input)?;
        let outputs = self.session.run(ort::inputs![input])?;
        extract_embeddings(&outputs, batch_size)
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
        let input_ids = tokenize(query, &mut self.tokenizer)?;
        timing.record(Phase::TextTokenization, tokenization_started);
        let input_ids = TensorRef::from_array_view(&input_ids)?;
        let inference_started = timing.start();
        let outputs = self.session.run(ort::inputs![input_ids])?;
        let mut embeddings = extract_embeddings(&outputs, 1)?;
        let embedding = embeddings.remove(0);
        timing.record(Phase::TextInference, inference_started);
        Ok(embedding)
    }
}

fn tokenize(
    query: &str,
    tokenizer: &mut tokenizers::Tokenizer,
) -> Result<Array2<i64>, VisionGrepError> {
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
    Array2::from_shape_vec((1, TEXT_TOKENS), ids).map_err(|source| {
        VisionGrepError::Io(std::io::Error::other(format!(
            "tokenizer returned unexpected sequence length: {source}"
        )))
    })
}

fn configure_tokenizer(tokenizer: &mut tokenizers::Tokenizer) -> Result<(), tokenizers::Error> {
    // OpenCLIP pads with zero after the end-of-text token; the exported text model pools at the
    // highest token ID and therefore sees the same EOT position as the reference implementation.
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

fn extract_embeddings(
    outputs: &ort::session::SessionOutputs<'_>,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>, VisionGrepError> {
    let output_value = outputs
        .values()
        .next()
        .ok_or(VisionGrepError::MissingModelOutput)?;
    let output = output_value.try_extract_array::<f32>()?;
    if output.shape() != [batch_size, EMBEDDING_DIM] {
        return Err(VisionGrepError::UnexpectedModelOutputShape {
            expected: vec![batch_size, EMBEDDING_DIM],
            actual: output.shape().to_vec(),
        });
    }
    let values = output.iter().copied().collect::<Vec<_>>();
    let (embeddings, _) = values.as_chunks::<EMBEDDING_DIM>();
    Ok(embeddings
        .iter()
        .map(|embedding| embedding.to_vec())
        .collect())
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
    use std::time::Instant;

    use ort::value::ValueType;
    use serde::Deserialize;
    use tokenizers::{Tokenizer, models::bpe::BPE};

    use super::*;
    use crate::embedding::IMAGE_SIZE_USIZE;

    const GOLDEN_VECTORS: &str = include_str!("../../tests/fixtures/datacomp_golden.json");

    #[derive(Deserialize)]
    struct GoldenFixture {
        contract: GoldenContract,
        reference: String,
        queries: Vec<GoldenCase>,
    }

    #[derive(Deserialize)]
    struct GoldenContract {
        openclip_revision: String,
        rclip_model_revision: String,
        textual_onnx_sha256: String,
        tokenizer_sha256: String,
    }

    #[derive(Deserialize)]
    struct GoldenCase {
        query: String,
        token_ids: Vec<i64>,
        embedding_le_hex: String,
    }

    fn decode_hex(value: &str) -> Vec<u8> {
        assert_eq!(value.len() % 2, 0);
        let (pairs, _) = value.as_bytes().as_chunks::<2>();
        pairs
            .iter()
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

    /// The current encoder declares a dynamic batch dimension, which is required before the
    /// indexing pipeline may concatenate preprocessed image tensors.
    #[test]
    #[ignore = "requires the pinned CLIP vision model in the visiongrep cache"]
    fn vision_model_contract_supports_dynamic_batches() {
        let paths = crate::model::model_paths().unwrap();
        let mut session = VisionSession::load(&paths).unwrap();
        let input = session.session.inputs().first().unwrap();

        let ValueType::Tensor { shape, .. } = input.dtype() else {
            panic!("vision model input is not a tensor: {:?}", input.dtype());
        };
        assert_eq!(
            &shape[..],
            &[-1, 3, IMAGE_SIZE_USIZE as i64, IMAGE_SIZE_USIZE as i64]
        );

        let single = Array4::<f32>::zeros((1, 3, IMAGE_SIZE_USIZE, IMAGE_SIZE_USIZE));
        let batched = Array4::<f32>::zeros((2, 3, IMAGE_SIZE_USIZE, IMAGE_SIZE_USIZE));

        let expected = session.run(&single).unwrap();
        let actual = session.run_batch(&batched).unwrap();
        assert_eq!(actual.len(), 2);
        for embedding in actual {
            assert_eq!(embedding, expected);
        }
    }

    /// Measures the pinned encoder directly; run in release mode with immutable artifacts.
    #[test]
    #[ignore = "release benchmark requiring the pinned CLIP vision model"]
    fn vision_batch_size_matrix() {
        let paths = crate::model::model_paths().unwrap();
        let mut session = VisionSession::load(&paths).unwrap();
        let samples = std::env::var("VISIONGREP_BATCH_SAMPLES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(5);
        assert!(samples > 0);

        let mut reports = Vec::new();
        for batch_size in [1, 2, 4, 8, 16] {
            let input = Array4::from_shape_fn(
                (batch_size, 3, IMAGE_SIZE_USIZE, IMAGE_SIZE_USIZE),
                |(_, c, y, x)| {
                    ((c * IMAGE_SIZE_USIZE * IMAGE_SIZE_USIZE + y * IMAGE_SIZE_USIZE + x) % 257)
                        as f32
                        / 128.0
                        - 1.0
                },
            );
            let expected = session.run_batch(&input).unwrap();
            assert_eq!(expected.len(), batch_size);
            assert!(expected.windows(2).all(|pair| pair[0] == pair[1]));

            let mut elapsed_ms = Vec::with_capacity(samples);
            for _ in 0..samples {
                let started = Instant::now();
                let actual = session.run_batch(&input).unwrap();
                elapsed_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                assert_eq!(actual, expected);
            }
            elapsed_ms.sort_by(f64::total_cmp);
            let p95_index = ((samples as f64 * 0.95).ceil() as usize)
                .saturating_sub(1)
                .min(samples - 1);
            let median_ms = elapsed_ms[samples / 2];
            reports.push(serde_json::json!({
                "batch_size": batch_size,
                "samples": samples,
                "median_ms": median_ms,
                "p95_ms": elapsed_ms[p95_index],
                "median_images_per_second": batch_size as f64 * 1_000.0 / median_ms,
            }));
        }

        println!("{}", serde_json::to_string(&reports).unwrap());
    }

    /// Run explicitly after installing the pinned model artifacts; ordinary unit tests stay fast
    /// and never initiate a 250 MB download.
    #[test]
    #[ignore = "requires the pinned CLIP text model and tokenizer in the visiongrep cache"]
    fn text_embeddings_match_openclip_golden_vectors() {
        let fixture: GoldenFixture = serde_json::from_str(GOLDEN_VECTORS).unwrap();
        assert_eq!(
            fixture.contract.openclip_revision,
            "4afec35ffe57a943d569ff7ee888061830164da8"
        );
        assert_eq!(
            fixture.contract.rclip_model_revision,
            "17b9d07433aad73f70d338d8a1c7a4cef83887e0"
        );
        assert_eq!(
            fixture.contract.textual_onnx_sha256,
            "ee267cd64f0f77362670ae0140476ed51ee8c5a761d41636e09997f2fdddcacc"
        );
        assert_eq!(
            fixture.contract.tokenizer_sha256,
            "924691ac288e54409236115652ad4aa250f48203de50a9e4722a6ecd48d6804a"
        );
        assert!(fixture.reference.contains("open_clip_torch 3.3.0"));

        let paths = crate::model::model_paths().unwrap();
        let mut session = TextSession::load(&paths).unwrap();
        let mut timing = crate::timing::TimingRecorder::disabled(crate::model::timing_metadata());
        let mut report_maximum_error = 0.0_f32;
        let mut report_minimum_cosine = 1.0_f32;
        for case in fixture.queries {
            let actual_tokens = tokenize(&case.query, &mut session.tokenizer).unwrap();
            assert_eq!(actual_tokens.into_raw_vec_and_offset().0, case.token_ids);
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
            let cosine = actual.dot(&expected);
            report_maximum_error = report_maximum_error.max(maximum_error);
            report_minimum_cosine = report_minimum_cosine.min(cosine);

            assert!(
                maximum_error <= 1e-4,
                "query {:?} exceeded the reference tolerance: {maximum_error}",
                case.query
            );
        }
        println!(
            "{}",
            serde_json::json!({
                "maximum_absolute_error": report_maximum_error,
                "minimum_cosine": report_minimum_cosine,
                "token_ids_exact": true,
            })
        );
    }
}
