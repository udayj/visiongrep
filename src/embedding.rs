use std::path::Path;
use std::time::{Duration, Instant};

use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};
use ndarray::Array4;

use crate::error::{EmbeddingError, ImagePreparationError, VisionGrepError};
use crate::model::{TextSession, VisionSession};
use crate::pillow_resize;
use crate::timing::{Phase, TimingRecorder};

pub(crate) const IMAGE_SIZE: u32 = 256;
pub(crate) const IMAGE_SIZE_USIZE: usize = 256;
pub(crate) const EMBEDDING_DIM: usize = 512;
const EMBEDDING_BYTES: usize = EMBEDDING_DIM * std::mem::size_of::<f32>();
const NORMALIZED_NORM_TOLERANCE: f32 = 1e-3;
const MAX_IMAGE_PIXELS: u64 = 100_000_000;
const MAX_IMAGE_WORKING_BYTES: u64 = 512 * 1024 * 1024;
const RGB_CHANNELS: u64 = 3;
const PREPROCESSED_RGB_BYTES: u64 = 256 * 256 * RGB_CHANNELS;
const INPUT_TENSOR_BYTES: u64 = PREPROCESSED_RGB_BYTES * 4;
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

#[derive(Debug, Clone, Copy)]
pub(crate) struct EmbeddingContract {
    pub(crate) image: &'static str,
    pub(crate) query: &'static str,
}

pub(crate) struct PreparedImage {
    pub(crate) values: Vec<f32>,
    decoding_elapsed: Option<Duration>,
    preprocessing_elapsed: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct NormalizedEmbedding(Box<[f32; EMBEDDING_DIM]>);

impl NormalizedEmbedding {
    pub(crate) fn from_model_output(mut values: Vec<f32>) -> Result<Self, EmbeddingError> {
        validate_dimension(&values)?;
        if values.iter().any(|value| !value.is_finite()) {
            return Err(EmbeddingError::NonFinite);
        }

        let norm = l2_norm(&values);
        if !norm.is_finite() || norm <= f32::EPSILON {
            return Err(EmbeddingError::Norm { norm });
        }
        for value in &mut values {
            *value /= norm;
        }

        fixed_embedding(values)
    }

    pub(crate) fn from_le_bytes(bytes: &[u8]) -> Result<Self, EmbeddingError> {
        if bytes.len() != EMBEDDING_BYTES {
            return Err(EmbeddingError::ByteLength {
                expected: EMBEDDING_BYTES,
                actual: bytes.len(),
            });
        }

        let values = bytes
            .chunks_exact(std::mem::size_of::<f32>())
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        if values.iter().any(|value| !value.is_finite()) {
            return Err(EmbeddingError::NonFinite);
        }

        let norm = l2_norm(&values);
        if !norm.is_finite() || (norm - 1.0).abs() > NORMALIZED_NORM_TOLERANCE {
            return Err(EmbeddingError::Norm { norm });
        }

        fixed_embedding(values)
    }

    pub(crate) fn to_le_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(EMBEDDING_BYTES);
        for value in self.0.iter() {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes
    }

    pub(crate) fn dot(&self, other: &Self) -> f32 {
        self.0
            .iter()
            .zip(other.0.iter())
            .map(|(left, right)| left * right)
            .sum()
    }

    #[cfg(test)]
    pub(crate) fn as_slice(&self) -> &[f32] {
        self.0.as_slice()
    }
}

pub(crate) fn prepare_image(
    path: &Path,
    measure_timing: bool,
) -> Result<PreparedImage, ImagePreparationError> {
    let decoding_started = measure_timing.then(Instant::now);
    let image = load_image(path)?;
    let decoding_elapsed = decoding_started.map(|started| started.elapsed());
    let preprocessing_started = measure_timing.then(Instant::now);
    let values = preprocess_pixels(image);
    let preprocessing_elapsed = preprocessing_started.map(|started| started.elapsed());
    Ok(PreparedImage {
        values,
        decoding_elapsed,
        preprocessing_elapsed,
    })
}

pub(crate) fn embed_prepared_images(
    prepared: Vec<PreparedImage>,
    session: &mut VisionSession,
    timing: &mut TimingRecorder,
) -> Result<Vec<NormalizedEmbedding>, VisionGrepError> {
    if prepared.is_empty() {
        return Ok(Vec::new());
    }

    let batch_size = prepared.len();
    let mut values = Vec::with_capacity(batch_size * 3 * IMAGE_SIZE_USIZE * IMAGE_SIZE_USIZE);
    for image in prepared {
        timing.record_duration(Phase::ImageDecoding, image.decoding_elapsed);
        timing.record_duration(Phase::ImagePreprocessing, image.preprocessing_elapsed);
        values.extend(image.values);
    }
    let input = Array4::from_shape_vec((batch_size, 3, IMAGE_SIZE_USIZE, IMAGE_SIZE_USIZE), values)
        .map_err(|source| VisionGrepError::ImageBatchShape { source })?;
    let inference_started = timing.start();
    let embeddings = session.run_batch(&input)?;
    timing.record(Phase::VisionInference, inference_started);
    embeddings
        .into_iter()
        .map(|embedding| {
            NormalizedEmbedding::from_model_output(embedding).map_err(|source| {
                VisionGrepError::InvalidModelEmbedding {
                    kind: "image",
                    source,
                }
            })
        })
        .collect()
}

/// Converts a natural-language query into a normalized embedding in the same CLIP vector space.
pub(crate) fn embed_text(
    query: &str,
    session: &mut TextSession,
    timing: &mut TimingRecorder,
) -> Result<NormalizedEmbedding, VisionGrepError> {
    let embedding = session.run(query, timing)?;
    NormalizedEmbedding::from_model_output(embedding).map_err(|source| {
        VisionGrepError::InvalidModelEmbedding {
            kind: "text",
            source,
        }
    })
}

fn validate_dimension(values: &[f32]) -> Result<(), EmbeddingError> {
    if values.len() == EMBEDDING_DIM {
        Ok(())
    } else {
        Err(EmbeddingError::Dimension {
            expected: EMBEDDING_DIM,
            actual: values.len(),
        })
    }
}

fn fixed_embedding(values: Vec<f32>) -> Result<NormalizedEmbedding, EmbeddingError> {
    validate_dimension(&values)?;
    let actual = values.len();
    let values = values
        .into_boxed_slice()
        .try_into()
        .map_err(|_| EmbeddingError::Dimension {
            expected: EMBEDDING_DIM,
            actual,
        })?;
    Ok(NormalizedEmbedding(values))
}

fn l2_norm(values: &[f32]) -> f32 {
    values.iter().map(|value| value * value).sum::<f32>().sqrt()
}

/// Decodes an image only after checking its declared resource requirements and applies orientation.
///
/// Dimensions and decoded byte count are inspected before pixel allocation to limit unconstrained
/// decompression and pathological inputs. EXIF orientation is applied before cropping so the crop follows
/// the image's intended visual orientation.
fn load_image(path: &Path) -> Result<DynamicImage, ImagePreparationError> {
    let reader = ImageReader::open(path)
        .map_err(|source| ImagePreparationError::Decode {
            path: path.to_owned(),
            source: source.into(),
        })?
        .with_guessed_format()
        .map_err(|source| ImagePreparationError::Decode {
            path: path.to_owned(),
            source: source.into(),
        })?;
    let mut decoder = reader
        .into_decoder()
        .map_err(|source| ImagePreparationError::Decode {
            path: path.to_owned(),
            source,
        })?;
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 {
        return Err(ImagePreparationError::InvalidDimensions {
            path: path.to_owned(),
            width,
            height,
        });
    }
    let pixel_count = u64::from(width).saturating_mul(u64::from(height));
    let decoded_bytes = decoder.total_bytes();
    let estimated_working_bytes = estimate_working_bytes(width, height, decoded_bytes);
    if pixel_count > MAX_IMAGE_PIXELS || estimated_working_bytes > MAX_IMAGE_WORKING_BYTES {
        return Err(ImagePreparationError::TooLarge {
            path: path.to_owned(),
            width,
            height,
            decoded_bytes,
            estimated_working_bytes,
        });
    }

    let orientation = decoder
        .orientation()
        .map_err(|source| ImagePreparationError::Decode {
            path: path.to_owned(),
            source,
        })?;
    let mut image =
        DynamicImage::from_decoder(decoder).map_err(|source| ImagePreparationError::Decode {
            path: path.to_owned(),
            source,
        })?;
    image.apply_orientation(orientation);
    Ok(image)
}

/// Produces the channel-first, normalized `[3, 256, 256]` values expected by the vision model.
fn preprocess_pixels(image: DynamicImage) -> Vec<f32> {
    let resized = resize_and_center_crop(image);
    let mut input = vec![0.0; 3 * IMAGE_SIZE_USIZE * IMAGE_SIZE_USIZE];

    for (pixel_index, pixel) in resized.pixels().enumerate() {
        for channel in 0..3 {
            let value = f32::from(pixel[channel]) / 255.0;
            let channel_offset = channel * IMAGE_SIZE_USIZE * IMAGE_SIZE_USIZE;
            input[channel_offset + pixel_index] = (value - CLIP_MEAN[channel]) / CLIP_STD[channel];
        }
    }

    input
}

/// Matches OpenCLIP evaluation: resize the short edge, then take the centered 256px crop.
fn resize_and_center_crop(image: DynamicImage) -> RgbImage {
    let image = image.into_rgb8();
    let (resized_width, resized_height) = resized_dimensions(image.width(), image.height());
    let resized = pillow_resize::resize_rgb(&image, resized_width, resized_height);
    let left = round_half_to_even((resized_width - IMAGE_SIZE) / 2, resized_width - IMAGE_SIZE);
    let top = round_half_to_even(
        (resized_height - IMAGE_SIZE) / 2,
        resized_height - IMAGE_SIZE,
    );
    image::imageops::crop_imm(&resized, left, top, IMAGE_SIZE, IMAGE_SIZE).to_image()
}

fn resized_dimensions(width: u32, height: u32) -> (u32, u32) {
    if width < height {
        (
            IMAGE_SIZE,
            ((u64::from(height) * u64::from(IMAGE_SIZE)) / u64::from(width))
                .min(u64::from(u32::MAX)) as u32,
        )
    } else {
        (
            ((u64::from(width) * u64::from(IMAGE_SIZE)) / u64::from(height))
                .min(u64::from(u32::MAX)) as u32,
            IMAGE_SIZE,
        )
    }
}

fn round_half_to_even(floor: u32, numerator: u32) -> u32 {
    if numerator % 2 == 0 || floor % 2 == 0 {
        floor
    } else {
        floor + 1
    }
}

fn estimate_working_bytes(width: u32, height: u32, decoded_bytes: u64) -> u64 {
    let pixel_count = u64::from(width).saturating_mul(u64::from(height));
    let (resized_width, resized_height) = resized_dimensions(width, height);
    let resized_bytes = u64::from(resized_width)
        .saturating_mul(u64::from(resized_height))
        .saturating_mul(RGB_CHANNELS);
    // Orientation can temporarily duplicate the decoded image, while conversion may allocate RGB.
    decoded_bytes
        .saturating_mul(2)
        .saturating_add(pixel_count.saturating_mul(RGB_CHANNELS))
        .saturating_add(resized_bytes)
        .saturating_add(PREPROCESSED_RGB_BYTES)
        .saturating_add(INPUT_TENSOR_BYTES)
}

#[cfg(test)]
mod tests {
    use image::{ImageFormat, Rgb};
    use serde::Deserialize;

    use super::*;

    const DATACOMP_GOLDEN: &str = include_str!("../tests/fixtures/datacomp_golden.json");

    #[derive(Deserialize)]
    struct GoldenFixture {
        contract: GoldenContract,
        queries: Vec<GoldenQuery>,
        images: Vec<GoldenImage>,
    }

    #[derive(Deserialize)]
    struct GoldenContract {
        openclip_revision: String,
        visual_onnx_sha256: String,
    }

    #[derive(Deserialize)]
    struct GoldenImage {
        name: String,
        width: u32,
        height: u32,
        seed: u32,
        embedding_le_hex: String,
    }

    #[derive(Deserialize)]
    struct GoldenQuery {
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
    fn normalize_embedding_returns_unit_vector() {
        let mut values = vec![0.0; EMBEDDING_DIM];
        values[0] = 3.0;
        values[1] = 4.0;
        let embedding = NormalizedEmbedding::from_model_output(values).unwrap();

        assert!((embedding.as_slice()[0] - 0.6).abs() < f32::EPSILON);
        assert!((embedding.as_slice()[1] - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn normalize_embedding_rejects_zero_norm() {
        assert!(NormalizedEmbedding::from_model_output(vec![0.0; EMBEDDING_DIM]).is_err());
    }

    #[test]
    fn normalized_embedding_rejects_wrong_dimension() {
        assert!(NormalizedEmbedding::from_model_output(vec![1.0]).is_err());
    }

    #[test]
    fn cached_embedding_requires_unit_norm() {
        let values = vec![1.0_f32; EMBEDDING_DIM];
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();

        assert!(NormalizedEmbedding::from_le_bytes(&bytes).is_err());
    }

    #[test]
    fn preprocessing_center_crops_landscape_images() {
        let image = RgbImage::from_fn(4, 2, |x, _| {
            if x == 0 || x == 3 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 255, 0])
            }
        });

        let resized = resize_and_center_crop(DynamicImage::ImageRgb8(image));

        assert_eq!(resized.dimensions(), (IMAGE_SIZE, IMAGE_SIZE));
        assert!((0..IMAGE_SIZE).all(|y| {
            let pixel = resized.get_pixel(IMAGE_SIZE / 2, y);
            pixel[1] > pixel[0]
        }));
    }

    #[test]
    fn preprocessing_center_crops_portrait_images() {
        let image = RgbImage::from_fn(2, 4, |_, y| {
            if y == 0 || y == 3 {
                image::Rgb([255, 0, 0])
            } else {
                image::Rgb([0, 255, 0])
            }
        });

        let resized = resize_and_center_crop(DynamicImage::ImageRgb8(image));

        assert_eq!(resized.dimensions(), (IMAGE_SIZE, IMAGE_SIZE));
        assert!((0..IMAGE_SIZE).all(|x| {
            let pixel = resized.get_pixel(x, IMAGE_SIZE / 2);
            pixel[1] > pixel[0]
        }));
    }

    #[test]
    fn preprocessing_scales_tiny_images() {
        let image = DynamicImage::new_rgb8(1, 1);

        assert_eq!(
            resize_and_center_crop(image).dimensions(),
            (IMAGE_SIZE, IMAGE_SIZE)
        );
    }

    #[test]
    fn working_memory_estimate_includes_secondary_buffers() {
        let pixel_count = 100_000_000;
        let decoded_bytes = pixel_count * RGB_CHANNELS;

        assert!(estimate_working_bytes(10_000, 10_000, decoded_bytes) > MAX_IMAGE_WORKING_BYTES);
    }

    #[test]
    #[ignore = "requires the pinned CLIP vision model in the visiongrep cache"]
    fn batched_and_single_image_inference_match() {
        let directory = tempfile::tempdir().unwrap();
        let paths = [
            directory.path().join("first.png"),
            directory.path().join("second.png"),
        ];
        for (seed, path) in [17_u8, 113].into_iter().zip(&paths) {
            let image = RgbImage::from_fn(320, 240, |x, y| {
                Rgb([
                    seed.wrapping_add(x as u8),
                    seed.wrapping_add(y as u8),
                    seed.wrapping_add((x + y) as u8),
                ])
            });
            image.save_with_format(path, ImageFormat::Png).unwrap();
        }

        let model_paths = crate::model::model_paths().unwrap();
        let mut session = VisionSession::load(&model_paths).unwrap();
        let mut timing = TimingRecorder::disabled(crate::model::timing_metadata());
        let mut singles = Vec::new();
        for path in &paths {
            let prepared = prepare_image(path, false).unwrap();
            singles.push(
                embed_prepared_images(vec![prepared], &mut session, &mut timing)
                    .unwrap()
                    .remove(0),
            );
        }
        let prepared = paths
            .iter()
            .map(|path| prepare_image(path, false).unwrap())
            .collect();
        let batched = embed_prepared_images(prepared, &mut session, &mut timing).unwrap();

        for (single, batched) in singles.iter().zip(&batched) {
            let max_absolute_error = single
                .as_slice()
                .iter()
                .zip(batched.as_slice())
                .map(|(left, right)| (left - right).abs())
                .fold(0.0_f32, f32::max);
            assert!(max_absolute_error <= 1e-6, "{max_absolute_error}");
        }
    }

    #[test]
    #[ignore = "requires the pinned DataComp vision model in the visiongrep cache"]
    fn image_embeddings_match_openclip_golden_vectors() {
        let fixture: GoldenFixture = serde_json::from_str(DATACOMP_GOLDEN).unwrap();
        assert_eq!(
            fixture.contract.openclip_revision,
            "4afec35ffe57a943d569ff7ee888061830164da8"
        );
        assert_eq!(
            fixture.contract.visual_onnx_sha256,
            "3f7e6f94e5a34bc7ee8aba84aec0f963f56974ab405fbcd334c8e1c3f832bd2c"
        );
        let directory = tempfile::tempdir().unwrap();
        let model_paths = crate::model::model_paths().unwrap();
        let mut session = VisionSession::load(&model_paths).unwrap();
        let mut timing = TimingRecorder::disabled(crate::model::timing_metadata());
        let mut report_maximum_error = 0.0_f32;
        let mut report_minimum_cosine = 1.0_f32;

        for case in fixture.images {
            let image = RgbImage::from_fn(case.width, case.height, |x, y| {
                Rgb([
                    x.wrapping_add(17 * case.seed) as u8,
                    (y.wrapping_mul(3).wrapping_add(29 * case.seed)) as u8,
                    (x.wrapping_add(y.wrapping_mul(2))
                        .wrapping_add(43 * case.seed)) as u8,
                ])
            });
            let path = directory.path().join(&case.name);
            image.save_with_format(&path, ImageFormat::Png).unwrap();
            let prepared = prepare_image(&path, false).unwrap();
            let actual = embed_prepared_images(vec![prepared], &mut session, &mut timing)
                .unwrap()
                .remove(0);
            let expected =
                NormalizedEmbedding::from_le_bytes(&decode_hex(&case.embedding_le_hex)).unwrap();
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
                "image {:?} exceeded the reference tolerance: max error {maximum_error}, cosine {cosine}",
                case.name
            );
        }
        println!(
            "{}",
            serde_json::json!({
                "maximum_absolute_error": report_maximum_error,
                "minimum_cosine": report_minimum_cosine,
            })
        );
    }

    #[test]
    #[ignore = "requires all pinned DataComp artifacts in the visiongrep cache"]
    fn cosine_scores_rankings_and_thresholds_match_openclip() {
        const THRESHOLD: f32 = 0.25;

        let fixture: GoldenFixture = serde_json::from_str(DATACOMP_GOLDEN).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let model_paths = crate::model::model_paths().unwrap();
        let mut vision_session = VisionSession::load(&model_paths).unwrap();
        let mut text_session = crate::model::TextSession::load(&model_paths).unwrap();
        let mut timing = TimingRecorder::disabled(crate::model::timing_metadata());
        let mut actual_images = Vec::new();
        let mut expected_images = Vec::new();
        let mut paths = Vec::new();
        let mut report_maximum_score_error = 0.0_f32;

        for case in &fixture.images {
            let image = RgbImage::from_fn(case.width, case.height, |x, y| {
                Rgb([
                    x.wrapping_add(17 * case.seed) as u8,
                    (y.wrapping_mul(3).wrapping_add(29 * case.seed)) as u8,
                    (x.wrapping_add(y.wrapping_mul(2))
                        .wrapping_add(43 * case.seed)) as u8,
                ])
            });
            let path = directory.path().join(&case.name);
            image.save_with_format(&path, ImageFormat::Png).unwrap();
            let prepared = prepare_image(&path, false).unwrap();
            actual_images.push(
                embed_prepared_images(vec![prepared], &mut vision_session, &mut timing)
                    .unwrap()
                    .remove(0),
            );
            expected_images.push(
                NormalizedEmbedding::from_le_bytes(&decode_hex(&case.embedding_le_hex)).unwrap(),
            );
            paths.push(case.name.as_str());
        }

        for query in fixture.queries {
            let actual_query = embed_text(&query.query, &mut text_session, &mut timing).unwrap();
            let expected_query =
                NormalizedEmbedding::from_le_bytes(&decode_hex(&query.embedding_le_hex)).unwrap();
            let actual_scores = actual_images
                .iter()
                .map(|image| actual_query.dot(image))
                .collect::<Vec<_>>();
            let expected_scores = expected_images
                .iter()
                .map(|image| expected_query.dot(image))
                .collect::<Vec<_>>();
            let maximum_score_error = actual_scores
                .iter()
                .zip(&expected_scores)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            report_maximum_score_error = report_maximum_score_error.max(maximum_score_error);
            assert!(
                maximum_score_error <= 2e-4,
                "query {:?} exceeded the score tolerance: {maximum_score_error}",
                query.query
            );

            let rank = |scores: &[f32]| {
                let mut ranking = scores.iter().copied().zip(&paths).collect::<Vec<_>>();
                ranking.sort_by(|(left_score, left_path), (right_score, right_path)| {
                    right_score
                        .total_cmp(left_score)
                        .then_with(|| left_path.cmp(right_path))
                });
                ranking
                    .into_iter()
                    .map(|(_, path)| *path)
                    .collect::<Vec<_>>()
            };
            assert_eq!(rank(&actual_scores), rank(&expected_scores));
            assert_eq!(
                actual_scores
                    .iter()
                    .map(|score| *score >= THRESHOLD)
                    .collect::<Vec<_>>(),
                expected_scores
                    .iter()
                    .map(|score| *score >= THRESHOLD)
                    .collect::<Vec<_>>()
            );
        }
        println!(
            "{}",
            serde_json::json!({
                "maximum_score_absolute_error": report_maximum_score_error,
                "rankings_exact": true,
                "threshold_decisions_exact": true,
            })
        );
    }
}
