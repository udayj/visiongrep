use std::path::Path;

use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage, imageops::FilterType};
use ndarray::Array4;

use crate::error::{EmbeddingError, VisionGrepError};
use crate::model::{TextSession, VisionSession};
use crate::timing::{Phase, TimingRecorder};

const IMAGE_SIZE: u32 = 224;
const IMAGE_SIZE_USIZE: usize = 224;
pub(crate) const EMBEDDING_DIM: usize = 512;
const EMBEDDING_BYTES: usize = EMBEDDING_DIM * std::mem::size_of::<f32>();
const NORMALIZED_NORM_TOLERANCE: f32 = 1e-3;
const MAX_IMAGE_PIXELS: u64 = 100_000_000;
const MAX_IMAGE_WORKING_BYTES: u64 = 512 * 1024 * 1024;
const RGB_CHANNELS: u64 = 3;
const PREPROCESSED_RGB_BYTES: u64 = 224 * 224 * RGB_CHANNELS;
const INPUT_TENSOR_BYTES: u64 = PREPROCESSED_RGB_BYTES * 4;
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

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

/// Converts one image into the normalized embedding representation used by storage and search.
pub(crate) fn embed_image(
    path: &Path,
    session: &mut VisionSession,
    timing: &mut TimingRecorder,
) -> Result<NormalizedEmbedding, VisionGrepError> {
    let input = preprocess_image(path, timing)?;
    let inference_started = timing.start();
    let embedding = session.run(&input)?;
    timing.record(Phase::VisionInference, inference_started);
    NormalizedEmbedding::from_model_output(embedding).map_err(|source| {
        VisionGrepError::InvalidModelEmbedding {
            kind: "image",
            source,
        }
    })
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

fn preprocess_image(
    path: &Path,
    timing: &mut TimingRecorder,
) -> Result<Array4<f32>, VisionGrepError> {
    let decoding_started = timing.start();
    let image = load_image(path)?;
    timing.record(Phase::ImageDecoding, decoding_started);
    let preprocessing_started = timing.start();
    let input = preprocess_pixels(image);
    timing.record(Phase::ImagePreprocessing, preprocessing_started);
    Ok(input)
}

/// Decodes an image only after checking its declared resource requirements and applies orientation.
///
/// Dimensions and decoded byte count are inspected before pixel allocation to limit unconstrained
/// decompression and pathological inputs. EXIF orientation is applied before cropping so the crop follows
/// the image's intended visual orientation.
fn load_image(path: &Path) -> Result<DynamicImage, VisionGrepError> {
    let reader = ImageReader::open(path)
        .map_err(|source| VisionGrepError::ImageDecode {
            path: path.to_owned(),
            source: source.into(),
        })?
        .with_guessed_format()
        .map_err(|source| VisionGrepError::ImageDecode {
            path: path.to_owned(),
            source: source.into(),
        })?;
    let mut decoder = reader
        .into_decoder()
        .map_err(|source| VisionGrepError::ImageDecode {
            path: path.to_owned(),
            source,
        })?;
    let (width, height) = decoder.dimensions();
    if width == 0 || height == 0 {
        return Err(VisionGrepError::InvalidImageDimensions {
            path: path.to_owned(),
            width,
            height,
        });
    }
    let pixel_count = u64::from(width).saturating_mul(u64::from(height));
    let decoded_bytes = decoder.total_bytes();
    let estimated_working_bytes = estimate_working_bytes(pixel_count, decoded_bytes);
    if pixel_count > MAX_IMAGE_PIXELS || estimated_working_bytes > MAX_IMAGE_WORKING_BYTES {
        return Err(VisionGrepError::ImageTooLarge {
            path: path.to_owned(),
            width,
            height,
            decoded_bytes,
            estimated_working_bytes,
        });
    }

    let orientation = decoder
        .orientation()
        .map_err(|source| VisionGrepError::ImageDecode {
            path: path.to_owned(),
            source,
        })?;
    let mut image =
        DynamicImage::from_decoder(decoder).map_err(|source| VisionGrepError::ImageDecode {
            path: path.to_owned(),
            source,
        })?;
    image.apply_orientation(orientation);
    Ok(image)
}

/// Produces the channel-first, normalized `[1, 3, 224, 224]` tensor expected by the vision model.
fn preprocess_pixels(image: DynamicImage) -> Array4<f32> {
    let resized = resize_and_center_crop(image);
    let mut input = Array4::<f32>::zeros((1, 3, IMAGE_SIZE_USIZE, IMAGE_SIZE_USIZE));

    for (pixel_index, pixel) in resized.pixels().enumerate() {
        let x = pixel_index % IMAGE_SIZE_USIZE;
        let y = pixel_index / IMAGE_SIZE_USIZE;
        for channel in 0..3 {
            let value = f32::from(pixel[channel]) / 255.0;
            input[[0, channel, y, x]] = (value - CLIP_MEAN[channel]) / CLIP_STD[channel];
        }
    }

    input
}

/// Preserves geometry by taking the centered short-edge square before resizing to model dimensions.
///
/// Cropping first avoids constructing a very large intermediate bitmap for extreme panoramas while
/// retaining the same centered square region as short-edge resize followed by center crop.
fn resize_and_center_crop(image: DynamicImage) -> RgbImage {
    let image = image.into_rgb8();
    let crop_size = image.width().min(image.height());
    let left = (image.width() - crop_size) / 2;
    let top = (image.height() - crop_size) / 2;
    let cropped = image::imageops::crop_imm(&image, left, top, crop_size, crop_size);

    image::imageops::resize(&*cropped, IMAGE_SIZE, IMAGE_SIZE, FilterType::CatmullRom)
}

fn estimate_working_bytes(pixel_count: u64, decoded_bytes: u64) -> u64 {
    // Orientation can temporarily duplicate the decoded image, while conversion may allocate RGB.
    decoded_bytes
        .saturating_mul(2)
        .saturating_add(pixel_count.saturating_mul(RGB_CHANNELS))
        .saturating_add(PREPROCESSED_RGB_BYTES)
        .saturating_add(INPUT_TENSOR_BYTES)
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(resized.pixels().all(|pixel| pixel[1] > pixel[0]));
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
        assert!(resized.pixels().all(|pixel| pixel[1] > pixel[0]));
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

        assert!(estimate_working_bytes(pixel_count, decoded_bytes) > MAX_IMAGE_WORKING_BYTES);
    }
}
