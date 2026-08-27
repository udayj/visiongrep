use std::path::Path;

use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage, imageops::FilterType};
use ndarray::Array4;

use crate::error::VisionGrepError;
use crate::model::{TextSession, VisionSession};

const IMAGE_SIZE: u32 = 224;
const IMAGE_SIZE_USIZE: usize = 224;
const MAX_IMAGE_PIXELS: u64 = 100_000_000;
const MAX_DECODED_IMAGE_BYTES: u64 = 512 * 1024 * 1024;
const CLIP_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const CLIP_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

/// Converts one image into the normalized embedding representation used by storage and search.
pub(crate) fn embed_image(
    path: &Path,
    session: &mut VisionSession,
) -> Result<Vec<f32>, VisionGrepError> {
    let input = preprocess_image(path)?;
    let embedding = session.run(&input)?;
    normalize_embedding(embedding, "image")
}

/// Converts a natural-language query into a normalized embedding in the same CLIP vector space.
pub(crate) fn embed_text(
    query: &str,
    session: &mut TextSession,
) -> Result<Vec<f32>, VisionGrepError> {
    let embedding = session.run(query)?;
    normalize_embedding(embedding, "text")
}

/// Enforces the embedding-layer invariant that every outgoing vector has unit L2 norm.
///
/// Rejecting zero and non-finite norms here lets persistence and similarity code rely on normalized
/// finite values without requiring every caller to repeat the check.
pub(crate) fn normalize_embedding(
    mut embedding: Vec<f32>,
    kind: &'static str,
) -> Result<Vec<f32>, VisionGrepError> {
    let norm = embedding
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();

    if !norm.is_finite() || norm <= f32::EPSILON {
        return Err(VisionGrepError::InvalidEmbeddingNorm { kind, norm });
    }

    for value in &mut embedding {
        *value /= norm;
    }

    Ok(embedding)
}

fn preprocess_image(path: &Path) -> Result<Array4<f32>, VisionGrepError> {
    let image = load_image(path)?;
    Ok(preprocess_pixels(&image))
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
    let pixel_count = u64::from(width).saturating_mul(u64::from(height));
    let decoded_bytes = decoder.total_bytes();
    if pixel_count > MAX_IMAGE_PIXELS || decoded_bytes > MAX_DECODED_IMAGE_BYTES {
        return Err(VisionGrepError::ImageTooLarge {
            path: path.to_owned(),
            width,
            height,
            decoded_bytes,
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
fn preprocess_pixels(image: &DynamicImage) -> Array4<f32> {
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
fn resize_and_center_crop(image: &DynamicImage) -> RgbImage {
    let image = image.to_rgb8();
    let crop_size = image.width().min(image.height());
    let left = (image.width() - crop_size) / 2;
    let top = (image.height() - crop_size) / 2;
    let cropped = image::imageops::crop_imm(&image, left, top, crop_size, crop_size).to_image();

    image::imageops::resize(&cropped, IMAGE_SIZE, IMAGE_SIZE, FilterType::CatmullRom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_embedding_returns_unit_vector() {
        let embedding = normalize_embedding(vec![3.0, 4.0], "test").unwrap();

        assert!((embedding[0] - 0.6).abs() < f32::EPSILON);
        assert!((embedding[1] - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn normalize_embedding_rejects_zero_norm() {
        assert!(normalize_embedding(vec![0.0, 0.0], "test").is_err());
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

        let resized = resize_and_center_crop(&DynamicImage::ImageRgb8(image));

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

        let resized = resize_and_center_crop(&DynamicImage::ImageRgb8(image));

        assert_eq!(resized.dimensions(), (IMAGE_SIZE, IMAGE_SIZE));
        assert!(resized.pixels().all(|pixel| pixel[1] > pixel[0]));
    }

    #[test]
    fn preprocessing_scales_tiny_images() {
        let image = DynamicImage::new_rgb8(1, 1);

        assert_eq!(
            resize_and_center_crop(&image).dimensions(),
            (IMAGE_SIZE, IMAGE_SIZE)
        );
    }
}
