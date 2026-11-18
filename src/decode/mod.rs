#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod heic_macos;
mod raw;

use std::io::{BufRead, Cursor, Seek};
use std::path::Path;

use image::metadata::Orientation;
use image::{DynamicImage, ImageDecoder, ImageError, ImageFormat, ImageReader};

use crate::embedding::{IMAGE_SIZE, resized_dimensions};
use crate::error::ImagePreparationError;

const MAX_IMAGE_PIXELS: u64 = 100_000_000;
const MAX_IMAGE_WORKING_BYTES: u64 = 512 * 1024 * 1024;
const RGB_CHANNELS: u64 = 3;
const PREPROCESSED_RGB_BYTES: u64 = (IMAGE_SIZE as u64).pow(2) * RGB_CHANNELS;
const INPUT_TENSOR_BYTES: u64 = PREPROCESSED_RGB_BYTES * 4;

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ImageKind {
    Standard,
    Raw,
    #[cfg(target_os = "macos")]
    Heic,
}

impl ImageKind {
    pub(crate) fn from_path(path: &Path) -> Option<Self> {
        let extension = path.extension()?.to_str()?.to_ascii_lowercase();
        match extension.as_str() {
            "jpg" | "jpeg" | "png" | "webp" | "bmp" | "gif" | "tif" | "tiff" => {
                Some(Self::Standard)
            }
            "cr2" | "nef" | "arw" | "dng" | "rw2" | "pef" | "raf" | "cr3" => Some(Self::Raw),
            #[cfg(target_os = "macos")]
            "heic" | "heif" | "hif" => Some(Self::Heic),
            _ => None,
        }
    }
}

pub(crate) fn load_image(path: &Path) -> Result<DynamicImage, ImagePreparationError> {
    // Direct image queries have always been able to sniff files with arbitrary extensions.
    match ImageKind::from_path(path).unwrap_or(ImageKind::Standard) {
        ImageKind::Standard => {
            let reader = ImageReader::open(path)
                .and_then(ImageReader::with_guessed_format)
                .map_err(|source| decode_error(path, source.into()))?;
            decode_image(path, reader, None)
        }
        ImageKind::Raw => {
            let preview = raw::read_preview(path)?;
            let reader = ImageReader::with_format(Cursor::new(preview.jpeg), ImageFormat::Jpeg);
            decode_image(path, reader, preview.orientation)
        }
        #[cfg(target_os = "macos")]
        ImageKind::Heic => heic_macos::load_heic(path),
    }
}

/// Inspects dimensions and working memory requirements before decoding pixels.
fn decode_image(
    path: &Path,
    reader: ImageReader<impl BufRead + Seek>,
    container_orientation: Option<Orientation>,
) -> Result<DynamicImage, ImagePreparationError> {
    let mut decoder = reader
        .into_decoder()
        .map_err(|source| decode_error(path, source))?;
    let (width, height) = decoder.dimensions();
    let decoded_bytes = decoder.total_bytes();
    check_working_size(path, width, height, decoded_bytes)?;

    let orientation = match container_orientation {
        Some(fallback) => decoder
            .exif_metadata()
            .map_err(|source| decode_error(path, source))?
            .as_deref()
            .and_then(Orientation::from_exif_chunk)
            .unwrap_or(fallback),
        None => decoder
            .orientation()
            .map_err(|source| decode_error(path, source))?,
    };
    let mut image =
        DynamicImage::from_decoder(decoder).map_err(|source| decode_error(path, source))?;
    image.apply_orientation(orientation);
    Ok(image)
}

fn check_working_size(
    path: &Path,
    width: u32,
    height: u32,
    decoded_bytes: u64,
) -> Result<(), ImagePreparationError> {
    let pixel_count = check_dimensions(path, width, height, decoded_bytes)?;
    let estimated_working_bytes = estimate_working_bytes(width, height, decoded_bytes, pixel_count);
    if estimated_working_bytes > MAX_IMAGE_WORKING_BYTES {
        return Err(ImagePreparationError::TooLarge {
            path: path.to_owned(),
            width,
            height,
            decoded_bytes,
            estimated_working_bytes,
        });
    }

    Ok(())
}

fn decode_error(path: &Path, source: ImageError) -> ImagePreparationError {
    ImagePreparationError::Decode {
        path: path.to_owned(),
        source,
    }
}

fn check_dimensions(
    path: &Path,
    width: u32,
    height: u32,
    decoded_bytes: u64,
) -> Result<u64, ImagePreparationError> {
    if width == 0 || height == 0 {
        return Err(ImagePreparationError::InvalidDimensions {
            path: path.to_owned(),
            width,
            height,
        });
    }
    let pixel_count = u64::from(width) * u64::from(height);
    if pixel_count > MAX_IMAGE_PIXELS {
        return Err(ImagePreparationError::TooLarge {
            path: path.to_owned(),
            width,
            height,
            decoded_bytes,
            estimated_working_bytes: estimate_working_bytes(
                width,
                height,
                decoded_bytes,
                pixel_count,
            ),
        });
    }
    Ok(pixel_count)
}

fn estimate_working_bytes(width: u32, height: u32, decoded_bytes: u64, pixel_count: u64) -> u64 {
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
    use image::codecs::gif::GifEncoder;
    use image::{Frame, Rgba, RgbaImage};

    use super::*;

    #[test]
    fn classifies_extensions_case_insensitively() {
        for extension in ["JpEg", "PNG", "WeBP", "BMP", "GIF", "TiF", "TiFF"] {
            assert_eq!(
                ImageKind::from_path(Path::new(&format!("photo.{extension}"))),
                Some(ImageKind::Standard)
            );
        }
        for extension in ["CR2", "NEF", "ARW", "DNG", "RW2", "PEF", "RAF", "CR3"] {
            assert_eq!(
                ImageKind::from_path(Path::new(&format!("photo.{extension}"))),
                Some(ImageKind::Raw)
            );
        }
        for path in ["photo", "photo.txt", "photo.orf"] {
            assert_eq!(ImageKind::from_path(Path::new(path)), None);
        }
        for path in ["photo.HEIC", "photo.HeIf", "photo.HIF"] {
            let kind = ImageKind::from_path(Path::new(path));
            #[cfg(target_os = "macos")]
            assert_eq!(kind, Some(ImageKind::Heic));
            #[cfg(not(target_os = "macos"))]
            assert_eq!(kind, None);
        }
    }

    #[test]
    fn gif_decodes_only_the_first_frame() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("animated.gif");
        let mut encoder = GifEncoder::new(std::fs::File::create(&path).unwrap());
        for color in [Rgba([255, 0, 0, 255]), Rgba([0, 255, 0, 255])] {
            let frame = Frame::new(RgbaImage::from_pixel(8, 6, color));
            encoder.encode_frame(frame).unwrap();
        }
        drop(encoder);

        let image = load_image(&path).unwrap().into_rgb8();
        assert_eq!(image.dimensions(), (8, 6));
        assert!(image.pixels().all(|pixel| pixel.0 == [255, 0, 0]));
    }

    #[test]
    fn working_memory_estimate_includes_secondary_buffers() {
        let path = Path::new("large.png");
        let decoded_bytes = MAX_IMAGE_PIXELS * RGB_CHANNELS;
        assert!(matches!(
            check_working_size(path, 10_000, 10_000, decoded_bytes),
            Err(ImagePreparationError::TooLarge { .. })
        ));
    }
}
