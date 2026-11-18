use std::path::Path;

use image::{DynamicImage, RgbImage};
use objc2_core_foundation::{
    CFBoolean, CFDictionary, CFNumber, CFString, CFType, CFURL, CGPoint, CGRect, CGSize,
};
use objc2_core_graphics::{
    CGBitmapContextCreate, CGColorSpace, CGContext, CGImage, CGImageAlphaInfo,
    CGImageByteOrderInfo, kCGColorSpaceSRGB,
};
use objc2_image_io::{
    CGImageSource, kCGImagePropertyPixelHeight, kCGImagePropertyPixelWidth,
    kCGImageSourceCreateThumbnailFromImageAlways, kCGImageSourceCreateThumbnailWithTransform,
    kCGImageSourceThumbnailMaxPixelSize,
};

use crate::embedding::IMAGE_SIZE;
use crate::error::ImagePreparationError;

use super::{check_dimensions, check_working_size};

pub(super) fn load_heic(path: &Path) -> Result<DynamicImage, ImagePreparationError> {
    let platform_error = || ImagePreparationError::PlatformDecode {
        path: path.to_owned(),
    };
    let url = CFURL::from_file_path(path).ok_or_else(platform_error)?;
    // SAFETY: The URL is retained for the call and there is no options dictionary to validate.
    let source = unsafe { CGImageSource::with_url(&url, None) }.ok_or_else(platform_error)?;
    // SAFETY: The source is a live ImageIO object, and the binding retains the returned CFString.
    let source_type = unsafe { source.r#type() }.ok_or_else(platform_error)?;
    if !["public.heic", "public.heif", "public.heics"]
        .iter()
        .any(|name| source_type == CFString::from_str(name))
    {
        return Err(platform_error());
    }

    // SAFETY: ImageIO accepts index zero and no options.
    let properties = unsafe { source.properties_at_index(0, None) }.ok_or_else(platform_error)?;
    // SAFETY: ImageIO returns an immutable dictionary with CFString keys and Core Foundation
    // values. The number type is checked on lookup.
    let (width, height) = unsafe {
        let properties = properties.cast_unchecked::<CFString, CFType>();
        (
            dimension(properties, kCGImagePropertyPixelWidth).ok_or_else(platform_error)?,
            dimension(properties, kCGImagePropertyPixelHeight).ok_or_else(platform_error)?,
        )
    };
    let full_rgba_bytes = rgba_bytes(width, height);
    check_dimensions(path, width, height, full_rgba_bytes)?;

    let short = u64::from(width.min(height));
    let long = u64::from(width.max(height));
    let max_pixel_size = (long * u64::from(IMAGE_SIZE)).div_ceil(short).min(long);
    let max_pixel_size =
        CFNumber::new_i64(i64::try_from(max_pixel_size).map_err(|_| platform_error())?);
    let enabled = CFBoolean::new(true);
    // SAFETY: These ImageIO constants are immutable CFString references. Each value has the type
    // required by its key: booleans for the two switches and an integer for the pixel limit.
    let thumbnail = unsafe {
        let options = CFDictionary::<CFString, CFType>::from_slices(
            &[
                kCGImageSourceCreateThumbnailFromImageAlways,
                kCGImageSourceCreateThumbnailWithTransform,
                kCGImageSourceThumbnailMaxPixelSize,
            ],
            &[enabled.as_ref(), enabled.as_ref(), max_pixel_size.as_ref()],
        );
        source.thumbnail_at_index(0, Some(options.as_opaque()))
    }
    .ok_or_else(platform_error)?;

    let columns = CGImage::width(Some(&thumbnail));
    let rows = CGImage::height(Some(&thumbnail));
    let width = u32::try_from(columns).map_err(|_| platform_error())?;
    let height = u32::try_from(rows).map_err(|_| platform_error())?;
    check_working_size(path, width, height, rgba_bytes(width, height))?;
    let stride = columns.checked_mul(4).ok_or_else(platform_error)?;
    let len = stride.checked_mul(rows).ok_or_else(platform_error)?;
    let mut pixels = vec![0_u8; len];

    // SAFETY: kCGColorSpaceSRGB is an immutable CFString provided by Core Graphics. The
    // initialized buffer has rows * stride bytes and is not accessed until the context is released.
    let context = unsafe {
        let color_space =
            CGColorSpace::with_name(Some(kCGColorSpaceSRGB)).ok_or_else(platform_error)?;
        CGBitmapContextCreate(
            pixels.as_mut_ptr().cast(),
            columns,
            rows,
            8,
            stride,
            Some(&color_space),
            CGImageAlphaInfo::NoneSkipLast.0 | CGImageByteOrderInfo::Order32Big.0,
        )
    }
    .ok_or_else(platform_error)?;
    let bounds = CGRect::new(
        CGPoint::ZERO,
        CGSize::new(f64::from(width), f64::from(height)),
    );
    CGContext::draw_image(Some(&context), bounds, Some(&thumbnail));
    drop(context);

    let rgb = pixels
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|&[r, g, b, _]| [r, g, b])
        .collect();
    RgbImage::from_raw(width, height, rgb)
        .map(DynamicImage::ImageRgb8)
        .ok_or_else(platform_error)
}

fn dimension(properties: &CFDictionary<CFString, CFType>, key: &CFString) -> Option<u32> {
    let value = properties.get(key)?;
    u32::try_from(value.downcast_ref::<CFNumber>()?.as_i64()?).ok()
}

fn rgba_bytes(width: u32, height: u32) -> u64 {
    (u64::from(width) * u64::from(height)).saturating_mul(4)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn honors_heic_rotation_and_keeps_the_short_edge_at_least_256_pixels() {
        // The synthetic 400x300 PNG has red above blue and EXIF orientation 6.
        // sips -s format heic orientation.png --out tests/fixtures/rotated.heic
        let path = Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/rotated.heic"
        ));
        let image = load_heic(path).unwrap().into_rgb8();
        let (width, height) = image.dimensions();
        assert!(height > width);
        assert!(width >= IMAGE_SIZE);
        let left = image.get_pixel(width / 4, height / 2);
        let right = image.get_pixel(3 * width / 4, height / 2);
        assert!(left[2] > left[0], "expected blue on the left, got {left:?}");
        assert!(
            right[0] > right[2],
            "expected red on the right, got {right:?}"
        );
    }

    #[test]
    fn rejects_a_png_renamed_to_heic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("photo.heic");
        RgbImage::new(16, 16)
            .save_with_format(&path, image::ImageFormat::Png)
            .unwrap();
        assert!(matches!(
            load_heic(&path),
            Err(ImagePreparationError::PlatformDecode { path: actual }) if actual == path
        ));
    }
}
