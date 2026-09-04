//! Pillow-compatible 8-bit RGB bicubic resize used by the OpenCLIP contract.
//!
//! The coefficient construction and fixed-point rounding follow Pillow 12.3.0's
//! `src/libImaging/Resample.c`. Pillow's MIT-CMU notice is reproduced in the repository's
//! third-party notices.

use image::RgbImage;

const SUPPORT: f64 = 2.0;
const PRECISION_BITS: u32 = 22;
const ROUNDING: i64 = 1 << (PRECISION_BITS - 1);

struct Coefficients {
    start: usize,
    weights: Vec<i32>,
}

pub(crate) fn resize_rgb(image: &RgbImage, output_width: u32, output_height: u32) -> RgbImage {
    if image.width() == output_width && image.height() == output_height {
        return image.clone();
    }

    let intermediate = if image.width() == output_width {
        image.clone()
    } else {
        let horizontal = coefficients(image.width(), output_width);
        let mut intermediate = RgbImage::new(output_width, image.height());
        for y in 0..image.height() as usize {
            for (output_x, coefficient) in horizontal.iter().enumerate() {
                let output_index = (y * output_width as usize + output_x) * 3;
                for channel in 0..3 {
                    let mut sum = ROUNDING;
                    for (offset, weight) in coefficient.weights.iter().enumerate() {
                        let input_index =
                            (y * image.width() as usize + coefficient.start + offset) * 3 + channel;
                        sum += i64::from(image.as_raw()[input_index]) * i64::from(*weight);
                    }
                    intermediate.as_mut()[output_index + channel] = clip(sum);
                }
            }
        }
        intermediate
    };

    if image.height() == output_height {
        return intermediate;
    }

    let vertical = coefficients(image.height(), output_height);
    let mut output = RgbImage::new(output_width, output_height);
    for (output_y, coefficient) in vertical.iter().enumerate() {
        for x in 0..output_width as usize {
            let output_index = (output_y * output_width as usize + x) * 3;
            for channel in 0..3 {
                let mut sum = ROUNDING;
                for (offset, weight) in coefficient.weights.iter().enumerate() {
                    let input_index =
                        ((coefficient.start + offset) * output_width as usize + x) * 3 + channel;
                    sum += i64::from(intermediate.as_raw()[input_index]) * i64::from(*weight);
                }
                output.as_mut()[output_index + channel] = clip(sum);
            }
        }
    }
    output
}

fn coefficients(input_size: u32, output_size: u32) -> Vec<Coefficients> {
    let scale = f64::from(input_size) / f64::from(output_size);
    let filter_scale = scale.max(1.0);
    let support = SUPPORT * filter_scale;
    let inverse_filter_scale = 1.0 / filter_scale;

    (0..output_size)
        .map(|output| {
            let center = (f64::from(output) + 0.5) * scale;
            let minimum = ((center - support + 0.5) as i64).max(0) as usize;
            let maximum = ((center + support + 0.5) as i64)
                .min(i64::from(input_size))
                .max(minimum as i64) as usize;
            let mut weights = (minimum..maximum)
                .map(|input| bicubic((input as f64 - center + 0.5) * inverse_filter_scale))
                .collect::<Vec<_>>();
            let sum = weights.iter().sum::<f64>();
            if sum != 0.0 {
                for weight in &mut weights {
                    *weight /= sum;
                }
            }
            let weights = weights
                .into_iter()
                .map(|weight| {
                    let scaled = weight * f64::from(1 << PRECISION_BITS);
                    if weight < 0.0 {
                        (-0.5 + scaled) as i32
                    } else {
                        (0.5 + scaled) as i32
                    }
                })
                .collect();
            Coefficients {
                start: minimum,
                weights,
            }
        })
        .collect()
}

fn bicubic(value: f64) -> f64 {
    let value = value.abs();
    if value < 1.0 {
        ((1.5 * value - 2.5) * value * value) + 1.0
    } else if value < 2.0 {
        (((value - 5.0) * value + 8.0) * value - 4.0) * -0.5
    } else {
        0.0
    }
}

fn clip(value: i64) -> u8 {
    (value >> PRECISION_BITS).clamp(0, 255) as u8
}

#[cfg(test)]
mod tests {
    use image::Rgb;

    use super::*;

    #[test]
    fn identity_resize_preserves_pixels() {
        let image = RgbImage::from_fn(3, 2, |x, y| Rgb([(x * 31) as u8, (y * 47) as u8, 99]));

        assert_eq!(resize_rgb(&image, 3, 2), image);
    }

    #[test]
    fn resize_matches_pillow_12_3_bicubic_fixture() {
        let image = RgbImage::from_fn(4, 3, |x, y| {
            Rgb([
                (x * 31 + y * 47) as u8,
                (x * 31 + y * 47 + 73) as u8,
                (x * 31 + y * 47 + 146) as u8,
            ])
        });
        let expected = [
            0, 70, 143, 18, 91, 164, 45, 118, 190, 71, 144, 220, 92, 165, 251, 26, 98, 170, 46,
            119, 205, 73, 145, 249, 99, 177, 213, 120, 211, 98, 67, 140, 217, 88, 161, 165, 115,
            194, 170, 141, 187, 141, 162, 137, 26, 95, 168, 253, 116, 189, 69, 143, 233, 2, 169,
            167, 50, 190, 0, 81,
        ];

        assert_eq!(resize_rgb(&image, 5, 4).as_raw(), &expected);
    }
}
