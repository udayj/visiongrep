use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufReader, Cursor, Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::Path;

use image::metadata::Orientation;
use image::{ImageFormat, ImageReader};

use crate::embedding::IMAGE_SIZE;
use crate::error::{ImagePreparationError, RawContainerError};

const MAX_IFDS: usize = 64;
const MAX_IFD_ENTRIES: u16 = 1024;
const MAX_BOXES: usize = 1024;
const MAX_PREVIEW_BYTES: u64 = 64 * 1024 * 1024;
// The image JPEG decoder buffers its input even when asked for dimensions only.
const MAX_JPEG_HEADER_BYTES: u64 = 1024 * 1024;
const CR3_PREVIEW_UUID: [u8; 16] = 0xeaf42b5e_1c98_4b88_b9fb_b7dc406e4d16_u128.to_be_bytes();
const CR3_METADATA_UUID: [u8; 16] = 0x85c0b687_820f_11e0_8111_f4ce462b6a48_u128.to_be_bytes();

#[derive(Debug)]
pub(super) struct Preview {
    pub(super) jpeg: Vec<u8>,
    pub(super) orientation: Option<Orientation>,
}

pub(super) fn read_preview(path: &Path) -> Result<Preview, ImagePreparationError> {
    let file = File::open(path).map_err(|source| raw_read_error(path, source))?;
    let size = file
        .metadata()
        .map_err(|source| raw_read_error(path, source))?
        .len();
    let mut input = RawFile {
        reader: BufReader::new(file),
        path,
    };
    let file = 0..size;
    let header = input.read::<8>(&file, 0)?;
    let container = if &header == b"FUJIFILM" {
        input.raf(&file)?
    } else if &header[4..] == b"ftyp" {
        input.cr3(&file)?
    } else {
        input.tiff(&file, TiffContents::Previews)?
    };

    let mut best = None;
    let mut invalid_range = None;
    for candidate in container.previews {
        let range = match candidate {
            Ok(range) => range,
            Err(error) => {
                invalid_range = Some(error);
                continue;
            }
        };
        let len = range.end - range.start;
        if !(2..=MAX_PREVIEW_BYTES).contains(&len) || input.read::<2>(&range, 0)? != [0xff, 0xd8] {
            continue;
        }

        let header_len = usize::try_from(len.min(MAX_JPEG_HEADER_BYTES))
            .map_err(|_| input.error(RawContainerError::OffsetOutOfBounds))?;
        let mut header = vec![0; header_len];
        input.read_into(range.start, &mut header)?;
        let reader = ImageReader::with_format(Cursor::new(header), ImageFormat::Jpeg);
        let Ok((width, height)) = reader.into_dimensions() else {
            continue;
        };
        let pixels = u64::from(width) * u64::from(height);
        if best
            .as_ref()
            .is_none_or(|(_, _, _, best_pixels)| pixels > *best_pixels)
        {
            best = Some((range, width, height, pixels));
        }
    }

    let (range, width, height, _) =
        best.ok_or_else(|| input.error(invalid_range.unwrap_or(RawContainerError::NoPreview)))?;
    if width.min(height) < IMAGE_SIZE {
        return Err(ImagePreparationError::RawPreviewTooSmall {
            path: path.to_owned(),
            width,
            height,
        });
    }
    let len = usize::try_from(range.end - range.start)
        .map_err(|_| input.error(RawContainerError::OffsetOutOfBounds))?;
    let mut jpeg = vec![0; len];
    input.read_into(range.start, &mut jpeg)?;
    Ok(Preview {
        jpeg,
        orientation: container.orientation,
    })
}

fn raw_read_error(path: &Path, source: io::Error) -> ImagePreparationError {
    ImagePreparationError::RawRead {
        path: path.to_owned(),
        source,
    }
}

#[derive(Default)]
struct Container {
    previews: Vec<Result<Range<u64>, RawContainerError>>,
    orientation: Option<Orientation>,
}

struct RawFile<'a> {
    reader: BufReader<File>,
    path: &'a Path,
}

impl RawFile<'_> {
    fn error(&self, source: RawContainerError) -> ImagePreparationError {
        ImagePreparationError::RawContainer {
            path: self.path.to_owned(),
            source,
        }
    }

    fn seek(&mut self, offset: u64) -> Result<(), ImagePreparationError> {
        self.reader
            .seek(SeekFrom::Start(offset))
            .map_err(|source| raw_read_error(self.path, source))?;
        Ok(())
    }

    fn read_into(&mut self, offset: u64, buf: &mut [u8]) -> Result<(), ImagePreparationError> {
        self.seek(offset)?;
        self.reader.read_exact(buf).map_err(|source| {
            if source.kind() == io::ErrorKind::UnexpectedEof {
                self.error(RawContainerError::Truncated)
            } else {
                raw_read_error(self.path, source)
            }
        })
    }

    fn read<const N: usize>(
        &mut self,
        region: &Range<u64>,
        offset: u64,
    ) -> Result<[u8; N], ImagePreparationError> {
        let range = self.structure_span(region, offset, N as u64)?;
        let mut bytes = [0; N];
        self.read_into(range.start, &mut bytes)?;
        Ok(bytes)
    }

    fn structure_span(
        &self,
        region: &Range<u64>,
        offset: u64,
        len: u64,
    ) -> Result<Range<u64>, ImagePreparationError> {
        // An in-range start with missing bytes is a truncated structure.
        span(region, offset, 0).map_err(|source| self.error(source))?;
        span(region, offset, len).map_err(|_| self.error(RawContainerError::Truncated))
    }

    fn raf(&mut self, file: &Range<u64>) -> Result<Container, ImagePreparationError> {
        if &self.read::<15>(file, 0)? != b"FUJIFILMCCD-RAW" {
            return Err(self.error(RawContainerError::UnrecognizedHeader));
        }
        let offset = u32::from_be_bytes(self.read(file, 84)?);
        let length = u32::from_be_bytes(self.read(file, 88)?);
        Ok(Container {
            previews: vec![span(file, u64::from(offset), u64::from(length))],
            orientation: None,
        })
    }
}

fn span(region: &Range<u64>, offset: u64, len: u64) -> Result<Range<u64>, RawContainerError> {
    let start = region
        .start
        .checked_add(offset)
        .ok_or(RawContainerError::OffsetOutOfBounds)?;
    let end = start
        .checked_add(len)
        .filter(|&end| end <= region.end)
        .ok_or(RawContainerError::OffsetOutOfBounds)?;
    Ok(start..end)
}

#[derive(Clone, Copy)]
enum ByteOrder {
    Little,
    Big,
}

impl ByteOrder {
    fn short(self, bytes: [u8; 2]) -> u16 {
        match self {
            Self::Little => u16::from_le_bytes(bytes),
            Self::Big => u16::from_be_bytes(bytes),
        }
    }

    fn long(self, bytes: [u8; 4]) -> u32 {
        match self {
            Self::Little => u32::from_le_bytes(bytes),
            Self::Big => u32::from_be_bytes(bytes),
        }
    }
}

#[derive(PartialEq, Eq)]
enum TiffContents {
    Previews,
    Orientation,
}

#[derive(Default)]
struct Ifd {
    compression: Option<u32>,
    strip_offset: Option<u32>,
    strip_len: Option<u32>,
    jpeg_offset: Option<u32>,
    jpeg_len: Option<u32>,
    jpg_from_raw: Option<(u32, u32)>,
}

impl RawFile<'_> {
    fn tiff(
        &mut self,
        region: &Range<u64>,
        contents: TiffContents,
    ) -> Result<Container, ImagePreparationError> {
        let [a, b, c, d, e, f, g, h] = self.read(region, 0)?;
        let order = match &[a, b, c, d] {
            b"II*\0" | b"IIU\0" => ByteOrder::Little,
            b"MM\0*" => ByteOrder::Big,
            _ => return Err(self.error(RawContainerError::UnrecognizedHeader)),
        };
        let root = order.long([e, f, g, h]);
        let mut container = Container::default();
        let mut pending = vec![root];
        let mut visited = HashSet::new();
        while let Some(offset) = pending.pop() {
            if offset == 0 {
                continue;
            }
            if !visited.insert(offset) {
                return Err(self.error(RawContainerError::IfdCycle));
            }
            if visited.len() > MAX_IFDS {
                return Err(self.error(RawContainerError::TooManyIfds));
            }
            let offset = u64::from(offset);
            let count = order.short(self.read(region, offset)?);
            if count > MAX_IFD_ENTRIES {
                return Err(self.error(RawContainerError::TooManyIfds));
            }
            let entries_size = u64::from(count) * 12;
            let entries = self.structure_span(region, offset + 2, entries_size + 4)?;
            let mut bytes = vec![0; usize::from(count) * 12];
            self.read_into(entries.start, &mut bytes)?;
            let mut ifd = Ifd::default();
            for &[a, b, c, d, e, f, g, h, i, j, k, l] in bytes.as_chunks::<12>().0 {
                let tag = order.short([a, b]);
                let kind = order.short([c, d]);
                let count = order.long([e, f, g, h]);
                let long = order.long([i, j, k, l]);
                let scalar = match (kind, count) {
                    (3, 1) => Some(u32::from(order.short([i, j]))),
                    (4 | 13, 1) => Some(long),
                    _ => None,
                };

                if tag == 0x0112 && offset == u64::from(root) {
                    container.orientation = scalar
                        .and_then(|value| u8::try_from(value).ok())
                        .and_then(Orientation::from_exif);
                }
                if contents == TiffContents::Orientation {
                    continue;
                }
                match tag {
                    0x0103 => ifd.compression = scalar,
                    0x0111 => ifd.strip_offset = scalar,
                    0x0117 => ifd.strip_len = scalar,
                    0x0201 => ifd.jpeg_offset = scalar,
                    0x0202 => ifd.jpeg_len = scalar,
                    0x002e if matches!(kind, 1 | 7) && count > 4 => {
                        ifd.jpg_from_raw = Some((long, count));
                    }
                    0x014a if matches!(kind, 4 | 13) => {
                        self.sub_ifds(region, order, count, long, &mut pending)?;
                    }
                    _ => {}
                }
            }
            if contents == TiffContents::Orientation {
                break;
            }
            let jpeg = ifd.jpeg_offset.zip(ifd.jpeg_len);
            let strip = match ifd.compression {
                Some(6 | 7) => ifd.strip_offset.zip(ifd.strip_len),
                _ => None,
            };
            for (offset, len) in [jpeg, strip, ifd.jpg_from_raw].into_iter().flatten() {
                container
                    .previews
                    .push(span(region, u64::from(offset), u64::from(len)));
            }
            pending.push(order.long(self.read(region, offset + 2 + entries_size)?));
        }
        Ok(container)
    }

    fn sub_ifds(
        &mut self,
        region: &Range<u64>,
        order: ByteOrder,
        count: u32,
        value: u32,
        pending: &mut Vec<u32>,
    ) -> Result<(), ImagePreparationError> {
        let len = u64::from(count) * 4;
        let count = usize::try_from(count)
            .ok()
            .filter(|&count| pending.len().saturating_add(count) <= MAX_IFDS)
            .ok_or_else(|| self.error(RawContainerError::TooManyIfds))?;
        match count {
            0 => {}
            1 => pending.push(value),
            _ => {
                let mut bytes = [0; MAX_IFDS * 4];
                let bytes = &mut bytes[..count * 4];
                let range = self.structure_span(region, u64::from(value), len)?;
                self.read_into(range.start, bytes)?;
                pending.extend(
                    bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|&word| order.long(word)),
                );
            }
        }
        Ok(())
    }
}

struct BmffBox {
    kind: [u8; 4],
    payload: Range<u64>,
}

impl RawFile<'_> {
    fn cr3(&mut self, file: &Range<u64>) -> Result<Container, ImagePreparationError> {
        let mut container = Container::default();
        let mut boxes = file.clone();
        let mut budget = MAX_BOXES;
        while let Some(bmff) = self.next_box(&mut boxes, &mut budget)? {
            let mut payload = bmff.payload;
            match &bmff.kind {
                b"uuid" if self.read::<16>(&payload, 0)? == CR3_PREVIEW_UUID => {
                    // Canon puts an eight-byte header between the UUID and its child boxes.
                    payload = self.structure_span(&payload, 24, 0)?.start..payload.end;
                    while let Some(preview) = self.next_box(&mut payload, &mut budget)? {
                        if &preview.kind == b"PRVW" {
                            // PRVW has a 16-byte header ending in the JPEG byte count.
                            let len = u32::from_be_bytes(self.read(&preview.payload, 12)?);
                            container
                                .previews
                                .push(span(&preview.payload, 16, u64::from(len)));
                        }
                    }
                }
                b"moov" => {
                    container.orientation = self.cr3_orientation(payload, &mut budget)?;
                }
                _ => {}
            }
        }
        Ok(container)
    }

    fn cr3_orientation(
        &mut self,
        mut movie: Range<u64>,
        budget: &mut usize,
    ) -> Result<Option<Orientation>, ImagePreparationError> {
        while let Some(bmff) = self.next_box(&mut movie, budget)? {
            let mut payload = bmff.payload;
            if &bmff.kind != b"uuid" || self.read::<16>(&payload, 0)? != CR3_METADATA_UUID {
                continue;
            }
            payload = self.structure_span(&payload, 16, 0)?.start..payload.end;
            while let Some(metadata) = self.next_box(&mut payload, budget)? {
                if &metadata.kind == b"CMT1" {
                    return Ok(self
                        .tiff(&metadata.payload, TiffContents::Orientation)?
                        .orientation);
                }
            }
        }
        Ok(None)
    }

    fn next_box(
        &mut self,
        remaining: &mut Range<u64>,
        budget: &mut usize,
    ) -> Result<Option<BmffBox>, ImagePreparationError> {
        if remaining.is_empty() {
            return Ok(None);
        }
        *budget = budget
            .checked_sub(1)
            .ok_or_else(|| self.error(RawContainerError::TooManyBoxes))?;
        let [a, b, c, d, e, f, g, h] = self.read(remaining, 0)?;
        let (len, header_len) = match u32::from_be_bytes([a, b, c, d]) {
            0 => (remaining.end - remaining.start, 8),
            1 => (u64::from_be_bytes(self.read(remaining, 8)?), 16),
            len => (u64::from(len), 8),
        };
        let len = len
            .checked_sub(header_len)
            .ok_or_else(|| self.error(RawContainerError::OffsetOutOfBounds))?;
        let payload = span(remaining, header_len, len).map_err(|source| self.error(source))?;
        remaining.start = payload.end;
        Ok(Some(BmffBox {
            kind: [e, f, g, h],
            payload,
        }))
    }
}

#[cfg(test)]
mod tests {
    use image::codecs::jpeg::JpegEncoder;
    use image::{ExtendedColorType, GenericImageView};

    use super::*;

    fn jpeg(width: u32, height: u32) -> Vec<u8> {
        let mut out = Vec::new();
        JpegEncoder::new(&mut out)
            .encode(
                &vec![0; (width * height) as usize],
                width,
                height,
                ExtendedColorType::L8,
            )
            .unwrap();
        out
    }

    fn write(name: &str, bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        (dir, path)
    }

    fn tiff(order: ByteOrder, directories: &[(usize, &[[u32; 4]], u32)]) -> Vec<u8> {
        let magic = match order {
            ByteOrder::Little => b"II*\0",
            ByteOrder::Big => b"MM\0*",
        };
        let short = |value: u16| match order {
            ByteOrder::Little => value.to_le_bytes(),
            ByteOrder::Big => value.to_be_bytes(),
        };
        let long = |value: u32| match order {
            ByteOrder::Little => value.to_le_bytes(),
            ByteOrder::Big => value.to_be_bytes(),
        };
        let mut bytes = vec![0; 256];
        bytes[..4].copy_from_slice(magic);
        bytes[4..8].copy_from_slice(&long(8));
        for &(offset, entries, next) in directories {
            let mut ifd = short(u16::try_from(entries.len()).unwrap()).to_vec();
            for &[tag, kind, count, value] in entries {
                ifd.extend(short(u16::try_from(tag).unwrap()));
                ifd.extend(short(u16::try_from(kind).unwrap()));
                ifd.extend(long(count));
                if (kind, count) == (3, 1) {
                    ifd.extend(short(u16::try_from(value).unwrap()));
                    ifd.extend([0; 2]);
                } else {
                    ifd.extend(long(value));
                }
            }
            ifd.extend(long(next));
            bytes[offset..offset + ifd.len()].copy_from_slice(&ifd);
        }
        bytes
    }

    #[test]
    fn finds_tiff_previews() {
        let preview = jpeg(320, 256);
        let len = u32::try_from(preview.len()).unwrap();
        let interchange = [[0x0201, 4, 1, 256], [0x0202, 4, 1, len]];
        let strip = [[0x0103, 3, 1, 6], [0x0111, 4, 1, 256], [0x0117, 4, 1, len]];
        let mut panasonic = tiff(ByteOrder::Little, &[(8, &[[0x002e, 7, len, 256]], 0)]);
        panasonic[2] = b'U';
        for mut bytes in [
            tiff(ByteOrder::Little, &[(8, &interchange, 0)]),
            tiff(ByteOrder::Big, &[(8, &interchange, 0)]),
            tiff(
                ByteOrder::Little,
                &[(8, &[[0x014a, 13, 1, 128]], 0), (128, &interchange, 0)],
            ),
            tiff(ByteOrder::Little, &[(8, &strip, 0)]),
            panasonic,
        ] {
            bytes.extend(&preview);
            let (_dir, path) = write("photo.dng", &bytes);
            assert_eq!(read_preview(&path).unwrap().jpeg, preview);
        }
    }

    #[test]
    fn largest_valid_preview_wins() {
        let small = jpeg(256, 256);
        let large = jpeg(320, 256);
        let small_len = u32::try_from(small.len()).unwrap();
        let large_len = u32::try_from(large.len()).unwrap();
        let entries = [
            [0x002e, 7, 5, 256 + small_len + large_len],
            [0x0103, 3, 1, 7],
            [0x0111, 4, 1, 256 + small_len],
            [0x0117, 4, 1, large_len],
            [0x0201, 4, 1, 256],
            [0x0202, 4, 1, small_len],
        ];
        let mut bytes = tiff(ByteOrder::Little, &[(8, &entries, 0)]);
        bytes.extend(small);
        bytes.extend(&large);
        bytes.extend(b"\xff\xd8bad");
        let (_dir, path) = write("photo.dng", &bytes);
        assert_eq!(read_preview(&path).unwrap().jpeg, large);
    }

    #[test]
    fn rejects_malformed_tiff_containers() {
        use RawContainerError::*;

        for (bytes, expected) in [
            (tiff(ByteOrder::Little, &[(8, &[], 8)]), IfdCycle),
            (
                tiff(
                    ByteOrder::Little,
                    &[(8, &[[0x0201, 4, 1, u32::MAX], [0x0202, 4, 1, 16]], 0)],
                ),
                OffsetOutOfBounds,
            ),
            (
                tiff(
                    ByteOrder::Little,
                    &[(8, &[[0x0201, 4, 1, 64], [0x0202, 4, 1, u32::MAX]], 0)],
                ),
                OffsetOutOfBounds,
            ),
            (b"II*\0\x08\0\0\0\x01\0".to_vec(), Truncated),
        ] {
            let (_dir, path) = write("broken.cr2", &bytes);
            let Err(ImagePreparationError::RawContainer { source, .. }) = read_preview(&path)
            else {
                panic!("expected a RAW container error");
            };
            assert_eq!(
                std::mem::discriminant(&source),
                std::mem::discriminant(&expected)
            );
        }
    }

    #[test]
    fn raf_preview_must_have_a_short_edge_of_at_least_256_pixels() {
        for height in [255, 256] {
            let preview = jpeg(320, height);
            let mut bytes = vec![0; 92];
            bytes[..15].copy_from_slice(b"FUJIFILMCCD-RAW");
            bytes[84..88].copy_from_slice(&92_u32.to_be_bytes());
            bytes[88..92].copy_from_slice(&u32::try_from(preview.len()).unwrap().to_be_bytes());
            bytes.extend(&preview);
            let (_dir, path) = write("photo.raf", &bytes);
            let result = read_preview(&path);
            if height == 255 {
                assert!(matches!(
                    result,
                    Err(ImagePreparationError::RawPreviewTooSmall {
                        width: 320,
                        height: 255,
                        ..
                    })
                ));
            } else {
                assert_eq!(result.unwrap().jpeg, preview);
            }
        }
    }

    #[test]
    fn jpeg_orientation_overrides_container_orientation() {
        let original = jpeg(320, 256);
        for (preview, expected) in [
            (original.clone(), (256, 320)),
            (with_jpeg_orientation(&original), (320, 256)),
        ] {
            let len = u32::try_from(preview.len()).unwrap();
            let entries = [[0x0112, 3, 1, 6], [0x0201, 4, 1, 256], [0x0202, 4, 1, len]];
            let mut bytes = tiff(ByteOrder::Little, &[(8, &entries, 0)]);
            bytes.extend(preview);
            let (_dir, path) = write("oriented.nef", &bytes);
            assert_eq!(
                super::super::load_image(&path).unwrap().dimensions(),
                expected
            );
        }
    }

    fn with_jpeg_orientation(jpeg: &[u8]) -> Vec<u8> {
        let exif = tiff(ByteOrder::Big, &[(8, &[[0x0112, 3, 1, 1]], 0)]);
        let mut output = vec![0xff, 0xd8, 0xff, 0xe1, 0, 34];
        output.extend(b"Exif\0\0");
        output.extend(&exif[..26]);
        output.extend(&jpeg[2..]);
        output
    }

    #[test]
    fn cr3_finds_preview_and_nested_cmt1_orientation() {
        fn atom(kind: &[u8; 4], fields: &[&[u8]]) -> Vec<u8> {
            let size = 8 + fields.iter().map(|field| field.len()).sum::<usize>();
            let mut bytes = u32::try_from(size).unwrap().to_be_bytes().to_vec();
            bytes.extend(kind);
            bytes.extend(fields.iter().flat_map(|field| field.iter()));
            bytes
        }
        let preview = jpeg(400, 300);
        let len = u32::try_from(preview.len()).unwrap().to_be_bytes();
        let prvw = atom(b"PRVW", &[&[0; 12], &len, &preview, &[0; 2]]);
        let tiff = tiff(ByteOrder::Big, &[(8, &[[0x0112, 3, 1, 6]], 0)]);
        let cmt1 = atom(b"CMT1", &[&tiff]);
        let mut bytes = atom(b"ftyp", &[b"crx \0\0\0\x01"]);
        bytes.extend(atom(
            b"moov",
            &[&atom(b"uuid", &[&CR3_METADATA_UUID, &cmt1])],
        ));
        bytes.extend(atom(b"uuid", &[&CR3_PREVIEW_UUID, &[0; 8], &prvw]));
        let (_dir, path) = write("photo.cr3", &bytes);
        assert_eq!(read_preview(&path).unwrap().jpeg, preview);
        assert_eq!(
            super::super::load_image(&path).unwrap().dimensions(),
            (300, 400)
        );
    }
}
