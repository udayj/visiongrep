use crate::embedding::embed_image;
use crate::error::VisionGrepError;
use crate::model::VisionSession;

use super::scan::ImageFile;
use super::store::{ImageIndex, ImageRecord};

pub(crate) enum IngestEvent {
    Started { total: u64 },
    ImageProcessed,
    ImageSkipped(VisionGrepError),
    Finished,
}

/// Embeds and persists changed images, treating unreadable and oversized files as recoverable.
///
/// Other failures abort the operation because they indicate an inference, persistence, or runtime
/// problem rather than one bad input file. Events report progress without coupling ingestion to a
/// particular presentation layer.
pub(crate) fn embed_into_index(
    index: &mut ImageIndex,
    files: &[ImageFile],
    session: &mut VisionSession,
    on_event: &mut impl FnMut(IngestEvent),
) -> Result<(), VisionGrepError> {
    report_started(files.len(), on_event)?;
    let result = (|| {
        for file in files {
            if let Some(embedding) = embed_file(file, session, on_event)? {
                index.upsert_embedding(file, &embedding)?;
            }
            on_event(IngestEvent::ImageProcessed);
        }
        Ok(())
    })();
    on_event(IngestEvent::Finished);
    result
}

/// Embeds files into memory with the same skip-versus-abort policy as persisted ingestion.
pub(crate) fn embed_images(
    files: &[ImageFile],
    session: &mut VisionSession,
    on_event: &mut impl FnMut(IngestEvent),
) -> Result<Vec<ImageRecord>, VisionGrepError> {
    report_started(files.len(), on_event)?;
    let result = (|| {
        let mut records = Vec::with_capacity(files.len());
        for file in files {
            if let Some(embedding) = embed_file(file, session, on_event)? {
                records.push(ImageRecord {
                    path: file.path.clone(),
                    embedding,
                });
            }
            on_event(IngestEvent::ImageProcessed);
        }
        Ok(records)
    })();
    on_event(IngestEvent::Finished);
    result
}

fn report_started(
    file_count: usize,
    on_event: &mut impl FnMut(IngestEvent),
) -> Result<(), VisionGrepError> {
    let total = u64::try_from(file_count).map_err(|source| VisionGrepError::NumericConversion {
        context: "reporting image ingestion progress",
        source,
    })?;
    on_event(IngestEvent::Started { total });
    Ok(())
}

fn embed_file(
    file: &ImageFile,
    session: &mut VisionSession,
    on_event: &mut impl FnMut(IngestEvent),
) -> Result<Option<Vec<f32>>, VisionGrepError> {
    match embed_image(&file.path, session) {
        Ok(embedding) => Ok(Some(embedding)),
        Err(
            error @ (VisionGrepError::ImageDecode { .. } | VisionGrepError::ImageTooLarge { .. }),
        ) => {
            on_event(IngestEvent::ImageSkipped(error));
            Ok(None)
        }
        Err(error) => Err(error),
    }
}
