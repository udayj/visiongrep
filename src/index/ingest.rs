use crate::embedding::{NormalizedEmbedding, embed_image};
use crate::error::VisionGrepError;
use crate::model::VisionSession;

use super::scan::{ImageFile, SearchRoot};
use super::store::{ImageIndex, ImageRecord, ImageUpdate};

const INDEX_BATCH_SIZE: usize = 256;

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
pub(crate) fn ingest_into_index(
    root: &SearchRoot,
    index: &mut ImageIndex,
    files: &[ImageFile],
    session: &mut VisionSession,
    on_event: &mut impl FnMut(IngestEvent),
) -> Result<(), VisionGrepError> {
    report_started(files.len(), on_event)?;
    let result = (|| {
        for files in files.chunks(INDEX_BATCH_SIZE) {
            let mut updates = Vec::with_capacity(files.len());
            for file in files {
                let update = match embed_image_with_skip_policy(root, file, session, on_event)? {
                    Some(embedding) => ImageUpdate::Upsert {
                        file: file.clone(),
                        embedding,
                    },
                    None => ImageUpdate::Delete {
                        relative_path: file.relative_path.clone(),
                    },
                };
                updates.push(update);
                on_event(IngestEvent::ImageProcessed);
            }
            index.apply_updates(updates)?;
        }
        Ok(())
    })();
    on_event(IngestEvent::Finished);
    result
}

/// Embeds files into memory with the same skip-versus-abort policy as persisted ingestion.
pub(crate) fn embed_images(
    root: &SearchRoot,
    files: &[ImageFile],
    session: &mut VisionSession,
    on_event: &mut impl FnMut(IngestEvent),
) -> Result<Vec<ImageRecord>, VisionGrepError> {
    report_started(files.len(), on_event)?;
    let result = (|| {
        let mut records = Vec::with_capacity(files.len());
        for file in files {
            if let Some(embedding) = embed_image_with_skip_policy(root, file, session, on_event)? {
                records.push(ImageRecord {
                    path: root.display_image_path(&file.relative_path),
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

fn embed_image_with_skip_policy(
    root: &SearchRoot,
    file: &ImageFile,
    session: &mut VisionSession,
    on_event: &mut impl FnMut(IngestEvent),
) -> Result<Option<NormalizedEmbedding>, VisionGrepError> {
    match embed_image(&root.image_path(&file.relative_path), session) {
        Ok(embedding) => Ok(Some(embedding)),
        Err(
            error @ (VisionGrepError::ImageDecode { .. }
            | VisionGrepError::ImageTooLarge { .. }
            | VisionGrepError::InvalidImageDimensions { .. }),
        ) => {
            on_event(IngestEvent::ImageSkipped(error));
            Ok(None)
        }
        Err(error) => Err(error),
    }
}
