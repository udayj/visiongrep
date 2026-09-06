use std::panic::catch_unwind;

use rayon::ThreadPool;
use rayon::prelude::*;

use crate::embedding::{NormalizedEmbedding, PreparedImage, embed_prepared_images, prepare_image};
use crate::error::{ImagePreparationError, VisionGrepError};
use crate::model::VisionSession;
use crate::timing::{Phase, TimingRecorder};

use super::scan::{ImageFile, SearchRoot};
use super::store::{ImageIndex, ImageRecord, ImageUpdate};

pub(super) const INDEX_BATCH_SIZE: usize = 256;
const VISION_BATCH_SIZE: usize = 8;
const MAX_PREPROCESSING_WORKERS: usize = 4;

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
    timing: &mut TimingRecorder,
) -> Result<(), VisionGrepError> {
    report_started(files.len(), on_event)?;
    let result = (|| {
        if files.is_empty() {
            return Ok(());
        }
        let pool = preprocessing_pool(files.len())?;
        for files in files.chunks(INDEX_BATCH_SIZE) {
            let mut updates = Vec::with_capacity(files.len());
            for files in files.chunks(VISION_BATCH_SIZE) {
                let embeddings = embed_image_batch_with_skip_policy(
                    root, files, session, &pool, on_event, timing,
                )?;
                for (file, embedding) in files.iter().zip(embeddings) {
                    let update = match embedding {
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
            }
            let writes_started = timing.start();
            index.apply_updates(updates)?;
            timing.record(Phase::DatabaseWrites, writes_started);
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
    timing: &mut TimingRecorder,
) -> Result<Vec<ImageRecord>, VisionGrepError> {
    report_started(files.len(), on_event)?;
    let result = (|| {
        if files.is_empty() {
            return Ok(Vec::new());
        }
        let pool = preprocessing_pool(files.len())?;
        let mut records = Vec::with_capacity(files.len());
        for files in files.chunks(VISION_BATCH_SIZE) {
            let embeddings =
                embed_image_batch_with_skip_policy(root, files, session, &pool, on_event, timing)?;
            for (file, embedding) in files.iter().zip(embeddings) {
                if let Some(embedding) = embedding {
                    records.push(ImageRecord {
                        path: root.display_image_path(&file.relative_path),
                        embedding,
                    });
                }
                on_event(IngestEvent::ImageProcessed);
            }
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

fn embed_image_batch_with_skip_policy(
    root: &SearchRoot,
    files: &[ImageFile],
    session: &mut VisionSession,
    pool: &ThreadPool,
    on_event: &mut impl FnMut(IngestEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<Option<NormalizedEmbedding>>, VisionGrepError> {
    let prepared = prepare_images_parallel(root, files, timing.is_enabled(), pool)?;
    infer_prepared_batch(files.len(), prepared, on_event, |valid_images| {
        embed_prepared_images(valid_images, session, timing)
    })
}

fn infer_prepared_batch(
    file_count: usize,
    prepared: Vec<Result<PreparedImage, ImagePreparationError>>,
    on_event: &mut impl FnMut(IngestEvent),
    infer: impl FnOnce(Vec<PreparedImage>) -> Result<Vec<NormalizedEmbedding>, VisionGrepError>,
) -> Result<Vec<Option<NormalizedEmbedding>>, VisionGrepError> {
    let mut valid_indices = Vec::with_capacity(file_count);
    let mut valid_images = Vec::with_capacity(file_count);
    for (index, result) in prepared.into_iter().enumerate() {
        match result {
            Ok(image) => {
                valid_indices.push(index);
                valid_images.push(image);
            }
            Err(error) => on_event(IngestEvent::ImageSkipped(error.into())),
        }
    }

    let expected = valid_indices.len();
    let embeddings = infer(valid_images)?;
    if embeddings.len() != expected {
        return Err(VisionGrepError::ImageBatchResultCount {
            expected,
            actual: embeddings.len(),
        });
    }
    let mut ordered = (0..file_count).map(|_| None).collect::<Vec<_>>();
    for (index, embedding) in valid_indices.into_iter().zip(embeddings) {
        ordered[index] = Some(embedding);
    }
    Ok(ordered)
}

// One operation owns the pool, so every batch reuses its workers without global configuration.
fn preprocessing_pool(file_count: usize) -> Result<ThreadPool, VisionGrepError> {
    let available = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    let worker_count = available
        .min(MAX_PREPROCESSING_WORKERS)
        .min(file_count)
        .max(1);
    rayon::ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build()
        .map_err(|source| VisionGrepError::ImagePreprocessingPool { source })
}

fn prepare_images_parallel(
    root: &SearchRoot,
    files: &[ImageFile],
    measure_timing: bool,
    pool: &ThreadPool,
) -> Result<Vec<Result<PreparedImage, ImagePreparationError>>, VisionGrepError> {
    pool.install(|| {
        // Rayon forwards worker panics here. Preserve the CLI's typed operational error boundary.
        catch_unwind(|| {
            // Slice iteration and Vec collection preserve input order, including failed images.
            files
                .par_iter()
                .map(|file| prepare_image(&root.image_path(&file.relative_path), measure_timing))
                .collect()
        })
    })
    .map_err(|_| VisionGrepError::ImagePreprocessingWorkerPanicked)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Instant;

    use image::{ImageFormat, Rgb, RgbImage};

    use super::*;
    use crate::index::discover_images;

    fn prepare_images_with_workers(
        root: &SearchRoot,
        files: &[ImageFile],
        measure_timing: bool,
        worker_count: usize,
    ) -> Result<Vec<Result<PreparedImage, ImagePreparationError>>, VisionGrepError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(worker_count.max(1))
            .build()
            .map_err(|source| VisionGrepError::ImagePreprocessingPool { source })?;
        prepare_images_parallel(root, files, measure_timing, &pool)
    }

    fn write_test_image(path: &std::path::Path, seed: u8) {
        let image = RgbImage::from_fn(96, 64, |x, y| {
            Rgb([
                seed.wrapping_add(x as u8),
                seed.wrapping_add(y as u8),
                seed.wrapping_add((x + y) as u8),
            ])
        });
        image.save_with_format(path, ImageFormat::Png).unwrap();
    }

    #[test]
    fn parallel_preprocessing_is_deterministic_and_ordered() {
        let directory = tempfile::tempdir().unwrap();
        for index in 0..8 {
            write_test_image(
                &directory.path().join(format!("{index:02}.png")),
                index as u8,
            );
        }
        let root = SearchRoot::resolve(directory.path()).unwrap();
        let files = discover_images(&root).unwrap();

        let single = prepare_images_with_workers(&root, &files, false, 1).unwrap();
        let parallel = prepare_images_with_workers(&root, &files, false, 4).unwrap();

        for (single, parallel) in single.into_iter().zip(parallel) {
            assert_eq!(single.unwrap().values, parallel.unwrap().values);
        }
    }

    #[test]
    fn parallel_preprocessing_propagates_errors_at_the_original_position() {
        let directory = tempfile::tempdir().unwrap();
        write_test_image(&directory.path().join("00.png"), 1);
        fs::write(directory.path().join("01.png"), b"not an image").unwrap();
        write_test_image(&directory.path().join("02.png"), 2);
        let root = SearchRoot::resolve(directory.path()).unwrap();
        let files = discover_images(&root).unwrap();

        let results = prepare_images_with_workers(&root, &files, false, 3).unwrap();

        assert!(results[0].is_ok());
        assert!(matches!(
            results[1],
            Err(ImagePreparationError::Decode { .. })
        ));
        assert!(results[2].is_ok());
    }

    #[test]
    fn batch_inference_errors_abort_without_dropping_images() {
        let directory = tempfile::tempdir().unwrap();
        write_test_image(&directory.path().join("00.png"), 1);
        write_test_image(&directory.path().join("01.png"), 2);
        let root = SearchRoot::resolve(directory.path()).unwrap();
        let files = discover_images(&root).unwrap();
        let prepared = prepare_images_with_workers(&root, &files, false, 2).unwrap();

        let error = infer_prepared_batch(files.len(), prepared, &mut |_| {}, |_| {
            Err(VisionGrepError::Io(std::io::Error::other(
                "injected inference failure",
            )))
        })
        .unwrap_err();

        assert!(matches!(error, VisionGrepError::Io(_)));
    }

    #[test]
    fn incomplete_batch_results_are_rejected() {
        let directory = tempfile::tempdir().unwrap();
        write_test_image(&directory.path().join("00.png"), 1);
        write_test_image(&directory.path().join("01.png"), 2);
        let root = SearchRoot::resolve(directory.path()).unwrap();
        let files = discover_images(&root).unwrap();
        let prepared = prepare_images_with_workers(&root, &files, false, 2).unwrap();

        let error = infer_prepared_batch(files.len(), prepared, &mut |_| {}, |_| Ok(Vec::new()))
            .unwrap_err();

        assert!(matches!(
            error,
            VisionGrepError::ImageBatchResultCount {
                expected: 2,
                actual: 0
            }
        ));
    }

    #[test]
    #[ignore = "release benchmark requiring VISIONGREP_PREPROCESS_DATASET"]
    fn preprocessing_worker_matrix() {
        let directory = std::env::var_os("VISIONGREP_PREPROCESS_DATASET")
            .map(std::path::PathBuf::from)
            .expect("VISIONGREP_PREPROCESS_DATASET must name a remote image corpus");
        let samples = std::env::var("VISIONGREP_PREPROCESS_SAMPLES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(5);
        assert!(samples > 0);
        let root = SearchRoot::resolve(&directory).unwrap();
        let files = discover_images(&root).unwrap();
        assert!(!files.is_empty());

        let mut reports = Vec::new();
        let mut expected_checksum = None;
        for worker_count in [1, 2, 4] {
            let mut elapsed_ms = Vec::with_capacity(samples);
            for _ in 0..samples {
                let started = Instant::now();
                let mut checksum = 0.0_f64;
                for files in files.chunks(VISION_BATCH_SIZE) {
                    for image in
                        prepare_images_with_workers(&root, files, false, worker_count).unwrap()
                    {
                        checksum += f64::from(image.unwrap().values[0]);
                    }
                }
                elapsed_ms.push(started.elapsed().as_secs_f64() * 1_000.0);
                if let Some(expected) = expected_checksum {
                    assert_eq!(checksum, expected);
                } else {
                    expected_checksum = Some(checksum);
                }
            }
            elapsed_ms.sort_by(f64::total_cmp);
            let p95_index = ((samples as f64 * 0.95).ceil() as usize)
                .saturating_sub(1)
                .min(samples - 1);
            let median_ms = elapsed_ms[samples / 2];
            reports.push(serde_json::json!({
                "corpus_size": files.len(),
                "workers": worker_count,
                "samples": samples,
                "median_ms": median_ms,
                "p95_ms": elapsed_ms[p95_index],
                "median_images_per_second": files.len() as f64 * 1_000.0 / median_ms,
            }));
        }

        println!("{}", serde_json::to_string(&reports).unwrap());
    }
}
