mod query;

use std::path::{Path, PathBuf};

pub(crate) use query::Query;
use query::{PreparedQuery, ResolvedQuery};

use crate::error::VisionGrepError;
use crate::index::{
    ImageFile, ImageIndex, IndexLocation, IngestEvent, SearchRoot, StagedImageIndex,
    discover_images, embed_images, ingest_into_index,
};
use crate::model::{ArtifactEvent, ArtifactVerification, Models, embedding_contract};
use crate::ranking::{Ranker, SearchResult};
use crate::timing::{CacheState, Phase, TimingRecorder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheMode {
    Use,
    Reindex,
    Disabled,
}

pub(crate) struct SearchRequest {
    query: Query,
    path: PathBuf,
    top: usize,
    threshold: f32,
    cache_mode: CacheMode,
    index_path: Option<PathBuf>,
    artifact_verification: ArtifactVerification,
}

impl SearchRequest {
    pub(crate) fn new(
        query: Query,
        path: PathBuf,
        top: usize,
        threshold: f32,
        cache_mode: CacheMode,
        index_path: Option<PathBuf>,
        artifact_verification: ArtifactVerification,
    ) -> Self {
        Self {
            query,
            path,
            top,
            threshold,
            cache_mode,
            index_path,
            artifact_verification,
        }
    }
}

pub(crate) enum SearchEvent {
    Index(IngestEvent),
    Artifact(ArtifactEvent),
}

/// Coordinates one search and reports progress without depending on terminal presentation.
pub(crate) fn search(
    request: &SearchRequest,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<SearchResult>, VisionGrepError> {
    let path_started = timing.start();
    validate_search_path(&request.path)?;
    let root = SearchRoot::resolve(&request.path)?;
    let query = request.query.resolve()?;
    timing.record(Phase::PathValidationCanonicalization, path_started);

    let discovery_started = timing.start();
    let files = discover_images(&root)?;
    timing.record(Phase::RecursiveDiscoveryMetadata, discovery_started);
    timing.set_corpus_size(files.len());

    match request.cache_mode {
        CacheMode::Disabled => {
            timing.set_index_cache_state(CacheState::NotApplicable);
            timing.set_query_cache_state(CacheState::NotApplicable);
            search_without_cache(&root, &files, request, query, on_event, timing)
        }
        CacheMode::Use => {
            let location =
                IndexLocation::resolve(root.filesystem_path(), request.index_path.as_deref())?;
            timing.set_index_cache_state(if ImageIndex::exists(&location) {
                CacheState::Hit
            } else {
                CacheState::Absent
            });
            let index_started = timing.start();
            let index = ImageIndex::open(&location, root.filesystem_path(), embedding_contract())?;
            timing.record(Phase::IndexOpenSchemaInitialization, index_started);
            search_with_cache(&root, &files, request, query, index, on_event, timing)
        }
        CacheMode::Reindex => {
            timing.set_index_cache_state(CacheState::Reindexed);
            let location =
                IndexLocation::resolve(root.filesystem_path(), request.index_path.as_deref())?;
            reindex_and_search(&root, &location, &files, request, query, on_event, timing)
        }
    }
}

/// Embeds every usable image for this invocation and deliberately leaves no persistent state.
///
/// Model initialization is deferred until an image is present, and the text model is not loaded
/// unless at least one image was embedded successfully.
fn search_without_cache(
    root: &SearchRoot,
    files: &[ImageFile],
    request: &SearchRequest,
    query: ResolvedQuery<'_>,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<SearchResult>, VisionGrepError> {
    let mut models = Models::new(request.artifact_verification);
    let image_records = if files.is_empty() {
        Vec::new()
    } else {
        let session = models.vision(&mut |event| on_event(SearchEvent::Artifact(event)), timing)?;
        embed_images(
            root,
            files,
            session,
            &mut |event| on_event(SearchEvent::Index(event)),
            timing,
        )?
    };

    let prepared = query.prepare(
        root,
        image_records,
        None,
        &mut models,
        &mut |event| on_event(SearchEvent::Artifact(event)),
        timing,
    )?;
    Ok(rank_query(prepared, request, timing))
}

/// Reconciles the on-disk index and loads only the model sessions required by the current state.
///
/// An unchanged directory plus cached text or an indexed query image needs no ONNX session.
/// Changed images and external image queries share a vision session; novel text uses a text session.
fn search_with_cache(
    root: &SearchRoot,
    files: &[ImageFile],
    request: &SearchRequest,
    query: ResolvedQuery<'_>,
    mut index: ImageIndex,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<SearchResult>, VisionGrepError> {
    let detection_started = timing.start();
    let plan = index.plan_reconciliation(files)?;
    timing.record(Phase::ChangedMissingImageDetection, detection_started);
    let reconciliation_started = timing.start();
    index.apply_reconciliation(&plan)?;
    timing.record(Phase::StaleEntryReconciliation, reconciliation_started);
    let missing = plan.missing();
    let mut models = Models::new(request.artifact_verification);
    if !missing.is_empty() {
        timing.set_index_cache_state(CacheState::Changed);
        let session = models.vision(&mut |event| on_event(SearchEvent::Artifact(event)), timing)?;
        ingest_into_index(
            root,
            &mut index,
            missing,
            session,
            &mut |event| {
                on_event(SearchEvent::Index(event));
            },
            timing,
        )?;
    }

    search_index(
        root,
        request,
        query,
        &mut index,
        &mut models,
        on_event,
        timing,
    )
}

/// Builds a complete sibling database and exposes it only after every image was processed.
fn reindex_and_search(
    root: &SearchRoot,
    location: &IndexLocation,
    files: &[ImageFile],
    request: &SearchRequest,
    query: ResolvedQuery<'_>,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<SearchResult>, VisionGrepError> {
    // Opening the active database first lets SQLite recover any hot journal before its main file is
    // eventually replaced. The connection is kept alive while the sibling index is constructed.
    let index_started = timing.start();
    let active_index = ImageIndex::open(location, root.filesystem_path(), embedding_contract())?;
    timing.record(Phase::IndexOpenSchemaInitialization, index_started);
    let mut models = Models::new(request.artifact_verification);
    let staged_started = timing.start();
    let mut staged =
        StagedImageIndex::create(location, root.filesystem_path(), embedding_contract())?;
    timing.record(Phase::IndexOpenSchemaInitialization, staged_started);

    if !files.is_empty() {
        let session = models.vision(&mut |event| on_event(SearchEvent::Artifact(event)), timing)?;
        ingest_into_index(
            root,
            staged.index_mut(),
            files,
            session,
            &mut |event| {
                on_event(SearchEvent::Index(event));
            },
            timing,
        )?;
    }

    // Finish all fallible model work against the staged database. Once installation starts, the
    // rebuild is complete and this invocation no longer needs to reopen the replacement.
    let results = search_index(
        root,
        request,
        query,
        staged.index_mut(),
        &mut models,
        on_event,
        timing,
    )?;
    drop(active_index);
    staged.install()?;
    Ok(results)
}

fn search_index(
    root: &SearchRoot,
    request: &SearchRequest,
    query: ResolvedQuery<'_>,
    index: &mut ImageIndex,
    models: &mut Models,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<SearchResult>, VisionGrepError> {
    let loading_started = timing.start();
    let image_records = index.all_embeddings(root.display_path())?;
    timing.record(Phase::CachedVectorLoadingDeserialization, loading_started);
    let prepared = query.prepare(
        root,
        image_records,
        Some(index),
        models,
        &mut |event| on_event(SearchEvent::Artifact(event)),
        timing,
    )?;
    Ok(rank_query(prepared, request, timing))
}

fn rank_query(
    prepared: Option<PreparedQuery>,
    request: &SearchRequest,
    timing: &mut TimingRecorder,
) -> Vec<SearchResult> {
    let Some(prepared) = prepared else {
        return Vec::new();
    };
    Ranker::new(&prepared.embedding, request.top, request.threshold).rank(prepared.images, timing)
}

fn validate_search_path(path: &Path) -> Result<(), VisionGrepError> {
    if !path.exists() {
        return Err(VisionGrepError::SearchPathMissing {
            path: path.to_owned(),
        });
    }

    if !path.is_dir() {
        return Err(VisionGrepError::SearchPathNotDirectory {
            path: path.to_owned(),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::PermissionsExt;
    use std::time::UNIX_EPOCH;

    use image::{Rgb, RgbImage};
    use rusqlite::{Connection, params};

    use super::*;
    use crate::embedding::NormalizedEmbedding;

    fn write_image(path: &Path) {
        RgbImage::from_pixel(16, 16, Rgb([120, 30, 90]))
            .save(path)
            .unwrap();
    }

    fn embedding() -> NormalizedEmbedding {
        let mut values = vec![0.0; crate::embedding::EMBEDDING_DIM];
        values[0] = 1.0;
        NormalizedEmbedding::from_model_output(values).unwrap()
    }

    fn seed_index(root: &Path, paths: &[&Path]) -> IndexLocation {
        let canonical_root = root.canonicalize().unwrap();
        let location = IndexLocation::resolve(&canonical_root, None).unwrap();
        ImageIndex::open(&location, &canonical_root, embedding_contract()).unwrap();
        let connection = Connection::open(location.path()).unwrap();
        for path in paths {
            let metadata = fs::metadata(root.join(path)).unwrap();
            let mtime_ns = i64::try_from(
                metadata
                    .modified()
                    .unwrap()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            )
            .unwrap();
            connection
                .execute(
                    "INSERT INTO images (path, mtime_ns, size, embedding) VALUES (?1, ?2, ?3, ?4)",
                    params![
                        path.as_os_str().as_bytes(),
                        mtime_ns,
                        i64::try_from(metadata.len()).unwrap(),
                        embedding().to_le_bytes()
                    ],
                )
                .unwrap();
        }
        location
    }

    fn image_request(root: &Path, query: &Path, cache_mode: CacheMode) -> SearchRequest {
        SearchRequest::new(
            Query::Image(query.to_owned()),
            root.to_owned(),
            1,
            0.25,
            cache_mode,
            None,
            ArtifactVerification::Fast,
        )
    }

    #[test]
    fn indexed_image_queries_reuse_embeddings_and_exclude_the_resolved_query() {
        let directory = tempfile::tempdir().unwrap();
        let query_path = directory.path().join("query.png");
        let duplicate_path = directory.path().join("duplicate.png");
        write_image(&query_path);
        fs::copy(&query_path, &duplicate_path).unwrap();
        let location = seed_index(
            directory.path(),
            &[Path::new("query.png"), Path::new("duplicate.png")],
        );
        let aliases = tempfile::tempdir().unwrap();
        let alias = aliases.path().join("alias.png");
        std::os::unix::fs::symlink(&query_path, &alias).unwrap();

        for query in [&query_path, &alias] {
            let request = image_request(&directory.path().join("."), query, CacheMode::Use);
            let mut timing = TimingRecorder::new(true, crate::model::timing_metadata());
            let results = search(
                &request,
                &mut |_| panic!("cached search should not load models or ingest images"),
                &mut timing,
            )
            .unwrap();
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].path, directory.path().join("./duplicate.png"));
            assert_eq!(results[0].score, 1.0);
            assert_eq!(timing.phase_invocations(Phase::ModelSessionConstruction), 0);
        }
        let index = ImageIndex::open(
            &location,
            &directory.path().canonicalize().unwrap(),
            embedding_contract(),
        )
        .unwrap();
        assert_eq!(index.all_embeddings(directory.path()).unwrap().len(), 2);
        let connection = Connection::open(location.path()).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT COUNT(*) FROM queries", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn a_corpus_containing_only_the_query_returns_no_matches() {
        let directory = tempfile::tempdir().unwrap();
        let query = directory.path().join("query.png");
        write_image(&query);
        seed_index(directory.path(), &[Path::new("query.png")]);
        let request = image_request(directory.path(), &query, CacheMode::Use);
        let mut timing = TimingRecorder::disabled(crate::model::timing_metadata());
        assert!(
            search(
                &request,
                &mut |_| panic!("unexpected model load"),
                &mut timing
            )
            .unwrap()
            .is_empty()
        );
    }

    #[test]
    fn invalid_query_images_fail_even_for_empty_searches() {
        let images = tempfile::tempdir().unwrap();
        let corrupt = images.path().join("corrupt.png");
        fs::write(&corrupt, b"not an image").unwrap();
        let missing = images.path().join("missing.png");
        for mode in [CacheMode::Use, CacheMode::Reindex, CacheMode::Disabled] {
            let root = tempfile::tempdir().unwrap();
            for query in [&corrupt, &missing, &images.path().to_owned()] {
                let request = image_request(root.path(), query, mode);
                let mut timing = TimingRecorder::new(true, crate::model::timing_metadata());
                let result = search(
                    &request,
                    &mut |_| panic!("invalid query should not need models"),
                    &mut timing,
                );
                assert!(matches!(
                    result,
                    Err(VisionGrepError::ImageDecode { .. }
                        | VisionGrepError::QueryImageOpen { .. }
                        | VisionGrepError::QueryImageNotFile { .. })
                ));
                assert_eq!(timing.phase_invocations(Phase::ModelSessionConstruction), 0);
            }
        }
    }

    #[test]
    fn external_query_does_not_populate_an_empty_index() {
        let images = tempfile::tempdir().unwrap();
        let query = images.path().join("external.png");
        write_image(&query);
        for mode in [CacheMode::Use, CacheMode::Reindex, CacheMode::Disabled] {
            let root = tempfile::tempdir().unwrap();
            let request = image_request(root.path(), &query, mode);
            let mut timing = TimingRecorder::disabled(crate::model::timing_metadata());
            assert!(
                search(
                    &request,
                    &mut |_| panic!("empty corpus should not need models"),
                    &mut timing
                )
                .unwrap()
                .is_empty()
            );
            let location = IndexLocation::resolve(root.path(), None).unwrap();
            if mode == CacheMode::Disabled {
                assert!(!location.path().exists());
            } else {
                let index = ImageIndex::open(
                    &location,
                    &root.path().canonicalize().unwrap(),
                    embedding_contract(),
                )
                .unwrap();
                assert!(index.all_embeddings(root.path()).unwrap().is_empty());
            }
        }
        assert_eq!(fs::read_dir(images.path()).unwrap().count(), 1);
    }

    #[test]
    fn cached_text_queries_keep_the_existing_no_model_path() {
        let directory = tempfile::tempdir().unwrap();
        write_image(&directory.path().join("image.png"));
        let location = seed_index(directory.path(), &[Path::new("image.png")]);
        let mut index = ImageIndex::open(
            &location,
            &directory.path().canonicalize().unwrap(),
            embedding_contract(),
        )
        .unwrap();
        index.upsert_query_embedding("robot", &embedding()).unwrap();
        drop(index);
        let request = SearchRequest::new(
            Query::Text("robot".to_owned()),
            directory.path().to_owned(),
            5,
            0.25,
            CacheMode::Use,
            None,
            ArtifactVerification::Fast,
        );
        let mut timing = TimingRecorder::disabled(crate::model::timing_metadata());
        assert_eq!(
            search(
                &request,
                &mut |_| panic!("text query is cached"),
                &mut timing
            )
            .unwrap()
            .len(),
            1
        );
    }

    #[test]
    fn custom_index_supports_a_read_only_empty_search_root() {
        let root = tempfile::tempdir().unwrap();
        let index_directory = tempfile::tempdir().unwrap();
        let index_path = index_directory.path().join("index.db");
        let original_permissions = fs::metadata(root.path()).unwrap().permissions();
        fs::set_permissions(root.path(), fs::Permissions::from_mode(0o555)).unwrap();
        let request = SearchRequest::new(
            Query::Text("robot".to_owned()),
            root.path().to_owned(),
            5,
            0.25,
            CacheMode::Use,
            Some(index_path.clone()),
            ArtifactVerification::Fast,
        );
        let mut timing = TimingRecorder::disabled(crate::model::timing_metadata());

        let results = search(&request, &mut |_| {}, &mut timing);
        fs::set_permissions(root.path(), original_permissions).unwrap();

        assert!(results.unwrap().is_empty());
        assert!(index_path.exists());
        assert!(!root.path().join(".visiongrep.db").exists());
    }
}
