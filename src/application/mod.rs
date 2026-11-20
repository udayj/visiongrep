mod query;

use std::path::{Path, PathBuf};

pub(crate) use query::Query;
use query::{PreparedQuery, ResolvedQuery};

use crate::error::VisionGrepError;
use crate::index::{
    ImageFile, ImageIndex, IndexLocation, IngestEvent, SearchRoot, discover_images, embed_images,
    ingest_into_index,
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

    pub(crate) fn can_use_server(&self) -> bool {
        self.cache_mode != CacheMode::Disabled
            && self.artifact_verification == ArtifactVerification::Fast
    }

    pub(crate) fn is_reindex(&self) -> bool {
        self.cache_mode == CacheMode::Reindex
    }

    pub(crate) fn artifact_verification(&self) -> ArtifactVerification {
        self.artifact_verification
    }

    pub(crate) fn query(&self) -> &Query {
        &self.query
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }

    pub(crate) fn index_path(&self) -> Option<&Path> {
        self.index_path.as_deref()
    }

    pub(crate) fn top(&self) -> usize {
        self.top
    }

    pub(crate) fn threshold(&self) -> f32 {
        self.threshold
    }
}

pub(crate) enum SearchEvent {
    WaitingForIndex,
    Index(IngestEvent),
    Artifact(ArtifactEvent),
}

#[derive(Clone, Copy)]
struct SearchOptions {
    top: usize,
    threshold: f32,
    cache_mode: CacheMode,
}

/// Coordinates one search and reports progress without depending on terminal presentation.
pub(crate) fn search_with_models(
    request: &SearchRequest,
    models: &Models,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<SearchResult>, VisionGrepError> {
    let path_started = timing.start();
    let root = resolve_root(&request.path)?;
    let query = request.query.resolve()?;
    timing.record(Phase::PathValidationCanonicalization, path_started);

    let options = SearchOptions {
        top: request.top,
        threshold: request.threshold,
        cache_mode: request.cache_mode,
    };
    if request.cache_mode == CacheMode::Disabled {
        return search_without_cache(&root, query, options, models, on_event, timing);
    }

    let location = IndexLocation::resolve(root.filesystem_path(), request.index_path.as_deref())?;
    search_cached(&root, &location, query, options, models, on_event, timing)
}

fn discover(
    root: &SearchRoot,
    timing: &mut TimingRecorder,
) -> Result<Vec<ImageFile>, VisionGrepError> {
    let started = timing.start();
    let files = discover_images(root)?;
    timing.record(Phase::RecursiveDiscoveryMetadata, started);
    timing.set_corpus_size(files.len());
    Ok(files)
}

/// Embeds every usable image for this invocation without reading or writing the index.
///
/// Model initialization is deferred until an image is present. Text inference loads its model
/// only after at least one corpus image has been embedded.
fn search_without_cache(
    root: &SearchRoot,
    query: ResolvedQuery<'_>,
    options: SearchOptions,
    models: &Models,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<SearchResult>, VisionGrepError> {
    timing.set_index_cache_state(CacheState::NotApplicable);
    timing.set_query_cache_state(CacheState::NotApplicable);
    let files = discover(root, timing)?;
    let image_records = if files.is_empty() {
        Vec::new()
    } else {
        embed_images(
            root,
            &files,
            models,
            &mut |event| on_event(SearchEvent::Index(event)),
            timing,
        )?
    };
    let prepared = query.prepare(
        root,
        image_records,
        None,
        models,
        &mut |event| on_event(SearchEvent::Artifact(event)),
        timing,
    )?;
    Ok(rank_query(prepared, options.top, options.threshold, timing))
}

fn refresh_index(
    root: &SearchRoot,
    files: &[ImageFile],
    index: &mut ImageIndex,
    models: &Models,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<(), VisionGrepError> {
    let detection_started = timing.start();
    let plan = index.plan_reconciliation(files)?;
    timing.record(Phase::ChangedMissingImageDetection, detection_started);
    let reconciliation_started = timing.start();
    index.apply_reconciliation(&plan)?;
    timing.record(Phase::StaleEntryReconciliation, reconciliation_started);
    let missing = plan.missing();
    if !missing.is_empty() {
        timing.set_index_cache_state(CacheState::Changed);
        ingest_into_index(
            root,
            index,
            missing,
            models,
            &mut |event| on_event(SearchEvent::Index(event)),
            timing,
        )?;
    }
    Ok(())
}

/// Brings a directory's index up to date and prepares both models without running a query.
///
/// Loading the text session installs missing artifacts and warms the optimized graph cache when
/// supported. Later searches reuse indexed image embeddings and can reuse the loaded text
/// session when they use the same server.
pub(crate) fn prepare_index(
    path: &Path,
    index_path: Option<&Path>,
    models: &Models,
    on_event: &mut impl FnMut(SearchEvent),
) -> Result<usize, VisionGrepError> {
    let mut timing = TimingRecorder::new(false, crate::model::timing_metadata());
    let root = resolve_root(path)?;
    let location = IndexLocation::resolve(root.filesystem_path(), index_path)?;
    let mut index = ImageIndex::open(
        &location,
        root.filesystem_path(),
        embedding_contract(),
        || {
            on_event(SearchEvent::WaitingForIndex);
        },
    )?;
    let files = discover_images(&root)?;
    refresh_index(&root, &files, &mut index, models, on_event, &mut timing)?;
    models.with_text(
        &mut |event| on_event(SearchEvent::Artifact(event)),
        &mut timing,
        |_, _| Ok(()),
    )?;
    index.image_count()
}

fn search_cached(
    root: &SearchRoot,
    location: &IndexLocation,
    query: ResolvedQuery<'_>,
    options: SearchOptions,
    models: &Models,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<SearchResult>, VisionGrepError> {
    timing.set_index_cache_state(if options.cache_mode == CacheMode::Reindex {
        CacheState::Reindexed
    } else if ImageIndex::exists(location) {
        CacheState::Hit
    } else {
        CacheState::Absent
    });
    let started = timing.start();
    let mut index = ImageIndex::open(
        location,
        root.filesystem_path(),
        embedding_contract(),
        || {
            on_event(SearchEvent::WaitingForIndex);
        },
    )?;
    timing.record(Phase::IndexOpenSchemaInitialization, started);
    let files = discover(root, timing)?;
    if options.cache_mode == CacheMode::Reindex {
        let started = timing.start();
        let mut staged = index.stage(root.filesystem_path(), embedding_contract())?;
        timing.record(Phase::IndexOpenSchemaInitialization, started);
        if !files.is_empty() {
            ingest_into_index(
                root,
                staged.index_mut(),
                &files,
                models,
                &mut |event| on_event(SearchEvent::Index(event)),
                timing,
            )?;
        }
        let results = search_index(
            root,
            query,
            options,
            staged.index_mut(),
            models,
            on_event,
            timing,
        )?;
        staged.install()?;
        return Ok(results);
    }
    refresh_index(root, &files, &mut index, models, on_event, timing)?;
    search_index(root, query, options, &mut index, models, on_event, timing)
}

fn search_index(
    root: &SearchRoot,
    query: ResolvedQuery<'_>,
    ranking: SearchOptions,
    index: &mut ImageIndex,
    models: &Models,
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
    Ok(rank_query(prepared, ranking.top, ranking.threshold, timing))
}

fn rank_query(
    prepared: Option<PreparedQuery>,
    top: usize,
    threshold: f32,
    timing: &mut TimingRecorder,
) -> Vec<SearchResult> {
    let Some(prepared) = prepared else {
        return Vec::new();
    };
    Ranker::new(&prepared.embedding, top, threshold).rank(prepared.images, timing)
}

pub(crate) fn resolve_root(path: &Path) -> Result<SearchRoot, VisionGrepError> {
    validate_search_path(path)?;
    SearchRoot::resolve(path)
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

    fn search(
        request: &SearchRequest,
        on_event: &mut impl FnMut(SearchEvent),
        timing: &mut TimingRecorder,
    ) -> Result<Vec<SearchResult>, VisionGrepError> {
        let models = Models::new(request.artifact_verification());
        search_with_models(request, &models, on_event, timing)
    }

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
        ImageIndex::open(&location, &canonical_root, embedding_contract(), || {}).unwrap();
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
            || {},
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
                    || {},
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
            || {},
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
