use std::path::{Path, PathBuf};

use crate::embedding::embed_text;
use crate::error::VisionGrepError;
use crate::index::{
    ImageFile, ImageIndex, IngestEvent, SearchRoot, StagedImageIndex, discover_images,
    embed_images, ingest_into_index,
};
use crate::model::{
    ArtifactEvent, ModelPaths, TextSession, VisionSession, ensure_text_artifacts,
    ensure_vision_artifacts, model_paths,
};
use crate::ranking::{Ranker, SearchResult};
use crate::timing::{CacheState, Phase, TimingRecorder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheMode {
    Use,
    Reindex,
    Disabled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArtifactVerification {
    Fast,
    Full,
}

pub(crate) struct SearchRequest {
    query: String,
    path: PathBuf,
    top: usize,
    threshold: f32,
    cache_mode: CacheMode,
    artifact_verification: ArtifactVerification,
}

impl SearchRequest {
    pub(crate) fn new(
        query: String,
        path: PathBuf,
        top: usize,
        threshold: f32,
        cache_mode: CacheMode,
        artifact_verification: ArtifactVerification,
    ) -> Self {
        Self {
            query,
            path,
            top,
            threshold,
            cache_mode,
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
    timing.record(Phase::PathValidationCanonicalization, path_started);

    let discovery_started = timing.start();
    let files = discover_images(&root)?;
    timing.record(Phase::RecursiveDiscoveryMetadata, discovery_started);
    timing.set_corpus_size(files.len());

    match request.cache_mode {
        CacheMode::Disabled => {
            timing.set_index_cache_state(CacheState::NotApplicable);
            timing.set_query_cache_state(CacheState::NotApplicable);
            search_without_cache(&root, &files, request, on_event, timing)
        }
        CacheMode::Use => {
            timing.set_index_cache_state(if ImageIndex::exists(root.filesystem_path()) {
                CacheState::Hit
            } else {
                CacheState::Absent
            });
            let index_started = timing.start();
            let index = ImageIndex::open(root.filesystem_path())?;
            timing.record(Phase::IndexOpenSchemaInitialization, index_started);
            search_with_cache(&root, &files, request, index, on_event, timing)
        }
        CacheMode::Reindex => {
            timing.set_index_cache_state(CacheState::Reindexed);
            reindex_and_search(&root, &files, request, on_event, timing)
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
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<SearchResult>, VisionGrepError> {
    if files.is_empty() {
        return Ok(Vec::new());
    }

    let paths = model_paths()?;
    ensure_vision(&paths, request.artifact_verification, on_event, timing)?;
    let session_started = timing.start();
    let mut vision = VisionSession::load(&paths)?;
    timing.record(Phase::ModelSessionConstruction, session_started);
    let image_records = embed_images(
        root,
        files,
        &mut vision,
        &mut |event| {
            on_event(SearchEvent::Index(event));
        },
        timing,
    )?;
    if image_records.is_empty() {
        return Ok(Vec::new());
    }

    ensure_text(&paths, request.artifact_verification, on_event, timing)?;
    let session_started = timing.start();
    let mut text = TextSession::load(&paths)?;
    timing.record(Phase::ModelSessionConstruction, session_started);
    let query_embedding = embed_text(&request.query, &mut text, timing)?;
    Ok(Ranker::new(&query_embedding, request.top, request.threshold).rank(image_records, timing))
}

/// Reconciles the on-disk index and loads only the model sessions required by the current state.
///
/// An unchanged directory plus an exact cached query needs no ONNX session. Changed images require
/// only the vision session, while a novel query requires only the text session.
fn search_with_cache(
    root: &SearchRoot,
    files: &[ImageFile],
    request: &SearchRequest,
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
    if !missing.is_empty() {
        timing.set_index_cache_state(CacheState::Changed);
        let paths = model_paths()?;
        ensure_vision(&paths, request.artifact_verification, on_event, timing)?;
        let session_started = timing.start();
        let mut vision = VisionSession::load(&paths)?;
        timing.record(Phase::ModelSessionConstruction, session_started);
        ingest_into_index(
            root,
            &mut index,
            missing,
            &mut vision,
            &mut |event| {
                on_event(SearchEvent::Index(event));
            },
            timing,
        )?;
    }

    search_index(root, request, &mut index, on_event, timing)
}

/// Builds a complete sibling database and exposes it only after every image was processed.
fn reindex_and_search(
    root: &SearchRoot,
    files: &[ImageFile],
    request: &SearchRequest,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<SearchResult>, VisionGrepError> {
    // Opening the active database first lets SQLite recover any hot journal before its main file is
    // eventually replaced. The connection is kept alive while the sibling index is constructed.
    let index_started = timing.start();
    let active_index = ImageIndex::open(root.filesystem_path())?;
    timing.record(Phase::IndexOpenSchemaInitialization, index_started);
    let mut vision = if files.is_empty() {
        None
    } else {
        let paths = model_paths()?;
        ensure_vision(&paths, request.artifact_verification, on_event, timing)?;
        let session_started = timing.start();
        let session = VisionSession::load(&paths)?;
        timing.record(Phase::ModelSessionConstruction, session_started);
        Some(session)
    };
    let staged_started = timing.start();
    let mut staged = StagedImageIndex::create(root.filesystem_path())?;
    timing.record(Phase::IndexOpenSchemaInitialization, staged_started);

    if let Some(vision) = &mut vision {
        ingest_into_index(
            root,
            staged.index_mut(),
            files,
            vision,
            &mut |event| {
                on_event(SearchEvent::Index(event));
            },
            timing,
        )?;
    }

    // Finish all fallible model work against the staged database. Once installation starts, the
    // rebuild is complete and this invocation no longer needs to reopen the replacement.
    let results = search_index(root, request, staged.index_mut(), on_event, timing)?;
    drop(active_index);
    staged.install()?;
    Ok(results)
}

fn search_index(
    root: &SearchRoot,
    request: &SearchRequest,
    index: &mut ImageIndex,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<Vec<SearchResult>, VisionGrepError> {
    let loading_started = timing.start();
    let image_records = index.all_embeddings(root.display_path())?;
    timing.record(Phase::CachedVectorLoadingDeserialization, loading_started);
    if image_records.is_empty() {
        return Ok(Vec::new());
    }

    let query_embedding = match index.query_embedding(&request.query)? {
        Some(embedding) => {
            timing.set_query_cache_state(CacheState::Hit);
            embedding
        }
        None => {
            timing.set_query_cache_state(CacheState::Miss);
            let paths = model_paths()?;
            ensure_text(&paths, request.artifact_verification, on_event, timing)?;
            let session_started = timing.start();
            let mut text = TextSession::load(&paths)?;
            timing.record(Phase::ModelSessionConstruction, session_started);
            let embedding = embed_text(&request.query, &mut text, timing)?;
            let writes_started = timing.start();
            index.upsert_query_embedding(&request.query, &embedding)?;
            timing.record(Phase::DatabaseWrites, writes_started);
            embedding
        }
    };

    Ok(Ranker::new(&query_embedding, request.top, request.threshold).rank(image_records, timing))
}

fn ensure_vision(
    paths: &ModelPaths,
    verification: ArtifactVerification,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<(), VisionGrepError> {
    ensure_vision_artifacts(
        paths,
        &mut |event| {
            on_event(SearchEvent::Artifact(event));
        },
        timing,
        verification,
    )
}

fn ensure_text(
    paths: &ModelPaths,
    verification: ArtifactVerification,
    on_event: &mut impl FnMut(SearchEvent),
    timing: &mut TimingRecorder,
) -> Result<(), VisionGrepError> {
    ensure_text_artifacts(
        paths,
        &mut |event| {
            on_event(SearchEvent::Artifact(event));
        },
        timing,
        verification,
    )
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
