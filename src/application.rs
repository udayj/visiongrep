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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CacheMode {
    Use,
    Reindex,
    Disabled,
}

pub(crate) struct SearchRequest {
    query: String,
    path: PathBuf,
    top: usize,
    threshold: f32,
    cache_mode: CacheMode,
}

impl SearchRequest {
    pub(crate) fn new(
        query: String,
        path: PathBuf,
        top: usize,
        threshold: f32,
        cache_mode: CacheMode,
    ) -> Self {
        Self {
            query,
            path,
            top,
            threshold,
            cache_mode,
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
) -> Result<Vec<SearchResult>, VisionGrepError> {
    validate_search_path(&request.path)?;
    let root = SearchRoot::resolve(&request.path)?;
    let files = discover_images(&root)?;

    match request.cache_mode {
        CacheMode::Disabled => search_without_cache(&root, &files, request, on_event),
        CacheMode::Use => {
            let index = ImageIndex::open(root.filesystem_path())?;
            search_with_cache(&root, &files, request, index, on_event)
        }
        CacheMode::Reindex => reindex_and_search(&root, &files, request, on_event),
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
) -> Result<Vec<SearchResult>, VisionGrepError> {
    if files.is_empty() {
        return Ok(Vec::new());
    }

    let paths = model_paths()?;
    ensure_vision(&paths, on_event)?;
    let mut vision = VisionSession::load(&paths)?;
    let image_records = embed_images(root, files, &mut vision, &mut |event| {
        on_event(SearchEvent::Index(event));
    })?;
    if image_records.is_empty() {
        return Ok(Vec::new());
    }

    ensure_text(&paths, on_event)?;
    let mut text = TextSession::load(&paths)?;
    let query_embedding = embed_text(&request.query, &mut text)?;
    Ok(Ranker::new(&query_embedding, request.top, request.threshold).rank(image_records))
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
) -> Result<Vec<SearchResult>, VisionGrepError> {
    index.remove_stale_entries(files)?;
    let missing = index.images_needing_embedding(files)?;
    if !missing.is_empty() {
        let paths = model_paths()?;
        ensure_vision(&paths, on_event)?;
        let mut vision = VisionSession::load(&paths)?;
        ingest_into_index(root, &mut index, &missing, &mut vision, &mut |event| {
            on_event(SearchEvent::Index(event));
        })?;
    }

    search_index(root, request, &mut index, on_event)
}

/// Builds a complete sibling database and exposes it only after every image was processed.
fn reindex_and_search(
    root: &SearchRoot,
    files: &[ImageFile],
    request: &SearchRequest,
    on_event: &mut impl FnMut(SearchEvent),
) -> Result<Vec<SearchResult>, VisionGrepError> {
    // Opening the active database first lets SQLite recover any hot journal before its main file is
    // eventually replaced. The connection is kept alive while the sibling index is constructed.
    let active_index = ImageIndex::open(root.filesystem_path())?;
    let mut vision = if files.is_empty() {
        None
    } else {
        let paths = model_paths()?;
        ensure_vision(&paths, on_event)?;
        Some(VisionSession::load(&paths)?)
    };
    let mut staged = StagedImageIndex::create(root.filesystem_path())?;

    if let Some(vision) = &mut vision {
        ingest_into_index(root, staged.index_mut(), files, vision, &mut |event| {
            on_event(SearchEvent::Index(event));
        })?;
    }

    // Finish all fallible model work against the staged database. Once installation starts, the
    // rebuild is complete and this invocation no longer needs to reopen the replacement.
    let results = search_index(root, request, staged.index_mut(), on_event)?;
    drop(active_index);
    staged.install()?;
    Ok(results)
}

fn search_index(
    root: &SearchRoot,
    request: &SearchRequest,
    index: &mut ImageIndex,
    on_event: &mut impl FnMut(SearchEvent),
) -> Result<Vec<SearchResult>, VisionGrepError> {
    let image_records = index.all_embeddings(root.display_path())?;
    if image_records.is_empty() {
        return Ok(Vec::new());
    }

    let query_embedding = match index.query_embedding(&request.query)? {
        Some(embedding) => embedding,
        None => {
            let paths = model_paths()?;
            ensure_text(&paths, on_event)?;
            let mut text = TextSession::load(&paths)?;
            let embedding = embed_text(&request.query, &mut text)?;
            index.upsert_query_embedding(&request.query, &embedding)?;
            embedding
        }
    };

    Ok(Ranker::new(&query_embedding, request.top, request.threshold).rank(image_records))
}

fn ensure_vision(
    paths: &ModelPaths,
    on_event: &mut impl FnMut(SearchEvent),
) -> Result<(), VisionGrepError> {
    ensure_vision_artifacts(paths, &mut |event| {
        on_event(SearchEvent::Artifact(event));
    })
}

fn ensure_text(
    paths: &ModelPaths,
    on_event: &mut impl FnMut(SearchEvent),
) -> Result<(), VisionGrepError> {
    ensure_text_artifacts(paths, &mut |event| {
        on_event(SearchEvent::Artifact(event));
    })
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
