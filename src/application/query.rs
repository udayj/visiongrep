use std::path::{Path, PathBuf};

use crate::embedding::{NormalizedEmbedding, embed_prepared_image, embed_text, prepare_image};
use crate::error::VisionGrepError;
use crate::index::{ImageIndex, ImageRecord, SearchRoot};
use crate::model::{ArtifactEvent, Models};
use crate::timing::{CacheState, Phase, TimingRecorder};

#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Query {
    Text(String),
    Image(PathBuf),
}

pub(super) enum ResolvedQuery<'a> {
    Text(&'a str),
    Image(PathBuf),
}

/// A query embedding and its candidates after applying query-specific exclusions.
pub(super) struct PreparedQuery {
    pub(super) embedding: NormalizedEmbedding,
    pub(super) images: Vec<ImageRecord>,
}

impl Query {
    pub(super) fn resolve(&self) -> Result<ResolvedQuery<'_>, VisionGrepError> {
        match self {
            Self::Text(text) => Ok(ResolvedQuery::Text(text)),
            Self::Image(path) => {
                let open_error = |source| VisionGrepError::QueryImageOpen {
                    path: path.clone(),
                    source,
                };
                let canonical = path.canonicalize().map_err(open_error)?;
                if !std::fs::metadata(&canonical).map_err(open_error)?.is_file() {
                    return Err(VisionGrepError::QueryImageNotFile { path: path.clone() });
                }
                // A cache hit must still reject an inaccessible query. Check the file type first
                // so opening a pipe or other special file cannot block waiting for a writer.
                std::fs::File::open(&canonical).map_err(open_error)?;
                Ok(ResolvedQuery::Image(canonical))
            }
        }
    }
}

impl ResolvedQuery<'_> {
    /// Obtains a query embedding and removes the query image from the candidates.
    ///
    /// Candidates must reflect the current corpus. The optional index caches text queries only;
    /// image queries reuse a candidate embedding or remain transient. An empty corpus still
    /// validates an external image, but does not require model loading or inference.
    pub(super) fn prepare(
        self,
        root: &SearchRoot,
        mut images: Vec<ImageRecord>,
        index: Option<&mut ImageIndex>,
        models: &mut Models,
        on_event: &mut impl FnMut(ArtifactEvent),
        timing: &mut TimingRecorder,
    ) -> Result<Option<PreparedQuery>, VisionGrepError> {
        let embedding = match self {
            Self::Text(text) => {
                if images.is_empty() {
                    return Ok(None);
                }
                prepare_text(text, index, models, on_event, timing)?
            }
            Self::Image(path) => {
                timing.set_query_cache_state(CacheState::NotApplicable);
                match take_query_image(root, &path, &mut images) {
                    Some(embedding) => {
                        if index.is_some() {
                            timing.set_query_cache_state(CacheState::Hit);
                        }
                        embedding
                    }
                    None => {
                        // A corrupt query is an error even when there are no candidates to rank.
                        let prepared = prepare_image(&path, timing.is_enabled())?;
                        if images.is_empty() {
                            return Ok(None);
                        }
                        let session = models.vision(on_event, timing)?;
                        // Only discovered corpus images belong in the persistent image index.
                        embed_prepared_image(prepared, session, timing)?
                    }
                }
            }
        };

        Ok(Some(PreparedQuery { embedding, images }))
    }
}

fn prepare_text(
    text: &str,
    index: Option<&mut ImageIndex>,
    models: &mut Models,
    on_event: &mut impl FnMut(ArtifactEvent),
    timing: &mut TimingRecorder,
) -> Result<NormalizedEmbedding, VisionGrepError> {
    match &index {
        Some(index) => {
            if let Some(embedding) = index.query_embedding(text)? {
                timing.set_query_cache_state(CacheState::Hit);
                return Ok(embedding);
            }
            timing.set_query_cache_state(CacheState::Miss);
        }
        None => timing.set_query_cache_state(CacheState::NotApplicable),
    }

    let mut session = models.load_text(on_event, timing)?;
    let embedding = embed_text(text, &mut session, timing)?;
    if let Some(index) = index {
        let writes_started = timing.start();
        index.upsert_query_embedding(text, &embedding)?;
        timing.record(Phase::DatabaseWrites, writes_started);
    }
    Ok(embedding)
}

fn take_query_image(
    root: &SearchRoot,
    canonical_query_path: &Path,
    images: &mut Vec<ImageRecord>,
) -> Option<NormalizedEmbedding> {
    // Discovery does not follow symlinks. Resolving the query once is enough to match a corpus
    // path, including a query supplied through a symlink or a differently spelled search root.
    let relative_path = canonical_query_path
        .strip_prefix(root.filesystem_path())
        .ok()?;
    let display_path = root.display_image_path(relative_path);
    let position = images.iter().position(|image| image.path == display_path)?;
    // Removing the query transfers its embedding and excludes it before top-K selection.
    Some(images.swap_remove(position).embedding)
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;
    use std::fs;
    use std::os::unix::ffi::OsStrExt;

    use super::*;
    use crate::model::ArtifactVerification;

    fn embedding() -> NormalizedEmbedding {
        let mut values = vec![0.0; crate::embedding::EMBEDDING_DIM];
        values[0] = 1.0;
        NormalizedEmbedding::from_model_output(values).unwrap()
    }

    #[test]
    fn no_cache_reuses_an_in_memory_query_and_preserves_native_paths() {
        let directory = tempfile::tempdir().unwrap();
        let root = SearchRoot::resolve(directory.path()).unwrap();
        let query_name = Path::new(OsStr::from_bytes(b"query-\xff.png"));
        let query = root.image_path(query_name);
        let matching = root.display_image_path(Path::new("duplicate.png"));
        let images = vec![
            ImageRecord {
                path: root.display_image_path(query_name),
                embedding: embedding(),
            },
            ImageRecord {
                path: matching.clone(),
                embedding: embedding(),
            },
        ];
        let mut models = Models::new(ArtifactVerification::Fast);
        let mut timing = TimingRecorder::disabled(crate::model::timing_metadata());
        let prepared = ResolvedQuery::Image(query)
            .prepare(
                &root,
                images,
                None,
                &mut models,
                &mut |_| panic!("query embedding is already available"),
                &mut timing,
            )
            .unwrap()
            .unwrap();

        assert_eq!(prepared.images.len(), 1);
        assert_eq!(prepared.images[0].path, matching);
        assert_eq!(prepared.embedding.as_slice(), embedding().as_slice());
        assert!(!directory.path().join(".visiongrep.db").exists());
    }

    #[test]
    fn image_query_resolution_uses_the_canonical_file_identity() {
        let directory = tempfile::tempdir().unwrap();
        let image = directory.path().join("image.png");
        let alias = directory.path().join("alias.png");
        fs::write(&image, []).unwrap();
        std::os::unix::fs::symlink(&image, &alias).unwrap();

        let query = Query::Image(alias);
        let ResolvedQuery::Image(resolved) = query.resolve().unwrap() else {
            panic!("expected an image query");
        };
        assert_eq!(resolved, image.canonicalize().unwrap());

        assert!(matches!(
            Query::Image(directory.path().to_owned()).resolve(),
            Err(VisionGrepError::QueryImageNotFile { .. })
        ));
        assert!(matches!(
            Query::Image(directory.path().join("missing.png")).resolve(),
            Err(VisionGrepError::QueryImageOpen { .. })
        ));
    }
}
