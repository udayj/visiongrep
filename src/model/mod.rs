mod artifacts;
mod graph_cache;
mod runtime;

use std::sync::Mutex;

use artifacts::ModelPaths;
pub(crate) use artifacts::{
    ArtifactEvent, ArtifactVerification, cache_dir, embedding_contract, model_paths,
    timing_metadata,
};
pub(crate) use runtime::{TextSession, VisionSession};

use crate::error::VisionGrepError;
use crate::timing::{Phase, TimingRecorder};

#[derive(Clone, Copy)]
pub(crate) enum SessionLifetime {
    Search,
    Process,
}

/// Loads model resources on demand and owns them for the requested lifetime.
pub(crate) struct Models {
    verification: ArtifactVerification,
    // When both are needed, acquire the index file lock before either model lock.
    vision: Mutex<Option<VisionSession>>,
    text: Mutex<Option<TextSession>>,
    lifetime: SessionLifetime,
}

impl Models {
    pub(crate) fn new(verification: ArtifactVerification) -> Self {
        Self::with_lifetime(verification, SessionLifetime::Search)
    }

    pub(crate) fn with_lifetime(
        verification: ArtifactVerification,
        lifetime: SessionLifetime,
    ) -> Self {
        Self {
            verification,
            vision: Mutex::new(None),
            text: Mutex::new(None),
            lifetime,
        }
    }

    /// Preloads the text session when both artifacts match their verification manifests.
    pub(crate) fn preload_text(&self) {
        let Ok(model_dir) = artifacts::model_dir() else {
            return;
        };
        let paths = ModelPaths::in_dir(&model_dir);
        self.preload_text_at(&paths);
    }

    fn preload_text_at(&self, paths: &ModelPaths) {
        let Ok(mut text) = self.text.lock() else {
            return;
        };
        if text.is_some() {
            return;
        }

        if !matches!(artifacts::text_artifacts_verified(paths), Ok(true)) {
            return;
        }

        let mut timing = TimingRecorder::new(false, timing_metadata());
        *text = load_text_session(paths, &mut timing).ok();
    }

    /// Reuses the vision session across corpus ingestion and query-image inference.
    pub(crate) fn with_vision<T>(
        &self,
        on_event: &mut impl FnMut(ArtifactEvent),
        timing: &mut TimingRecorder,
        run: impl FnOnce(&mut VisionSession, &mut TimingRecorder) -> Result<T, VisionGrepError>,
    ) -> Result<T, VisionGrepError> {
        let mut vision = self
            .vision
            .lock()
            .map_err(|_| VisionGrepError::ModelSessionPoisoned { model: "vision" })?;
        let session = match vision.as_mut() {
            Some(session) => session,
            None => {
                let paths = model_paths()?;
                artifacts::ensure_vision_artifacts(&paths, on_event, timing, self.verification)?;
                let started = timing.start();
                let session = VisionSession::load(&paths)?;
                timing.record(Phase::ModelSessionConstruction, started);
                vision.insert(session)
            }
        };
        run(session, timing)
    }

    /// One-shot searches release vision resources before text inference to bound peak memory.
    /// Persistent searches retain both sessions for later requests.
    pub(crate) fn with_text<T>(
        &self,
        on_event: &mut impl FnMut(ArtifactEvent),
        timing: &mut TimingRecorder,
        run: impl FnOnce(&mut TextSession, &mut TimingRecorder) -> Result<T, VisionGrepError>,
    ) -> Result<T, VisionGrepError> {
        if let SessionLifetime::Search = self.lifetime {
            let mut vision = self
                .vision
                .lock()
                .map_err(|_| VisionGrepError::ModelSessionPoisoned { model: "vision" })?;
            *vision = None;
        }
        let mut text = self
            .text
            .lock()
            .map_err(|_| VisionGrepError::ModelSessionPoisoned { model: "text" })?;
        let session = match text.as_mut() {
            Some(session) => session,
            None => {
                let paths = model_paths()?;
                artifacts::ensure_text_artifacts(&paths, on_event, timing, self.verification)?;
                text.insert(load_text_session(&paths, timing)?)
            }
        };
        run(session, timing)
    }
}

fn load_text_session(
    paths: &ModelPaths,
    timing: &mut TimingRecorder,
) -> Result<TextSession, VisionGrepError> {
    let started = timing.start();
    let session = TextSession::load(paths)?;
    timing.record(Phase::ModelSessionConstruction, started);
    Ok(session)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    #[test]
    fn preload_text_skips_missing_or_unverified_artifacts() {
        let dir = tempfile::tempdir().unwrap();
        let paths = ModelPaths::in_dir(dir.path());
        let models = Models::with_lifetime(ArtifactVerification::Fast, SessionLifetime::Process);

        models.preload_text_at(&paths);
        assert!(models.text.lock().unwrap().is_none());
        assert!(fs::read_dir(dir.path()).unwrap().next().is_none());

        fs::write(&paths.text_model, b"not a model").unwrap();
        fs::write(&paths.tokenizer, b"not a tokenizer").unwrap();
        models.preload_text_at(&paths);
        assert!(models.text.lock().unwrap().is_none());
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
        assert_eq!(fs::read(&paths.text_model).unwrap(), b"not a model");
        assert_eq!(fs::read(&paths.tokenizer).unwrap(), b"not a tokenizer");
    }
}
