mod artifacts;
mod graph_cache;
mod runtime;

use std::sync::Mutex;

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
    // Always acquire the index file lock before either model lock.
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
                let started = timing.start();
                let session = TextSession::load(&paths)?;
                timing.record(Phase::ModelSessionConstruction, started);
                text.insert(session)
            }
        };
        run(session, timing)
    }
}
