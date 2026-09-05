mod artifacts;
mod runtime;

pub(crate) use artifacts::{
    ArtifactEvent, ArtifactVerification, embedding_contract, model_paths, timing_metadata,
};
pub(crate) use runtime::{TextSession, VisionSession};

use crate::error::VisionGrepError;
use crate::timing::{Phase, TimingRecorder};

/// Owns the model resources needed by one search, loading artifacts only on demand.
pub(crate) struct Models {
    verification: ArtifactVerification,
    vision: Option<VisionSession>,
}

impl Models {
    pub(crate) fn new(verification: ArtifactVerification) -> Self {
        Self {
            verification,
            vision: None,
        }
    }

    /// Reuses the vision session across corpus ingestion and query-image inference.
    pub(crate) fn vision(
        &mut self,
        on_event: &mut impl FnMut(ArtifactEvent),
        timing: &mut TimingRecorder,
    ) -> Result<&mut VisionSession, VisionGrepError> {
        match self.vision {
            Some(ref mut session) => Ok(session),
            None => {
                let paths = model_paths()?;
                artifacts::ensure_vision_artifacts(&paths, on_event, timing, self.verification)?;
                let started = timing.start();
                let session = VisionSession::load(&paths)?;
                timing.record(Phase::ModelSessionConstruction, started);
                Ok(self.vision.insert(session))
            }
        }
    }

    /// Releases the vision session before loading the text model to keep peak memory bounded.
    /// Text inference follows corpus ingestion, and its session is needed for only one query.
    pub(crate) fn load_text(
        &mut self,
        on_event: &mut impl FnMut(ArtifactEvent),
        timing: &mut TimingRecorder,
    ) -> Result<TextSession, VisionGrepError> {
        self.vision = None;
        let paths = model_paths()?;
        artifacts::ensure_text_artifacts(&paths, on_event, timing, self.verification)?;
        let started = timing.start();
        let session = TextSession::load(&paths)?;
        timing.record(Phase::ModelSessionConstruction, started);
        Ok(session)
    }
}
