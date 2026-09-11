mod artifacts;
mod runtime;

pub(crate) use artifacts::{
    ArtifactEvent, ArtifactVerification, embedding_contract, model_paths, timing_metadata,
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
    vision: Option<VisionSession>,
    text: Option<TextSession>,
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
            vision: None,
            text: None,
            lifetime,
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

    /// One-shot searches release vision resources before text inference to bound peak memory.
    /// Persistent searches retain both sessions for later requests.
    pub(crate) fn load_text(
        &mut self,
        on_event: &mut impl FnMut(ArtifactEvent),
        timing: &mut TimingRecorder,
    ) -> Result<&mut TextSession, VisionGrepError> {
        if let SessionLifetime::Search = self.lifetime {
            self.vision = None;
        }
        if let Some(ref mut session) = self.text {
            return Ok(session);
        }
        let paths = model_paths()?;
        artifacts::ensure_text_artifacts(&paths, on_event, timing, self.verification)?;
        let started = timing.start();
        let session = TextSession::load(&paths)?;
        timing.record(Phase::ModelSessionConstruction, started);
        Ok(self.text.insert(session))
    }
}
