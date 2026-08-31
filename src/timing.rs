use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::error::VisionGrepError;

const TIMING_SCHEMA_VERSION: u32 = 1;

#[derive(Debug)]
pub(crate) enum TimingDestination {
    Stderr,
    File(PathBuf),
}

impl TimingDestination {
    pub(crate) fn new(path: Option<PathBuf>) -> Self {
        path.map_or(Self::Stderr, Self::File)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Phase {
    PathValidationCanonicalization,
    RecursiveDiscoveryMetadata,
    IndexOpenSchemaInitialization,
    StaleEntryReconciliation,
    ChangedMissingImageDetection,
    ArtifactValidation,
    ArtifactDownload,
    ModelSessionConstruction,
    ImageDecoding,
    ImagePreprocessing,
    VisionInference,
    DatabaseWrites,
    TextTokenization,
    TextInference,
    CachedVectorLoadingDeserialization,
    SimilarityScoring,
    TopKSelection,
    OutputSerialization,
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CacheState {
    NotApplicable,
    Absent,
    Hit,
    Miss,
    Changed,
    Reindexed,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ModelMetadata {
    pub(crate) contract: &'static str,
    pub(crate) vision_revision: &'static str,
    pub(crate) vision_sha256: &'static str,
    pub(crate) text_revision: &'static str,
    pub(crate) text_sha256: &'static str,
    pub(crate) tokenizer_sha256: &'static str,
}

#[derive(Debug, Default)]
struct PhaseMeasurement {
    elapsed: Duration,
    invocations: u64,
}

#[derive(Debug, Serialize)]
struct PhaseReport {
    phase: Phase,
    elapsed_ms: f64,
    invocations: u64,
}

#[derive(Debug, Serialize)]
struct EnvironmentMetadata {
    commit: &'static str,
    model: ModelMetadata,
    os: &'static str,
    architecture: &'static str,
    logical_cpu_count: Option<usize>,
    total_memory_bytes: Option<u64>,
    onnx_runtime_version: &'static str,
    execution_provider: &'static str,
    build_profile: &'static str,
    corpus_size: usize,
    query_cache_state: CacheState,
    index_cache_state: CacheState,
}

#[derive(Debug, Serialize)]
struct TimingReport {
    schema_version: u32,
    environment: EnvironmentMetadata,
    phases: Vec<PhaseReport>,
    total_wall_ms: f64,
}

pub(crate) struct TimingRecorder {
    enabled: bool,
    wall_started: Instant,
    total_wall: Option<Duration>,
    phases: BTreeMap<Phase, PhaseMeasurement>,
    environment: EnvironmentMetadata,
}

impl TimingRecorder {
    pub(crate) fn new(enabled: bool, model: ModelMetadata) -> Self {
        Self {
            enabled,
            wall_started: Instant::now(),
            total_wall: None,
            phases: BTreeMap::new(),
            environment: EnvironmentMetadata {
                commit: option_env!("VISIONGREP_BUILD_COMMIT")
                    .or(option_env!("GITHUB_SHA"))
                    .unwrap_or("unknown"),
                model,
                os: std::env::consts::OS,
                architecture: std::env::consts::ARCH,
                logical_cpu_count: std::thread::available_parallelism().ok().map(usize::from),
                total_memory_bytes: total_memory_bytes(),
                onnx_runtime_version: "1.24.2",
                execution_provider: "cpu",
                build_profile: if cfg!(debug_assertions) {
                    "debug"
                } else {
                    "release"
                },
                corpus_size: 0,
                query_cache_state: CacheState::NotApplicable,
                index_cache_state: CacheState::NotApplicable,
            },
        }
    }

    #[cfg(test)]
    pub(crate) fn disabled(model: ModelMetadata) -> Self {
        Self::new(false, model)
    }

    pub(crate) fn start(&self) -> Instant {
        Instant::now()
    }

    pub(crate) fn record(&mut self, phase: Phase, started: Instant) {
        if !self.enabled {
            return;
        }

        let measurement = self.phases.entry(phase).or_default();
        measurement.elapsed += started.elapsed();
        measurement.invocations += 1;
    }

    pub(crate) fn set_corpus_size(&mut self, corpus_size: usize) {
        self.environment.corpus_size = corpus_size;
    }

    pub(crate) fn set_query_cache_state(&mut self, state: CacheState) {
        self.environment.query_cache_state = state;
    }

    pub(crate) fn set_index_cache_state(&mut self, state: CacheState) {
        self.environment.index_cache_state = state;
    }

    pub(crate) fn finish(&mut self) {
        if self.enabled {
            self.total_wall = Some(self.wall_started.elapsed());
        }
    }

    pub(crate) fn write(&self, destination: &TimingDestination) -> Result<(), VisionGrepError> {
        if !self.enabled {
            return Ok(());
        }

        match destination {
            TimingDestination::Stderr => {
                let stderr = io::stderr();
                let mut output = BufWriter::new(stderr.lock());
                self.write_to(&mut output)
            }
            TimingDestination::File(path) => {
                let file = File::create(path).map_err(|source| VisionGrepError::TimingFile {
                    operation: "creating",
                    path: path.clone(),
                    source,
                })?;
                let mut output = BufWriter::new(file);
                self.write_to(&mut output)
            }
        }
    }

    fn write_to(&self, output: &mut impl Write) -> Result<(), VisionGrepError> {
        if !self.enabled {
            return Ok(());
        }

        let phases = self
            .phases
            .iter()
            .map(|(phase, measurement)| PhaseReport {
                phase: *phase,
                elapsed_ms: milliseconds(measurement.elapsed),
                invocations: measurement.invocations,
            })
            .collect();
        let report = TimingReport {
            schema_version: TIMING_SCHEMA_VERSION,
            environment: EnvironmentMetadata {
                commit: self.environment.commit,
                model: self.environment.model.clone(),
                os: self.environment.os,
                architecture: self.environment.architecture,
                logical_cpu_count: self.environment.logical_cpu_count,
                total_memory_bytes: self.environment.total_memory_bytes,
                onnx_runtime_version: self.environment.onnx_runtime_version,
                execution_provider: self.environment.execution_provider,
                build_profile: self.environment.build_profile,
                corpus_size: self.environment.corpus_size,
                query_cache_state: self.environment.query_cache_state,
                index_cache_state: self.environment.index_cache_state,
            },
            phases,
            total_wall_ms: milliseconds(
                self.total_wall
                    .unwrap_or_else(|| self.wall_started.elapsed()),
            ),
        };

        serde_json::to_writer(&mut *output, &report)
            .map_err(|source| VisionGrepError::TimingSerialize { source })?;
        writeln!(output).map_err(|source| VisionGrepError::TimingFile {
            operation: "writing",
            path: PathBuf::from("<stderr or timing output>"),
            source,
        })?;
        output
            .flush()
            .map_err(|source| VisionGrepError::TimingFile {
                operation: "flushing",
                path: PathBuf::from("<stderr or timing output>"),
                source,
            })?;
        Ok(())
    }
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn total_memory_bytes() -> Option<u64> {
    linux_total_memory_bytes().or_else(macos_total_memory_bytes)
}

fn linux_total_memory_bytes() -> Option<u64> {
    let memory = std::fs::read_to_string(Path::new("/proc/meminfo")).ok()?;
    let kilobytes = memory
        .lines()
        .find_map(|line| line.strip_prefix("MemTotal:"))?
        .split_whitespace()
        .next()?
        .parse::<u64>()
        .ok()?;
    kilobytes.checked_mul(1024)
}

fn macos_total_memory_bytes() -> Option<u64> {
    if std::env::consts::OS != "macos" {
        return None;
    }

    let output = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_metadata() -> ModelMetadata {
        ModelMetadata {
            contract: "test",
            vision_revision: "vision",
            vision_sha256: "a",
            text_revision: "text",
            text_sha256: "b",
            tokenizer_sha256: "c",
        }
    }

    #[test]
    fn disabled_timing_produces_no_output() {
        let timing = TimingRecorder::disabled(model_metadata());
        let mut output = Vec::new();

        timing.write_to(&mut output).unwrap();

        assert!(output.is_empty());
    }

    #[test]
    fn report_is_machine_readable_and_contains_cache_state() {
        let mut timing = TimingRecorder::new(true, model_metadata());
        let started = timing.start();
        timing.record(Phase::SimilarityScoring, started);
        timing.set_corpus_size(42);
        timing.set_query_cache_state(CacheState::Hit);
        timing.finish();
        let mut output = Vec::new();

        timing.write_to(&mut output).unwrap();
        let report: serde_json::Value = serde_json::from_slice(&output).unwrap();

        assert_eq!(report["schema_version"], 1);
        assert_eq!(report["environment"]["corpus_size"], 42);
        assert_eq!(report["environment"]["query_cache_state"], "hit");
        assert_eq!(report["phases"][0]["phase"], "similarity_scoring");
    }
}
