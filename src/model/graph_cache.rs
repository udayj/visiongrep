use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};

use ort::execution_providers::CPUExecutionProvider;
use ort::session::{Session, builder::GraphOptimizationLevel};
use sha2::{Digest, Sha256};
use tempfile::Builder;

use crate::error::VisionGrepError;

const CACHE_FORMAT_VERSION: u32 = 1;
const OPTIMIZATION_CONFIGURATION: &str = "level=3;format=onnx";
const CPU_PROVIDER_CONFIGURATION: &str = "CPUExecutionProvider;arena=true";

/// Loads an optimized graph when one is compatible and builds it opportunistically otherwise.
///
/// Graph caching is deliberately fail-open. Filesystem failures, a corrupt graph, and another
/// process building the same graph all fall back to the verified source model.
pub(super) fn load_session(
    source_model: &Path,
    source_sha256: &str,
) -> Result<Session, VisionGrepError> {
    let Some(cache_path) = cache_path(source_model, source_sha256) else {
        return load_source_model(source_model);
    };
    load_session_with_cache(source_model, &cache_path)
}

fn load_session_with_cache(
    source_model: &Path,
    cache_path: &Path,
) -> Result<Session, VisionGrepError> {
    if let Some(session) = load_cached_model(cache_path) {
        return Ok(session);
    }

    let Some(parent) = cache_path.parent() else {
        return load_source_model(source_model);
    };
    if fs::create_dir_all(parent).is_err() {
        return load_source_model(source_model);
    }

    let lock_path = cache_path.with_extension("lock");
    let Some(_lock) = CacheBuildLock::acquire(&lock_path) else {
        return load_source_model(source_model);
    };

    // A different process may have completed the cache between our first read and lock creation.
    if let Some(session) = load_cached_model(cache_path) {
        return Ok(session);
    }
    if cache_path.exists() && fs::remove_file(cache_path).is_err() {
        return load_source_model(source_model);
    }

    build_cached_model(source_model, cache_path).map_or_else(|| load_source_model(source_model), Ok)
}

fn load_source_model(path: &Path) -> Result<Session, VisionGrepError> {
    let mut builder = configured_builder(GraphOptimizationLevel::Level3)?;
    builder
        .commit_from_file(path)
        .map_err(VisionGrepError::Inference)
}

fn load_cached_model(path: &Path) -> Option<Session> {
    if !path.is_file() {
        return None;
    }
    let mut builder = configured_builder(GraphOptimizationLevel::Disable).ok()?;
    builder.commit_from_file(path).ok()
}

fn build_cached_model(source_model: &Path, cache_path: &Path) -> Option<Session> {
    let parent = cache_path.parent()?;
    // ONNX Runtime chooses the serialization format from this extension.
    let temporary = Builder::new()
        .prefix(".building-")
        .suffix(".onnx")
        .tempfile_in(parent)
        .ok()?;
    let temporary = temporary.into_temp_path();
    let builder = configured_builder(GraphOptimizationLevel::Level3).ok()?;
    let mut builder = builder.with_optimized_model_path(&temporary).ok()?;
    let session = builder.commit_from_file(source_model).ok()?;

    if let Ok(output) = File::open(&temporary)
        && output.metadata().is_ok_and(|metadata| metadata.len() > 0)
        && output.sync_all().is_ok()
    {
        let _ = temporary.persist_noclobber(cache_path);
    }
    Some(session)
}

fn configured_builder(
    optimization_level: GraphOptimizationLevel,
) -> Result<ort::session::builder::SessionBuilder, VisionGrepError> {
    Session::builder()?
        .with_no_environment_execution_providers()
        .map_err(session_builder_error)?
        .with_execution_providers([CPUExecutionProvider::default()
            .with_arena_allocator(true)
            .build()])
        .map_err(session_builder_error)?
        .with_optimization_level(optimization_level)
        .map_err(session_builder_error)
}

fn session_builder_error(
    source: ort::Error<ort::session::builder::SessionBuilder>,
) -> VisionGrepError {
    VisionGrepError::SessionBuilder {
        source: Box::new(source),
    }
}

fn cache_path(source_model: &Path, source_sha256: &str) -> Option<PathBuf> {
    let hardware = platform_compatibility()?;
    let parent = source_model.parent()?;
    let key = cache_key(CacheIdentity {
        format_version: CACHE_FORMAT_VERSION,
        source_sha256,
        runtime_build: ort::info(),
        optimization: OPTIMIZATION_CONFIGURATION,
        provider: CPU_PROVIDER_CONFIGURATION,
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        hardware: &hardware,
    });
    Some(parent.join("optimized").join(format!("{key}.onnx")))
}

struct CacheIdentity<'a> {
    format_version: u32,
    source_sha256: &'a str,
    runtime_build: &'a str,
    optimization: &'a str,
    provider: &'a str,
    os: &'a str,
    arch: &'a str,
    hardware: &'a str,
}

fn cache_key(identity: CacheIdentity<'_>) -> String {
    let mut hasher = Sha256::new();
    for component in [
        &identity.format_version.to_string(),
        identity.source_sha256,
        identity.runtime_build,
        identity.optimization,
        identity.provider,
        identity.os,
        identity.arch,
        identity.hardware,
    ] {
        hasher.update(component.as_bytes());
        hasher.update([0]);
    }
    format!("{:x}", hasher.finalize())
}

#[cfg(all(
    target_os = "linux",
    any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
))]
fn platform_compatibility() -> Option<String> {
    let cpu_info = fs::read_to_string("/proc/cpuinfo").ok()?;
    let first_processor = cpu_info.split("\n\n").next()?;
    let identity = first_processor
        .lines()
        .filter_map(|line| line.split_once(':'))
        .map(|(key, value)| (key.trim(), value.trim()))
        .filter(|(key, _)| {
            matches!(
                *key,
                "model name"
                    | "cpu family"
                    | "model"
                    | "stepping"
                    | "CPU implementer"
                    | "CPU architecture"
                    | "CPU variant"
                    | "CPU part"
                    | "CPU revision"
                    | "flags"
                    | "Features"
            )
        })
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join(";");
    (!identity.is_empty()).then(|| {
        format!(
            "{identity};rust_detected_features={}",
            detected_cpu_features()
        )
    })
}

#[cfg(all(
    target_os = "macos",
    any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
))]
fn platform_compatibility() -> Option<String> {
    let output = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "hw.model", "machdep.cpu.brand_string"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let identity = std::str::from_utf8(&output.stdout).ok()?.trim();
    (!identity.is_empty()).then(|| format!("{identity};features={}", detected_cpu_features()))
}

#[cfg(not(any(
    all(
        target_os = "linux",
        any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
    ),
    all(
        target_os = "macos",
        any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")
    )
)))]
fn platform_compatibility() -> Option<String> {
    None
}

#[cfg(all(
    any(target_os = "linux", target_os = "macos"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
fn detected_cpu_features() -> String {
    [
        ("sse2", std::is_x86_feature_detected!("sse2")),
        ("sse3", std::is_x86_feature_detected!("sse3")),
        ("ssse3", std::is_x86_feature_detected!("ssse3")),
        ("sse4.1", std::is_x86_feature_detected!("sse4.1")),
        ("sse4.2", std::is_x86_feature_detected!("sse4.2")),
        ("avx", std::is_x86_feature_detected!("avx")),
        ("avx2", std::is_x86_feature_detected!("avx2")),
        ("fma", std::is_x86_feature_detected!("fma")),
        ("avx512f", std::is_x86_feature_detected!("avx512f")),
    ]
    .into_iter()
    .filter_map(|(name, present)| present.then_some(name))
    .collect::<Vec<_>>()
    .join(",")
}

#[cfg(all(any(target_os = "linux", target_os = "macos"), target_arch = "aarch64"))]
fn detected_cpu_features() -> String {
    [
        ("neon", std::arch::is_aarch64_feature_detected!("neon")),
        ("fp", std::arch::is_aarch64_feature_detected!("fp")),
        ("fp16", std::arch::is_aarch64_feature_detected!("fp16")),
        (
            "dotprod",
            std::arch::is_aarch64_feature_detected!("dotprod"),
        ),
        ("sve", std::arch::is_aarch64_feature_detected!("sve")),
        ("sve2", std::arch::is_aarch64_feature_detected!("sve2")),
    ]
    .into_iter()
    .filter_map(|(name, present)| present.then_some(name))
    .collect::<Vec<_>>()
    .join(",")
}

struct CacheBuildLock {
    _file: File,
}

impl CacheBuildLock {
    fn acquire(path: &Path) -> Option<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .ok()?;
        file.try_lock().ok()?;
        Some(Self { _file: file })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use ndarray::{Array2, Array4};
    use ort::value::TensorRef;

    use super::*;
    use crate::embedding::IMAGE_SIZE_USIZE;

    fn identity<'a>(source: &'a str, runtime: &'a str, hardware: &'a str) -> CacheIdentity<'a> {
        CacheIdentity {
            format_version: CACHE_FORMAT_VERSION,
            source_sha256: source,
            runtime_build: runtime,
            optimization: OPTIMIZATION_CONFIGURATION,
            provider: CPU_PROVIDER_CONFIGURATION,
            os: "test-os",
            arch: "test-arch",
            hardware,
        }
    }

    #[test]
    fn cache_key_changes_with_model_runtime_and_hardware() {
        let baseline = cache_key(identity("model-a", "runtime-a", "cpu-a"));
        assert_ne!(
            baseline,
            cache_key(identity("model-b", "runtime-a", "cpu-a"))
        );
        assert_ne!(
            baseline,
            cache_key(identity("model-a", "runtime-b", "cpu-a"))
        );
        assert_ne!(
            baseline,
            cache_key(identity("model-a", "runtime-a", "cpu-b"))
        );

        let mut changed_schema = identity("model-a", "runtime-a", "cpu-a");
        changed_schema.format_version += 1;
        assert_ne!(baseline, cache_key(changed_schema));

        let mut changed_optimization = identity("model-a", "runtime-a", "cpu-a");
        changed_optimization.optimization = "level=2;format=onnx";
        assert_ne!(baseline, cache_key(changed_optimization));

        let mut changed_provider = identity("model-a", "runtime-a", "cpu-a");
        changed_provider.provider = "CPUExecutionProvider;arena=false";
        assert_ne!(baseline, cache_key(changed_provider));
    }

    #[test]
    fn production_key_uses_actual_runtime_and_host_identity() {
        let Some(hardware) = platform_compatibility() else {
            eprintln!("skipping host-identity assertion: platform identity is unavailable");
            return;
        };
        let expected = cache_key(CacheIdentity {
            format_version: CACHE_FORMAT_VERSION,
            source_sha256: "digest",
            runtime_build: ort::info(),
            optimization: OPTIMIZATION_CONFIGURATION,
            provider: CPU_PROVIDER_CONFIGURATION,
            os: std::env::consts::OS,
            arch: std::env::consts::ARCH,
            hardware: &hardware,
        });
        let path = cache_path(Path::new("models/source.onnx"), "digest").unwrap();

        assert_eq!(
            path,
            Path::new("models/optimized").join(format!("{expected}.onnx"))
        );
    }

    #[test]
    fn cache_build_lock_allows_only_one_builder() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("model.lock");
        let first = CacheBuildLock::acquire(&path).unwrap();

        assert!(CacheBuildLock::acquire(&path).is_none());
        drop(first);
        assert!(CacheBuildLock::acquire(&path).is_some());
    }

    #[test]
    fn unsupported_or_unidentifiable_platforms_skip_cache() {
        if platform_compatibility().is_none() {
            let model = Path::new("model.onnx");
            assert!(cache_path(model, "digest").is_none());
        }
    }

    #[cfg(unix)]
    #[test]
    fn cache_path_preserves_non_utf8_model_directories() {
        use std::os::unix::ffi::OsStringExt;

        let Some(_) = platform_compatibility() else {
            eprintln!("skipping native-path assertion: platform identity is unavailable");
            return;
        };
        let directory = std::ffi::OsString::from_vec(b"models-\xff".to_vec());
        let model = PathBuf::from(&directory).join("source.onnx");
        let path = cache_path(&model, "digest").unwrap();

        assert!(path.starts_with(&directory));
    }

    fn text_output(session: &mut Session) -> Vec<f32> {
        let input = Array2::<i64>::zeros((1, 77));
        let input = TensorRef::from_array_view(&input).unwrap();
        let outputs = session.run(ort::inputs![input]).unwrap();
        outputs
            .values()
            .next()
            .unwrap()
            .try_extract_array::<f32>()
            .unwrap()
            .iter()
            .copied()
            .collect()
    }

    fn vision_output(session: &mut Session) -> Vec<f32> {
        let input = Array4::<f32>::zeros((1, 3, IMAGE_SIZE_USIZE, IMAGE_SIZE_USIZE));
        let input = TensorRef::from_array_view(&input).unwrap();
        let outputs = session.run(ort::inputs![input]).unwrap();
        outputs
            .values()
            .next()
            .unwrap()
            .try_extract_array::<f32>()
            .unwrap()
            .iter()
            .copied()
            .collect()
    }

    #[test]
    #[ignore = "requires the pinned CLIP model artifacts"]
    fn optimized_graphs_preserve_outputs_and_recover_from_cache_failures() {
        platform_compatibility()
            .expect("supported test hosts must expose a conservative CPU identity");
        let paths = crate::model::model_paths().unwrap();
        let directory = tempfile::tempdir().unwrap();

        let mut baseline_text = load_source_model(&paths.text_model).unwrap();
        let expected_text = text_output(&mut baseline_text);
        let text_cache = directory.path().join("text.onnx");
        let mut created_text = load_session_with_cache(&paths.text_model, &text_cache).unwrap();
        assert!(text_cache.metadata().unwrap().len() > 0);
        assert_eq!(text_output(&mut created_text), expected_text);

        let mut cached_text = load_session_with_cache(&paths.text_model, &text_cache).unwrap();
        assert_eq!(text_output(&mut cached_text), expected_text);

        fs::write(&text_cache, b"truncated").unwrap();
        let mut recovered_text = load_session_with_cache(&paths.text_model, &text_cache).unwrap();
        assert!(text_cache.metadata().unwrap().len() > b"truncated".len() as u64);
        assert_eq!(text_output(&mut recovered_text), expected_text);

        let invalid_parent = directory.path().join("not-a-directory");
        fs::write(&invalid_parent, b"file").unwrap();
        let mut uncached_text =
            load_session_with_cache(&paths.text_model, &invalid_parent.join("text.onnx")).unwrap();
        assert_eq!(text_output(&mut uncached_text), expected_text);

        fs::remove_file(&text_cache).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        std::thread::scope(|scope| {
            let mut workers = Vec::new();
            for _ in 0..2 {
                let barrier = Arc::clone(&barrier);
                let source = &paths.text_model;
                let cache = &text_cache;
                let expected = &expected_text;
                workers.push(scope.spawn(move || {
                    barrier.wait();
                    let mut session = load_session_with_cache(source, cache).unwrap();
                    assert_eq!(text_output(&mut session), *expected);
                }));
            }
            for worker in workers {
                worker.join().unwrap();
            }
        });
        let mut cached_after_concurrency = load_cached_model(&text_cache).unwrap();
        assert_eq!(text_output(&mut cached_after_concurrency), expected_text);

        let mut baseline_vision = load_source_model(&paths.vision_model).unwrap();
        let expected_vision = vision_output(&mut baseline_vision);
        let vision_cache = directory.path().join("vision.onnx");
        let mut created_vision =
            load_session_with_cache(&paths.vision_model, &vision_cache).unwrap();
        assert!(vision_cache.metadata().unwrap().len() > 0);
        assert_eq!(vision_output(&mut created_vision), expected_vision);

        let mut cached_vision =
            load_session_with_cache(&paths.vision_model, &vision_cache).unwrap();
        assert_eq!(vision_output(&mut cached_vision), expected_vision);
    }
}
