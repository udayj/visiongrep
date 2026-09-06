use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Shares installed model bytes while keeping each test's cache writes isolated.
pub(super) fn link_models(cache: &Path, artifacts: &[&str]) {
    let installed_cache = match std::env::var_os("XDG_CACHE_HOME").filter(|path| !path.is_empty()) {
        Some(path) => PathBuf::from(path),
        None => PathBuf::from(std::env::var_os("HOME").expect("HOME must be set")).join(".cache"),
    };
    let models = cache.join("visiongrep/models");
    fs::create_dir_all(&models).unwrap();
    for name in artifacts {
        let source = installed_cache.join("visiongrep/models").join(name);
        assert!(
            source.is_file(),
            "missing {}; run cargo build --release and sh scripts/prepare-test-models.sh first",
            source.display(),
        );
        std::os::unix::fs::symlink(source.canonicalize().unwrap(), models.join(name)).unwrap();
    }
}

/// Missing or invalid fixture artifacts must fail instead of triggering network downloads.
pub(super) fn prevent_downloads(command: &mut Command) {
    command
        .env("HTTPS_PROXY", "http://127.0.0.1:9")
        .env("HTTP_PROXY", "http://127.0.0.1:9")
        .env("ALL_PROXY", "http://127.0.0.1:9")
        .env("NO_PROXY", "")
        .env_remove("https_proxy")
        .env_remove("http_proxy")
        .env_remove("all_proxy")
        .env_remove("no_proxy");
}
