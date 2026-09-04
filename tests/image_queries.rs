use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use image::{Rgb, RgbImage};
use rusqlite::Connection;
use serde_json::Value;

struct SearchOutput {
    results: Value,
    timing: Value,
}

impl SearchOutput {
    fn first_path(&self) -> PathBuf {
        PathBuf::from(self.results[0]["path"].as_str().unwrap())
    }

    fn invocations(&self, phase: &str) -> u64 {
        self.timing["phases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|entry| entry["phase"] == phase)
            .map_or(0, |entry| entry["invocations"].as_u64().unwrap())
    }
}

fn search_image(
    corpus: &Path,
    query: &Path,
    cache: &Path,
    scratch: &Path,
    flags: &[&str],
) -> SearchOutput {
    let timing_path = scratch.join("timing.json");
    let output = Command::new(env!("CARGO_BIN_EXE_visiongrep"))
        .arg("--image")
        .arg(query)
        .arg(corpus)
        .args([
            "--quiet",
            "--json",
            "--threshold",
            "-1",
            "--top",
            "10",
            "--timing",
            "--timing-file",
        ])
        .arg(&timing_path)
        .args(flags)
        .env("XDG_CACHE_HOME", cache)
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    SearchOutput {
        results: serde_json::from_slice(&output.stdout).unwrap(),
        timing: serde_json::from_slice(&fs::read(timing_path).unwrap()).unwrap(),
    }
}

#[derive(Debug, PartialEq, Eq)]
struct IndexedImage {
    path: Vec<u8>,
    embedding: Vec<u8>,
}

fn indexed_embeddings(index_path: &Path) -> Vec<IndexedImage> {
    let connection = Connection::open(index_path).unwrap();
    let mut statement = connection
        .prepare("SELECT path, embedding FROM images ORDER BY path")
        .unwrap();
    let images = statement
        .query_map([], |row| {
            Ok(IndexedImage {
                path: row.get(0)?,
                embedding: row.get(1)?,
            })
        })
        .unwrap();
    images.collect::<Result<_, _>>().unwrap()
}

/// Requires only the pinned vision model; text artifacts deliberately remain absent.
#[test]
#[ignore = "requires XDG_CACHE_HOME containing the pinned DataComp vision model"]
fn image_query_end_to_end() {
    let installed_cache = PathBuf::from(
        std::env::var_os("XDG_CACHE_HOME").expect("set XDG_CACHE_HOME to the pinned model cache"),
    );
    let source_model = installed_cache.join("visiongrep/models/datacomp_vision.onnx");
    assert!(source_model.is_file(), "missing {}", source_model.display());

    let scratch = tempfile::tempdir().unwrap();
    let corpus = scratch.path().join("photos");
    let external = scratch.path().join("external.png");
    let cache = scratch.path().join("cache");
    let models = cache.join("visiongrep/models");
    fs::create_dir(&corpus).unwrap();
    fs::create_dir_all(&models).unwrap();
    // A symlink avoids copying hundreds of megabytes while keeping test cache writes isolated.
    std::os::unix::fs::symlink(
        source_model.canonicalize().unwrap(),
        models.join("datacomp_vision.onnx"),
    )
    .unwrap();

    let query = corpus.join("query.png");
    let duplicate = corpus.join("duplicate.png");
    let other = corpus.join("other.png");
    RgbImage::from_fn(96, 64, |x, y| {
        if x < 48 && y < 32 {
            Rgb([255, 20, 20])
        } else {
            Rgb([20, 20, 255])
        }
    })
    .save(&query)
    .unwrap();
    RgbImage::from_pixel(80, 120, Rgb([30, 240, 20]))
        .save(&other)
        .unwrap();
    fs::copy(&query, &duplicate).unwrap();
    fs::copy(&query, &external).unwrap();
    let index = corpus.join(".visiongrep.db");

    let fresh = search_image(&corpus, &external, &cache, scratch.path(), &[]);
    assert_eq!(fresh.first_path(), duplicate);
    assert_eq!(fresh.invocations("model_session_construction"), 1);
    assert_eq!(fresh.invocations("vision_inference"), 2);
    let original_index = indexed_embeddings(&index);
    assert_eq!(original_index.len(), 3);

    let repeated = search_image(&corpus, &external, &cache, scratch.path(), &[]);
    assert_eq!(repeated.first_path(), duplicate);
    assert_eq!(repeated.invocations("model_session_construction"), 1);
    assert_eq!(repeated.invocations("vision_inference"), 1);
    assert_eq!(indexed_embeddings(&index), original_index);

    let indexed = search_image(&corpus, &query, &cache, scratch.path(), &[]);
    assert_eq!(indexed.first_path(), duplicate);
    assert_eq!(indexed.results.as_array().unwrap().len(), 2);
    assert_eq!(indexed.invocations("model_session_construction"), 0);
    assert_eq!(indexed.invocations("vision_inference"), 0);

    fs::copy(&other, &external).unwrap();
    let changed_external = search_image(&corpus, &external, &cache, scratch.path(), &[]);
    assert_eq!(changed_external.first_path(), other);
    assert_eq!(indexed_embeddings(&index), original_index);

    fs::copy(&other, &query).unwrap();
    let changed_indexed = search_image(&corpus, &query, &cache, scratch.path(), &[]);
    assert_eq!(changed_indexed.first_path(), other);
    assert_eq!(changed_indexed.invocations("model_session_construction"), 1);
    assert_eq!(changed_indexed.invocations("vision_inference"), 1);
    let changed_index = indexed_embeddings(&index);
    assert_ne!(changed_index, original_index);

    let no_cache = search_image(&corpus, &external, &cache, scratch.path(), &["--no-cache"]);
    assert_eq!(no_cache.first_path(), other);
    assert_eq!(no_cache.invocations("model_session_construction"), 1);
    assert_eq!(indexed_embeddings(&index), changed_index);

    let reindexed = search_image(&corpus, &external, &cache, scratch.path(), &["--reindex"]);
    assert_eq!(reindexed.first_path(), other);
    assert_eq!(reindexed.invocations("model_session_construction"), 1);
    assert_eq!(indexed_embeddings(&index).len(), 3);
    let connection = Connection::open(&index).unwrap();
    assert_eq!(
        connection
            .query_row("SELECT COUNT(*) FROM queries", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
    drop(connection);

    let preserved = indexed_embeddings(&index);
    fs::write(&external, b"invalid query").unwrap();
    let failed_reindex = Command::new(env!("CARGO_BIN_EXE_visiongrep"))
        .arg("--image")
        .arg(&external)
        .arg(&corpus)
        .args(["--quiet", "--json", "--reindex"])
        .env("XDG_CACHE_HOME", &cache)
        .output()
        .unwrap();
    assert_eq!(failed_reindex.status.code(), Some(2));
    assert!(failed_reindex.stdout.is_empty());
    assert!(String::from_utf8_lossy(&failed_reindex.stderr).contains("failed to decode image"));
    assert_eq!(indexed_embeddings(&index), preserved);
    assert!(!models.join("datacomp_text.onnx").exists());
    assert!(!models.join("datacomp_tokenizer.json").exists());
}
