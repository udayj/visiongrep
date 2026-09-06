mod model_cache;

use std::collections::BTreeSet;
use std::fs;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use image::{Rgb, RgbImage};
use rusqlite::{Connection, OpenFlags};
use serde::Deserialize;

const QUERY: &str = "a red and blue picture";
const ARTIFACTS: [&str; 3] = [
    "datacomp_vision.onnx",
    "datacomp_text.onnx",
    "datacomp_tokenizer.json",
];

struct Fixture {
    scratch: tempfile::TempDir,
    corpus: PathBuf,
    cache: PathBuf,
}

impl Fixture {
    fn empty() -> Self {
        let scratch = tempfile::tempdir().unwrap();
        let corpus = scratch.path().join("photos with spaces");
        let cache = scratch.path().join("cache");
        fs::create_dir(&corpus).unwrap();
        Self {
            scratch,
            corpus,
            cache,
        }
    }

    fn with_images() -> Self {
        let fixture = Self::empty();
        write_image(&fixture.corpus.join("red.png"), Rgb([240, 20, 30]));
        fs::create_dir(fixture.corpus.join("nested")).unwrap();
        write_image(&fixture.corpus.join("nested/blue.png"), Rgb([20, 30, 240]));
        fs::write(fixture.corpus.join("notes.txt"), "not an image").unwrap();
        fixture
    }

    fn with_models() -> Self {
        let fixture = Self::with_images();
        model_cache::link_models(&fixture.cache, &ARTIFACTS);
        fixture
    }

    fn index(&self) -> PathBuf {
        self.corpus.join(".visiongrep.db")
    }

    fn command(&self) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_visiongrep"));
        command
            .current_dir(self.scratch.path())
            .env("XDG_CACHE_HOME", &self.cache)
            .env("NO_COLOR", "1");
        model_cache::prevent_downloads(&mut command);
        command
    }

    fn text_command(&self, query: &str) -> Command {
        let mut command = self.command();
        command.arg(query).arg(&self.corpus);
        command
    }

    fn search(&self, query: &str, flags: &[&str]) -> SearchOutput {
        self.run_search(self.text_command(query), flags)
    }

    fn run_search(&self, mut command: Command, flags: &[&str]) -> SearchOutput {
        let timing_path = self.scratch.path().join("timing.json");
        if !flags.contains(&"--top") {
            command.args(["--top", "10"]);
        }
        let output = command
            .args(["--json", "--threshold", "-1", "--timing", "--timing-file"])
            .arg(&timing_path)
            .args(flags)
            .output()
            .unwrap();
        assert_status(&output, 0);
        let results: Vec<JsonSearchResult> = serde_json::from_slice(&output.stdout).unwrap();
        assert!(!results.is_empty());
        assert!(results.iter().all(|result| result.score.is_finite()
            && (-1.0..=1.0).contains(&result.score)
            && result.path.is_file()));
        assert!(
            results
                .windows(2)
                .all(|pair| pair[0].score >= pair[1].score)
        );
        SearchOutput {
            results,
            timing: serde_json::from_slice(&fs::read(timing_path).unwrap()).unwrap(),
            stderr: String::from_utf8(output.stderr).unwrap(),
        }
    }
}

// An independent consumer of the CLI's JSON contract, not the internal ranking type.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct JsonSearchResult {
    path: PathBuf,
    score: f32,
}

#[derive(Deserialize)]
struct Timing {
    phases: Vec<Phase>,
}

#[derive(Deserialize)]
struct Phase {
    phase: String,
    invocations: u64,
}

struct SearchOutput {
    results: Vec<JsonSearchResult>,
    timing: Timing,
    stderr: String,
}

impl SearchOutput {
    fn invocations(&self, phase: &str) -> u64 {
        self.timing
            .phases
            .iter()
            .find(|entry| entry.phase == phase)
            .map_or(0, |entry| entry.invocations)
    }

    fn assert_inference(&self, images: u64, text: u64) {
        assert_eq!(self.invocations("image_decoding"), images);
        assert_eq!(self.invocations("vision_inference") > 0, images > 0);
        assert_eq!(self.invocations("text_inference"), text);
        assert_eq!(self.invocations("artifact_download"), 0);
    }

    fn assert_paths(&self, paths: &[PathBuf]) {
        let actual: BTreeSet<_> = self.results.iter().map(|result| &result.path).collect();
        assert_eq!(actual, paths.iter().collect());
        assert_eq!(self.results.len(), paths.len());
    }
}

#[derive(Debug, PartialEq, Eq)]
struct IndexedImage {
    path: Vec<u8>,
    embedding: Vec<u8>,
}

fn indexed_images(path: &Path) -> Vec<IndexedImage> {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let mut statement = connection
        .prepare("SELECT path, embedding FROM images ORDER BY path")
        .unwrap();
    statement
        .query_map([], |row| {
            Ok(IndexedImage {
                path: row.get(0)?,
                embedding: row.get(1)?,
            })
        })
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

fn cached_queries(path: &Path) -> Vec<String> {
    let connection = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY).unwrap();
    let mut statement = connection
        .prepare("SELECT query FROM queries ORDER BY query")
        .unwrap();
    statement
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap()
}

fn write_image(path: &Path, color: Rgb<u8>) {
    RgbImage::from_pixel(64, 48, color).save(path).unwrap();
}

fn assert_status(output: &Output, expected: i32) {
    assert_eq!(
        output.status.code(),
        Some(expected),
        "stdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "downloads the pinned models (about 610 MB); requires network access"]
fn text_query_without_index_or_models_downloads_and_caches_artifacts() {
    let fixture = Fixture::with_images();
    assert!(!fixture.index().exists());
    assert!(!fixture.cache.exists());
    // This is the only text E2E allowed to use the caller's network/proxy configuration.
    let mut command = Command::new(env!("CARGO_BIN_EXE_visiongrep"));
    command
        .arg(QUERY)
        .arg(&fixture.corpus)
        .env("XDG_CACHE_HOME", &fixture.cache);
    let fresh = fixture.run_search(command, &[]);
    assert_eq!(fresh.results.len(), 2);
    assert_eq!(fresh.invocations("artifact_download"), 3);
    assert_eq!(indexed_images(&fixture.index()).len(), 2);
    assert_eq!(cached_queries(&fixture.index()), [QUERY]);
    for name in ARTIFACTS {
        let path = fixture.cache.join("visiongrep/models").join(name);
        assert!(path.is_file());
        assert!(
            path.with_file_name(format!("{name}.verified.json"))
                .is_file()
        );
    }
    let repeated = fixture.search(QUERY, &[]);
    repeated.assert_inference(0, 0);
    assert_eq!(repeated.invocations("model_session_construction"), 0);
}

#[test]
fn text_query_without_index_builds_index_with_installed_models() {
    let fixture = Fixture::with_models();
    assert!(!fixture.index().exists());
    let fresh = fixture.search(QUERY, &[]);
    fresh.assert_paths(&[
        fixture.corpus.join("red.png"),
        fixture.corpus.join("nested/blue.png"),
    ]);
    fresh.assert_inference(2, 1);
    assert_eq!(indexed_images(&fixture.index()).len(), 2);
    assert_eq!(cached_queries(&fixture.index()), [QUERY]);
    assert!(fresh.stderr.contains("Indexing 2 images"));
}

#[test]
fn text_query_with_index_reuses_images_and_caches_each_query() {
    let fixture = Fixture::with_models();
    let fresh = fixture.search(QUERY, &[]);
    let original = indexed_images(&fixture.index());
    let repeated = fixture.search(QUERY, &[]);
    assert_eq!(repeated.results, fresh.results);
    repeated.assert_inference(0, 0);
    assert_eq!(repeated.invocations("model_session_construction"), 0);
    let novel = fixture.search("a blue picture", &[]);
    novel.assert_inference(0, 1);
    assert_eq!(novel.invocations("model_session_construction"), 1);
    assert_eq!(indexed_images(&fixture.index()), original);
    assert_eq!(cached_queries(&fixture.index()).len(), 2);

    fs::remove_dir_all(&fixture.cache).unwrap();
    let offline = fixture.search(QUERY, &[]);
    assert_eq!(offline.results, fresh.results);
    offline.assert_inference(0, 0);
    assert_eq!(offline.invocations("model_session_construction"), 0);
    assert_eq!(offline.results.len(), 2);
    assert!(!fixture.cache.exists());
}

#[test]
fn text_query_reconciles_deleted_modified_and_new_images() {
    let fixture = Fixture::with_models();
    let stable = fixture.corpus.join("stable.png");
    write_image(&stable, Rgb([90, 80, 70]));
    fixture.search(QUERY, &[]);
    let original = indexed_images(&fixture.index());
    fs::remove_file(fixture.corpus.join("red.png")).unwrap();
    // Changing dimensions guarantees different metadata without relying on clock resolution.
    RgbImage::from_pixel(120, 80, Rgb([30, 240, 20]))
        .save(fixture.corpus.join("nested/blue.png"))
        .unwrap();
    let new = fixture.corpus.join("new.png");
    write_image(&new, Rgb([250, 220, 10]));
    let changed = fixture.search(QUERY, &[]);
    changed.assert_paths(&[stable, new, fixture.corpus.join("nested/blue.png")]);
    changed.assert_inference(2, 0);
    let updated = indexed_images(&fixture.index());
    assert_eq!(updated.len(), 3);
    assert!(!updated.iter().any(|row| row.path == b"red.png"));
    assert_eq!(
        updated.iter().find(|row| row.path == b"stable.png"),
        original.iter().find(|row| row.path == b"stable.png")
    );
    assert_ne!(
        updated.iter().find(|row| row.path == b"nested/blue.png"),
        original.iter().find(|row| row.path == b"nested/blue.png")
    );
    fixture.search(QUERY, &[]).assert_inference(0, 0);
}

#[test]
fn text_query_reindex_rebuilds_images_and_query_cache() {
    let fixture = Fixture::with_models();
    fixture.search("old query", &[]);
    fixture.search(QUERY, &[]);
    let rebuilt = fixture.search(QUERY, &["--reindex"]);
    rebuilt.assert_inference(2, 1);
    assert_eq!(indexed_images(&fixture.index()).len(), 2);
    assert_eq!(cached_queries(&fixture.index()), [QUERY]);
    fixture.search(QUERY, &[]).assert_inference(0, 0);
}

#[test]
fn text_query_no_cache_neither_reads_nor_writes_index() {
    let fixture = Fixture::with_models();
    fixture
        .search(QUERY, &["--no-cache"])
        .assert_inference(2, 1);
    assert!(!fixture.index().exists());
    fixture.search(QUERY, &[]);
    let original = fs::read(fixture.index()).unwrap();
    let new = fixture.corpus.join("new.png");
    write_image(&new, Rgb([30, 240, 20]));
    let uncached = fixture.search("another query", &["--no-cache"]);
    uncached.assert_inference(3, 1);
    assert_eq!(uncached.results.len(), 3);
    assert_eq!(fs::read(fixture.index()).unwrap(), original);
    assert_eq!(cached_queries(&fixture.index()), [QUERY]);
    // A malformed index must also be completely ignored.
    fs::write(fixture.index(), b"not sqlite").unwrap();
    fixture
        .search(QUERY, &["--no-cache"])
        .assert_inference(3, 1);
    assert_eq!(fs::read(fixture.index()).unwrap(), b"not sqlite");
}

#[test]
fn text_query_with_external_index_supports_reuse_and_reindex() {
    let fixture = Fixture::with_models();
    // Relative custom index paths resolve from cwd, not the search root.
    let flags = ["--index-path", "external.db"];
    fixture.search(QUERY, &flags).assert_inference(2, 1);
    let index = fixture.scratch.path().join("external.db");
    assert_eq!(indexed_images(&index).len(), 2);
    assert!(!fixture.index().exists());
    fixture.search(QUERY, &flags).assert_inference(0, 0);
    fixture
        .search(QUERY, &["--index-path", "external.db", "--reindex"])
        .assert_inference(2, 1);
    assert!(!fixture.index().exists());
    assert_eq!(cached_queries(&index), [QUERY]);
}

#[test]
fn text_query_json_paths_text_and_null_outputs_agree() {
    let fixture = Fixture::with_models();
    fixture.search(QUERY, &[]);
    let json = fixture.search(QUERY, &["--quiet"]);
    json.assert_inference(0, 0);
    assert!(json.stderr.is_empty());
    for (flags, separator) in [(vec!["--paths-only"], b'\n'), (vec!["--null"], 0)] {
        let output = fixture
            .text_command(QUERY)
            .args(["--threshold", "-1"])
            .args(flags)
            .output()
            .unwrap();
        assert_status(&output, 0);
        assert!(output.stderr.is_empty());
        let mut expected = Vec::new();
        for result in &json.results {
            expected.extend_from_slice(result.path.as_os_str().as_bytes());
            expected.push(separator);
        }
        assert_eq!(output.stdout, expected);
    }
    let output = fixture
        .text_command(QUERY)
        .args(["--threshold", "-1"])
        .output()
        .unwrap();
    assert_status(&output, 0);
    let mut expected = String::from("score\tpath\n");
    for result in &json.results {
        expected.push_str(&format!("{:.3}\t{}\n", result.score, result.path.display()));
    }
    assert_eq!(String::from_utf8(output.stdout).unwrap(), expected);
    assert!(output.stderr.is_empty());

    let mut child = fixture
        .text_command(QUERY)
        .args(["--threshold", "-1", "--paths-only"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Close the reader before results are written, as with a downstream pipeline exiting early.
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    assert_status(&output, 0);
    assert!(output.stderr.is_empty());
}

#[test]
fn text_query_applies_top_and_threshold_and_returns_exit_one_for_no_matches() {
    let fixture = Fixture::with_models();
    let all = fixture.search(QUERY, &[]);
    let top = fixture.search(QUERY, &["--top", "1"]);
    assert_eq!(top.results.len(), 1);
    assert_eq!(top.results[0].path, all.results[0].path);
    assert!(all.results.iter().all(|result| result.score < 1.0));
    for flags in [vec!["--json"], vec!["--paths-only"], vec!["--null"]] {
        let json = flags == ["--json"];
        let output = fixture
            .text_command(QUERY)
            .args(["--threshold", "1"])
            .args(flags)
            .output()
            .unwrap();
        assert_status(&output, 1);
        assert_eq!(output.stdout, if json { b"[]\n".as_slice() } else { b"" });
        assert!(output.stderr.is_empty());
    }
}

#[test]
fn corrupt_corpus_images_are_skipped_with_diagnostics_on_stderr() {
    let fixture = Fixture::with_models();
    fs::write(fixture.corpus.join("corrupt.png"), b"not an image").unwrap();
    let search = fixture.search(QUERY, &[]);
    assert_eq!(search.results.len(), 2);
    assert!(search.stderr.contains("warning:"));
    assert!(search.stderr.contains("corrupt.png"));
    assert_eq!(indexed_images(&fixture.index()).len(), 2);
}

#[test]
fn a_newly_corrupt_indexed_image_loses_its_stale_embedding_and_can_recover() {
    let fixture = Fixture::with_models();
    fixture.search(QUERY, &[]);
    let original = indexed_images(&fixture.index());
    let corrupt = fixture.corpus.join("red.png");
    // Different bytes and size guarantee reconciliation without relying on mtime resolution.
    fs::write(&corrupt, b"no longer an image").unwrap();

    let changed = fixture.search(QUERY, &[]);

    changed.assert_paths(&[fixture.corpus.join("nested/blue.png")]);
    changed.assert_inference(0, 0);
    assert!(changed.stderr.contains("red.png"));
    assert!(changed.stderr.contains("warning:"));
    let remaining = indexed_images(&fixture.index());
    assert_eq!(remaining.len(), 1);
    assert_eq!(
        remaining[0],
        *original
            .iter()
            .find(|row| row.path == b"nested/blue.png")
            .unwrap()
    );

    write_image(&corrupt, Rgb([240, 20, 30]));
    let repaired = fixture.search(QUERY, &[]);
    repaired.assert_paths(&[corrupt, fixture.corpus.join("nested/blue.png")]);
    repaired.assert_inference(1, 0);
    assert_eq!(indexed_images(&fixture.index()).len(), 2);
    fixture.search(QUERY, &[]).assert_inference(0, 0);
}

#[test]
fn no_cache_skips_corrupt_images_without_changing_the_existing_index() {
    let fixture = Fixture::with_models();
    fixture.search(QUERY, &[]);
    let original = fs::read(fixture.index()).unwrap();
    fs::write(fixture.corpus.join("red.png"), b"not an image").unwrap();

    let search = fixture.search(QUERY, &["--no-cache"]);

    search.assert_paths(&[fixture.corpus.join("nested/blue.png")]);
    search.assert_inference(1, 1);
    assert!(search.stderr.contains("warning:"));
    assert!(search.stderr.contains("red.png"));
    assert_eq!(fs::read(fixture.index()).unwrap(), original);
}

#[test]
fn an_entirely_corrupt_corpus_returns_no_matches_without_inference() {
    for mode in [None, Some("--no-cache"), Some("--reindex")] {
        let fixture = Fixture::with_models();
        for path in ["red.png", "nested/blue.png"] {
            fs::write(fixture.corpus.join(path), b"not an image").unwrap();
        }
        let timing_path = fixture.scratch.path().join("timing.json");
        let output = fixture
            .text_command(QUERY)
            .args(["--json", "--timing", "--timing-file"])
            .arg(&timing_path)
            .args(mode)
            .output()
            .unwrap();

        assert_status(&output, 1);
        assert_eq!(output.stdout, b"[]\n");
        let stderr = String::from_utf8(output.stderr).unwrap();
        assert!(stderr.contains("warning:"));
        assert!(stderr.contains("red.png"));
        assert!(stderr.contains("blue.png"));
        let timing: Timing = serde_json::from_slice(&fs::read(timing_path).unwrap()).unwrap();
        assert!(timing.phases.iter().all(|phase| !matches!(
            phase.phase.as_str(),
            "vision_inference" | "text_inference" | "artifact_download"
        ) || phase.invocations == 0));
        if mode == Some("--no-cache") {
            assert!(!fixture.index().exists());
        } else {
            assert!(indexed_images(&fixture.index()).is_empty());
        }
    }
}

#[test]
fn nine_images_survive_full_and_partial_vision_batches_with_and_without_cache() {
    let fixture = Fixture::with_models();
    let mut paths = vec![
        fixture.corpus.join("red.png"),
        fixture.corpus.join("nested/blue.png"),
    ];
    for index in 0_u8..7 {
        let path = fixture.corpus.join(format!("image-{index}.png"));
        write_image(&path, Rgb([index * 30, 90, 200]));
        paths.push(path);
    }

    let indexed = fixture.search(QUERY, &[]);

    indexed.assert_paths(&paths);
    indexed.assert_inference(9, 1);
    assert_eq!(indexed.invocations("vision_inference"), 2);
    assert_eq!(indexed_images(&fixture.index()).len(), 9);
    let uncached = fixture.search(QUERY, &["--no-cache"]);
    uncached.assert_paths(&paths);
    uncached.assert_inference(9, 1);
    assert_eq!(uncached.invocations("vision_inference"), 2);
    for result in &indexed.results {
        let other = uncached
            .results
            .iter()
            .find(|other| other.path == result.path)
            .unwrap();
        assert!((result.score - other.score).abs() < 1e-5);
    }
    let repeated = fixture.search(QUERY, &[]);
    repeated.assert_inference(0, 0);
    assert_eq!(repeated.results, indexed.results);
}

#[test]
fn ingestion_persists_images_on_both_sides_of_the_database_batch_boundary() {
    let fixture = Fixture::empty();
    model_cache::link_models(&fixture.cache, &ARTIFACTS);
    let first = fixture.corpus.join("000.png");
    let last = fixture.corpus.join("256.png");
    write_image(&first, Rgb([240, 20, 30]));
    write_image(&last, Rgb([20, 30, 240]));
    // Exercise 257 discovered files while keeping real model inference inexpensive.
    for index in 1..256 {
        fs::write(
            fixture.corpus.join(format!("{index:03}.png")),
            b"not an image",
        )
        .unwrap();
    }

    let search = fixture.search(QUERY, &[]);

    search.assert_paths(&[first, last]);
    search.assert_inference(2, 1);
    assert_eq!(search.invocations("vision_inference"), 2);
    // Two image transactions plus the text query cache write.
    assert_eq!(search.invocations("database_writes"), 3);
    let persisted = indexed_images(&fixture.index());
    assert_eq!(persisted.len(), 2);
    assert_eq!(persisted[0].path, b"000.png");
    assert_eq!(persisted[1].path, b"256.png");
}

#[test]
fn empty_corpus_needs_no_models_and_returns_exit_one() {
    for mode in [None, Some("--reindex"), Some("--no-cache")] {
        let fixture = Fixture::empty();
        fs::write(fixture.corpus.join("notes.txt"), "not an image").unwrap();
        let output = fixture
            .text_command(QUERY)
            .arg("--json")
            .args(mode)
            .output()
            .unwrap();
        assert_status(&output, 1);
        assert_eq!(output.stdout, b"[]\n");
        assert!(output.stderr.is_empty());
        assert!(!fixture.cache.exists());
        assert_eq!(fixture.index().exists(), mode != Some("--no-cache"));
    }
}

#[test]
fn invalid_arguments_fail_before_creating_cache_or_index() {
    let fixture = Fixture::empty();
    for flags in [
        vec!["--top", "0"],
        vec!["--threshold", "NaN"],
        vec!["--threshold", "1.1"],
        vec!["--reindex", "--no-cache"],
        vec!["--json", "--paths-only"],
        vec!["--json", "--null"],
        vec!["--index-path", "outside.db", "--no-cache"],
        vec!["--image", "image.png"],
    ] {
        let output = fixture.text_command(QUERY).args(&flags).output().unwrap();
        assert_status(&output, 2);
        assert!(output.stdout.is_empty(), "{flags:?}");
        assert!(!output.stderr.is_empty(), "{flags:?}");
    }
    let blank = fixture.text_command("   ").output().unwrap();
    assert_status(&blank, 2);
    assert!(blank.stdout.is_empty());
    assert!(!fixture.index().exists());
    assert!(!fixture.cache.exists());
}

#[test]
fn missing_or_non_directory_search_roots_are_operational_errors() {
    let fixture = Fixture::empty();
    let file = fixture.scratch.path().join("file");
    fs::write(&file, "not a directory").unwrap();
    for path in [fixture.scratch.path().join("missing"), file] {
        let output = fixture
            .command()
            .arg(QUERY)
            .arg(path)
            .arg("--json")
            .output()
            .unwrap();
        assert_status(&output, 2);
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
    assert!(!fixture.cache.exists());
}

#[test]
fn invalid_image_queries_are_operational_errors_even_for_empty_corpus() {
    let fixture = Fixture::empty();
    let corrupt = fixture.scratch.path().join("corrupt.png");
    fs::write(&corrupt, b"invalid image").unwrap();
    for query in [
        fixture.scratch.path().join("missing.png"),
        corrupt,
        fixture.scratch.path().to_owned(),
    ] {
        for mode in [None, Some("--reindex"), Some("--no-cache")] {
            let output = fixture
                .command()
                .arg("--image")
                .arg(&query)
                .arg(&fixture.corpus)
                .arg("--json")
                .args(mode)
                .output()
                .unwrap();
            assert_status(&output, 2);
            assert!(output.stdout.is_empty());
            assert!(!output.stderr.is_empty());
        }
    }
    assert!(!fixture.cache.exists());
}

#[test]
fn failed_text_reindex_preserves_the_previous_index() {
    let fixture = Fixture::with_models();
    fixture.search(QUERY, &[]);
    let original = fs::read(fixture.index()).unwrap();
    fs::remove_file(fixture.cache.join("visiongrep/models/datacomp_text.onnx")).unwrap();
    let output = fixture
        .text_command(QUERY)
        .args(["--reindex", "--json", "--quiet"])
        .output()
        .unwrap();
    assert_status(&output, 2);
    assert!(output.stdout.is_empty());
    assert!(!output.stderr.is_empty());
    assert_eq!(fs::read(fixture.index()).unwrap(), original);
    fixture.search(QUERY, &[]).assert_inference(0, 0);
    assert!(fs::read_dir(&fixture.corpus).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .as_bytes()
            .starts_with(b".visiongrep.db.reindex-")
    }));
}

#[test]
fn external_index_rejects_a_different_search_root() {
    let fixture = Fixture::empty();
    let output = fixture
        .text_command(QUERY)
        .args(["--json", "--index-path", "external.db"])
        .output()
        .unwrap();
    assert_status(&output, 1);
    let index = fixture.scratch.path().join("external.db");
    let original = fs::read(&index).unwrap();
    let other = fixture.scratch.path().join("other");
    fs::create_dir(&other).unwrap();
    let output = fixture
        .command()
        .arg(QUERY)
        .arg(&other)
        .args(["--json", "--index-path", "external.db"])
        .output()
        .unwrap();
    assert_status(&output, 2);
    assert!(output.stdout.is_empty());
    assert!(!output.stderr.is_empty());
    assert_eq!(fs::read(index).unwrap(), original);
    assert!(!fixture.index().exists());
    assert!(!other.join(".visiongrep.db").exists());
    assert!(!fixture.cache.exists());
}

#[test]
fn unavailable_download_returns_exit_two_without_caching_partial_results() {
    let fixture = Fixture::with_images();
    let output = fixture
        .text_command(QUERY)
        .args(["--json", "--quiet"])
        .output()
        .unwrap();
    assert_status(&output, 2);
    assert!(output.stdout.is_empty());
    assert!(!output.stderr.is_empty());
    assert!(indexed_images(&fixture.index()).is_empty());
    assert!(cached_queries(&fixture.index()).is_empty());
    assert_eq!(
        fs::read_dir(fixture.cache.join("visiongrep/models"))
            .unwrap()
            .count(),
        0
    );
}

#[test]
fn timing_on_stderr_keeps_stdout_as_result_json() {
    let fixture = Fixture::empty();
    let output = fixture
        .text_command(QUERY)
        .args(["--json", "--quiet", "--timing"])
        .output()
        .unwrap();
    assert_status(&output, 1);
    assert_eq!(output.stdout, b"[]\n");
    let timing: Timing = serde_json::from_slice(&output.stderr).unwrap();
    assert!(
        timing
            .phases
            .iter()
            .any(|phase| { phase.phase == "output_serialization" && phase.invocations == 1 })
    );
    assert!(!fixture.cache.exists());
}

#[test]
fn invalid_timing_destination_returns_an_operational_error() {
    let fixture = Fixture::empty();
    let missing = fixture.scratch.path().join("missing");
    for destination in [fixture.corpus.clone(), missing.join("timing.json")] {
        let output = fixture
            .text_command(QUERY)
            .args(["--json", "--quiet", "--timing", "--timing-file"])
            .arg(&destination)
            .output()
            .unwrap();
        assert_status(&output, 2);
        // Results are written before the timing report is attempted.
        assert_eq!(output.stdout, b"[]\n");
        let diagnostic = String::from_utf8(output.stderr).unwrap();
        assert!(diagnostic.contains(destination.to_str().unwrap()));
        assert!(diagnostic.contains("timing"));
    }
    assert!(!missing.exists());
    assert!(fixture.corpus.is_dir());
    assert!(!fixture.cache.exists());
}

#[test]
fn text_query_accepts_full_model_verification_with_installed_artifacts() {
    let fixture = Fixture::with_models();
    let verified = fixture.search(QUERY, &["--verify-models"]);
    verified.assert_inference(2, 1);
    verified.assert_paths(&[
        fixture.corpus.join("red.png"),
        fixture.corpus.join("nested/blue.png"),
    ]);
    assert_eq!(cached_queries(&fixture.index()), [QUERY]);
}
