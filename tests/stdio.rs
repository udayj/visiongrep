mod model_cache;

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use image::{Rgb, RgbImage};
use serde_json::{Value, json};

#[test]
fn live_requests_refresh_images_reuse_models_and_recover_from_errors() {
    let scratch = tempfile::tempdir().unwrap();
    let corpus = scratch.path().join("photos");
    let cache = scratch.path().join("cache");
    let index = scratch.path().join("photos.db");
    fs::create_dir(&corpus).unwrap();
    let red = corpus.join("red.png");
    let blue = corpus.join("blue.png");
    RgbImage::from_pixel(16, 16, Rgb([240, 20, 30]))
        .save(&red)
        .unwrap();
    RgbImage::from_pixel(16, 16, Rgb([20, 30, 240]))
        .save(&blue)
        .unwrap();
    let artifacts = [
        "datacomp_vision.onnx",
        "datacomp_text.onnx",
        "datacomp_tokenizer.json",
    ];
    model_cache::link_models(&cache, &artifacts);
    let mut command = Command::new(env!("CARGO_BIN_EXE_visiongrep"));
    command
        .args(["serve", "--stdio", "--index-path"])
        .arg(&index)
        .arg(&corpus)
        .env("XDG_CACHE_HOME", &cache)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    model_cache::prevent_downloads(&mut command);
    let mut child = command.spawn().unwrap();
    let mut input = child.stdin.take().unwrap();
    let mut output = BufReader::new(child.stdout.take().unwrap());
    let mut exchange = |request: Value| {
        serde_json::to_writer(&mut input, &request).unwrap();
        writeln!(input).unwrap();
        input.flush().unwrap();
        let mut line = String::new();
        assert!(output.read_line(&mut line).unwrap() > 0);
        serde_json::from_str::<Value>(&line).unwrap()
    };
    let first = exchange(json!({"id":1, "query":"a red picture", "threshold":-1}));
    assert_eq!(first["id"], 1);
    assert_eq!(first["results"].as_array().unwrap().len(), 2);
    assert!(index.is_file());
    assert!(!corpus.join(".visiongrep.db").exists());

    // Subsequent novel text and changed images must work without reopening model artifacts.
    for name in artifacts {
        fs::remove_file(cache.join("visiongrep/models").join(name)).unwrap();
    }
    let green = corpus.join("green.png");
    fs::remove_file(&blue).unwrap();
    RgbImage::from_pixel(24, 24, Rgb([20, 240, 30]))
        .save(&red)
        .unwrap();
    RgbImage::from_pixel(16, 16, Rgb([20, 240, 30]))
        .save(&green)
        .unwrap();
    let second = exchange(json!({"id":2, "query":"a green picture", "threshold":-1}));
    let results = second["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert!(
        results
            .iter()
            .any(|result| result["path"] == red.to_str().unwrap())
    );
    assert!(
        results
            .iter()
            .any(|result| result["path"] == green.to_str().unwrap())
    );
    let connection = rusqlite::Connection::open(&index).unwrap();
    let stored_size: i64 = connection
        .query_row(
            "SELECT size FROM images WHERE path = ?1",
            [b"red.png".as_slice()],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(
        stored_size,
        i64::try_from(fs::metadata(&red).unwrap().len()).unwrap()
    );

    let failure = exchange(json!({"id":3, "image":scratch.path().join("missing.png")}));
    assert_eq!(failure["id"], 3);
    assert!(failure["error"].is_string());
    let image = exchange(json!({"id":4, "image":red, "threshold":-1}));
    assert_eq!(image["results"].as_array().unwrap().len(), 1);
    assert_eq!(image["results"][0]["path"], green.to_str().unwrap());
    let empty = exchange(json!({"id":5, "query":"a green picture", "threshold":1}));
    assert_eq!(empty, json!({"id":5, "results":[]}));
    drop(input);
    assert!(child.wait().unwrap().success());
}

#[test]
fn startup_failure_is_reported_on_stderr() {
    let scratch = tempfile::tempdir().unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_visiongrep"))
        .args(["serve", "--stdio"])
        .arg(scratch.path().join("missing"))
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(!output.stderr.is_empty());
}
