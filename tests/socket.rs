mod model_cache;

use std::ffi::OsString;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::fs::MetadataExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};

use image::{Rgb, RgbImage};
use rusqlite::{Connection, params};

const QUERY: &str = "cached query";

struct Fixture {
    scratch: tempfile::TempDir,
    corpus: PathBuf,
    cache: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        // Unix-domain socket paths have a small platform-specific length limit.
        let scratch = tempfile::Builder::new()
            .prefix("vg-")
            .tempdir_in("/tmp")
            .unwrap();
        let corpus = scratch.path().join("photos");
        let cache = scratch.path().join("cache");
        fs::create_dir(&corpus).unwrap();
        Self {
            scratch,
            corpus,
            cache,
        }
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

    fn search_command(&self, flags: &[&str]) -> Command {
        let mut command = self.command();
        command.arg(QUERY).arg("./photos").args(flags);
        command
    }

    fn search(&self, flags: &[&str]) -> Output {
        self.search_command(flags).output().unwrap()
    }

    fn seed_cached_image(&self, name: &std::ffi::OsStr) -> bool {
        self.seed_cached_image_in(&self.corpus, name)
    }

    fn seed_cached_image_in(&self, corpus: &Path, name: &std::ffi::OsStr) -> bool {
        // Let the application create the schema and its model contract metadata.
        let output = self
            .command()
            .arg(QUERY)
            .arg(corpus)
            .arg("--no-server")
            .output()
            .unwrap();
        assert_status(&output, 1);
        let image = corpus.join(name);
        if let Err(error) = fs::write(&image, b"cached image contents") {
            // APFS rejects arbitrary filename bytes; Linux filesystems generally permit them.
            if cfg!(target_os = "macos") && error.raw_os_error() == Some(92) {
                return false;
            }
            panic!("failed to create cached image: {error}");
        }
        let metadata = fs::metadata(&image).unwrap();
        let mtime = i64::try_from(
            metadata
                .modified()
                .unwrap()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        )
        .unwrap();
        let mut image_embedding = vec![0.0_f32; 512];
        image_embedding[0] = 0.5;
        image_embedding[1] = 0.75_f32.sqrt();
        let mut query_embedding = vec![0.0_f32; 512];
        query_embedding[0] = 1.0;
        let bytes = |values: &[f32]| {
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        };
        let connection = Connection::open(corpus.join(".visiongrep.db")).unwrap();
        connection
            .execute(
                "INSERT INTO images (path, mtime_ns, size, embedding) VALUES (?1, ?2, ?3, ?4)",
                params![
                    name.as_bytes(),
                    mtime,
                    i64::try_from(metadata.len()).unwrap(),
                    bytes(&image_embedding)
                ],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO queries (query, embedding) VALUES (?1, ?2)",
                params![QUERY, bytes(&query_embedding)],
            )
            .unwrap();
        true
    }

    fn start_server(&self) -> Server {
        let child = self
            .command()
            .current_dir("/")
            .args(["serve", "--socket"])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut server = Server { child };
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(path) = self.socket_path()
                && UnixStream::connect(&path).is_ok()
            {
                return server;
            }
            assert!(
                server.child.try_wait().unwrap().is_none(),
                "server exited during startup"
            );
            assert!(Instant::now() < deadline, "server did not start");
            thread::sleep(Duration::from_millis(10));
        }
    }

    fn with_models() -> Self {
        let fixture = Self::new();
        model_cache::link_models(
            &fixture.cache,
            &[
                "datacomp_vision_int8.onnx",
                "datacomp_text.onnx",
                "datacomp_tokenizer.json",
            ],
        );
        fixture
    }

    fn stop(&self) -> Output {
        self.command().args(["server", "stop"]).output().unwrap()
    }

    fn socket_path(&self) -> Option<PathBuf> {
        let run = self.cache.join("visiongrep/run");
        fs::read_dir(run)
            .ok()?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.extension().is_some_and(|ext| ext == "sock"))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if self.socket_path().is_some() {
            let _ = self.stop();
        }
    }
}

fn assert_status(output: &Output, code: i32) {
    assert_eq!(
        output.status.code(),
        Some(code),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

struct Server {
    child: Child,
}

impl Server {
    fn wait_for_exit(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success());
                return;
            }
            assert!(Instant::now() < deadline, "server did not exit");
            thread::sleep(Duration::from_millis(10));
        }
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn cached_results_match_with_and_without_server_and_preserve_path_bytes() {
    let fixture = Fixture::new();
    let name = OsString::from_vec(b"photo-\xff.png".to_vec());
    if !fixture.seed_cached_image(&name) {
        return;
    }
    let found = fixture.search(&["--no-server", "--paths-only", "--threshold", "-1"]);
    let empty = fixture.search(&["--no-server", "--paths-only", "--threshold", "1"]);
    assert_eq!(found.status.code(), Some(0));
    assert_eq!(found.stdout, b"./photos/photo-\xff.png\n");
    assert_eq!(empty.status.code(), Some(1));
    let _server = fixture.start_server();
    let served = fixture.search(&["--paths-only", "--threshold", "-1"]);
    let served_empty = fixture.search(&["--paths-only", "--threshold", "1"]);
    assert_eq!(served.status.code(), found.status.code());
    assert_eq!(served.stdout, found.stdout);
    assert_eq!(served_empty.status.code(), empty.status.code());
    assert_eq!(served_empty.stdout, empty.stdout);
    let null = fixture.search(&["--null", "--threshold", "-1"]);
    assert_eq!(null.stdout, b"./photos/photo-\xff.png\0");
    assert_eq!(
        fixture
            .search(&["--json", "--threshold", "-1"])
            .status
            .code(),
        Some(2)
    );
}

#[test]
fn all_output_formats_match_for_a_cached_search() {
    let fixture = Fixture::new();
    assert!(fixture.seed_cached_image(std::ffi::OsStr::new("photo.png")));
    let flags = [
        vec!["--threshold", "-1"],
        vec!["--json", "--threshold", "-1"],
        vec!["--paths-only", "--threshold", "-1"],
        vec!["--null", "--threshold", "-1"],
        vec!["--paths-only", "--threshold", "1"],
    ];
    let one_shot: Vec<_> = flags
        .iter()
        .map(|flags| {
            fixture
                .search_command(flags)
                .arg("--no-server")
                .output()
                .unwrap()
        })
        .collect();
    assert!(fixture.socket_path().is_none());
    let mut inode = None;
    for (flags, expected) in flags.iter().zip(one_shot) {
        let actual = fixture.search(flags);
        assert_eq!(actual.status.code(), expected.status.code());
        assert_eq!(actual.stdout, expected.stdout);
        let current = fs::metadata(fixture.socket_path().unwrap()).unwrap().ino();
        assert_eq!(*inode.get_or_insert(current), current);
    }
}

#[test]
fn relative_query_image_is_resolved_in_client_directory() {
    let fixture = Fixture::new();
    let mut name = OsString::from_vec(b"query-\xff.png".to_vec());
    if !fixture.seed_cached_image(&name) {
        name = OsString::from("query.png");
        assert!(fixture.seed_cached_image(&name));
    }
    let _server = fixture.start_server();
    let output = fixture
        .command()
        .arg("--image")
        .arg(PathBuf::from("./photos").join(name))
        .arg("./photos")
        .args(["--paths-only", "--threshold", "-1"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.code(),
        Some(1),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn concurrent_start_replaces_a_stale_socket_and_a_duplicate_exits_quietly() {
    let fixture = Fixture::new();
    let server = fixture.start_server();
    let path = fixture.socket_path().unwrap();
    drop(server);
    assert!(path.exists());

    let first = fixture
        .search_command(&[])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let second = fixture
        .search_command(&[])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    assert_status(&first.wait_with_output().unwrap(), 1);
    assert_status(&second.wait_with_output().unwrap(), 1);
    assert!(UnixStream::connect(&path).is_ok());
    let duplicate = fixture
        .command()
        .args(["serve", "--socket"])
        .output()
        .unwrap();
    assert_status(&duplicate, 0);
    assert!(duplicate.stdout.is_empty());
    assert!(duplicate.stderr.is_empty());
}

#[test]
fn an_unusable_socket_path_does_not_break_one_shot_search() {
    let fixture = Fixture::new();
    let long_cache_path = fixture.cache.join("x".repeat(150));
    let output = fixture
        .command()
        .env("XDG_CACHE_HOME", long_cache_path)
        .arg(QUERY)
        .arg("./photos")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
}

#[test]
fn special_modes_bypass_a_running_socket() {
    let fixture = Fixture::new();
    let server = fixture.start_server();
    let path = fixture.socket_path().unwrap();
    drop(server);
    fs::remove_file(&path).unwrap();
    let listener = UnixListener::bind(&path).unwrap();
    listener.set_nonblocking(true).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let worker_stop = Arc::clone(&stop);
    let worker = thread::spawn(move || {
        while !worker_stop.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    stream.set_nonblocking(false).unwrap();
                    let mut request = String::new();
                    BufReader::new(&stream).read_line(&mut request).unwrap();
                    writeln!(stream, r#"{{"Error":"unexpected server connection"}}"#).unwrap();
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5))
                }
                Err(error) => panic!("accept failed: {error}"),
            }
        }
    });
    // A normal search connects and fails, proving that the fake server is reachable.
    assert_eq!(fixture.search(&[]).status.code(), Some(2));
    for flag in ["--no-cache", "--verify-models", "--timing", "--no-server"] {
        assert_eq!(fixture.search(&[flag]).status.code(), Some(1), "{flag}");
    }
    let opted_out_stop = fixture
        .command()
        .args(["--no-server", "server", "stop"])
        .output()
        .unwrap();
    assert_status(&opted_out_stop, 2);
    assert!(
        String::from_utf8_lossy(&opted_out_stop.stderr)
            .contains("--no-server cannot be used with server stop")
    );
    assert_status(&fixture.search(&[]), 2);
    stop.store(true, Ordering::Relaxed);
    worker.join().unwrap();
    fs::remove_file(path).unwrap();
}

#[test]
fn one_server_searches_two_folders_with_client_relative_paths() {
    let fixture = Fixture::new();
    let other = fixture.scratch.path().join("other");
    fs::create_dir(&other).unwrap();
    assert!(fixture.seed_cached_image(std::ffi::OsStr::new("first.png")));
    assert!(fixture.seed_cached_image_in(&other, std::ffi::OsStr::new("second.png")));
    let _server = fixture.start_server();
    let first = fixture.search(&["--paths-only"]);
    let second = fixture
        .command()
        .args([
            QUERY,
            "./other",
            "--paths-only",
            "--index-path",
            "other/.visiongrep.db",
        ])
        .output()
        .unwrap();
    assert_status(&first, 0);
    assert_status(&second, 0);
    assert_eq!(first.stdout, b"./photos/first.png\n");
    assert_eq!(second.stdout, b"./other/second.png\n");
}

#[test]
fn stop_without_a_server_is_silent_and_never_starts_one() {
    let fixture = Fixture::new();
    let output = fixture.stop();
    assert_status(&output, 0);
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
    assert!(!fixture.cache.join("visiongrep/run").exists());
}

#[test]
fn stop_replies_before_an_in_flight_request_finishes_and_then_removes_the_socket() {
    let fixture = Fixture::new();
    assert!(fixture.seed_cached_image(std::ffi::OsStr::new("first.png")));
    let mut server = fixture.start_server();
    let path = fixture.socket_path().unwrap();
    let root = fixture.corpus.canonicalize().unwrap();
    let request = serde_json::json!({
        "Search": {
            "target": {
                "root": root.as_os_str().as_bytes(),
                "index": root.join(".visiongrep.db").as_os_str().as_bytes(),
            },
            "query": { "Text": QUERY },
            "top": 5,
            "threshold": -1.0,
            "reindex": false,
        }
    });
    // Queue the search connection before stop, keeping its request in flight until stop replies.
    let mut in_flight = UnixStream::connect(&path).unwrap();
    write!(in_flight, "{request}").unwrap();

    let stopped = fixture.stop();
    assert_status(&stopped, 0);
    assert!(stopped.stdout.is_empty());
    assert!(stopped.stderr.is_empty());
    assert!(server.child.try_wait().unwrap().is_none());

    writeln!(in_flight).unwrap();
    in_flight
        .set_read_timeout(Some(Duration::from_secs(5)))
        .unwrap();
    let mut response = String::new();
    BufReader::new(in_flight).read_line(&mut response).unwrap();
    let response: serde_json::Value = serde_json::from_str(&response).unwrap();
    assert_eq!(response["Results"].as_array().unwrap().len(), 1);
    server.wait_for_exit();
    assert!(!path.exists());
}

#[test]
fn reindex_and_search_serialize_in_and_out_of_the_server() {
    let fixture = Fixture::with_models();
    RgbImage::from_pixel(16, 16, Rgb([240, 20, 30]))
        .save(fixture.corpus.join("red.png"))
        .unwrap();
    let served = fixture.search(&["--json", "--threshold", "-1"]);
    let one_shot = fixture.search(&["--json", "--threshold", "-1", "--no-server"]);
    assert_status(&served, 0);
    assert_status(&one_shot, 0);
    assert_eq!(served.stdout, one_shot.stdout);

    let indexed = fixture
        .command()
        .args(["index", "./photos"])
        .output()
        .unwrap();
    assert_status(&indexed, 0);
    assert!(indexed.stdout.is_empty());
    assert!(String::from_utf8_lossy(&indexed.stderr).contains("Indexed 1 images"));

    for (rebuild_mode, search_mode) in [
        (&[][..], &[][..]),
        (&["--no-server"][..], &[][..]),
        (&["--no-server"][..], &["--no-server"][..]),
    ] {
        let rebuild = fixture
            .search_command(&["--reindex", "--threshold", "-1"])
            .args(rebuild_mode)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let search = fixture
            .search_command(&["--threshold", "-1"])
            .args(search_mode)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        assert_status(&rebuild.wait_with_output().unwrap(), 0);
        assert_status(&search.wait_with_output().unwrap(), 0);
        let connection = Connection::open(fixture.corpus.join(".visiongrep.db")).unwrap();
        assert_eq!(
            connection
                .query_row("PRAGMA integrity_check", [], |row| row.get::<_, String>(0))
                .unwrap(),
            "ok"
        );
        for table in ["images", "queries"] {
            let rows: i64 = connection
                .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(rows, 1);
        }
    }
}

#[test]
fn novel_text_search_completes_while_another_folder_is_indexing() {
    let fixture = Fixture::with_models();
    let image = RgbImage::from_pixel(16, 16, Rgb([240, 20, 30]));
    image.save(fixture.corpus.join("red.png")).unwrap();
    let prepared = fixture
        .command()
        .args(["index", "./photos"])
        .output()
        .unwrap();
    assert_status(&prepared, 0);

    let batch = fixture.scratch.path().join("batch");
    fs::create_dir(&batch).unwrap();
    for number in 0..96 {
        image.save(batch.join(format!("{number}.png"))).unwrap();
    }
    let mut indexing = fixture
        .command()
        .args(["index", "./batch"])
        .stderr(Stdio::piped())
        .stdout(Stdio::null())
        .spawn()
        .unwrap();
    let mut progress = BufReader::new(indexing.stderr.take().unwrap());
    let mut line = String::new();
    loop {
        line.clear();
        assert_ne!(
            progress.read_line(&mut line).unwrap(),
            0,
            "indexing never started"
        );
        if line.contains("Indexing 96 images") {
            break;
        }
    }
    let novel = fixture
        .command()
        .args(["a novel query", "./photos", "--threshold", "-1"])
        .output()
        .unwrap();
    let still_indexing = indexing.try_wait().unwrap().is_none();
    let mut stderr = String::new();
    std::io::Read::read_to_string(&mut progress, &mut stderr).unwrap();
    assert!(indexing.wait().unwrap().success(), "{stderr}");
    assert_status(&novel, 0);
    assert!(
        still_indexing,
        "the text query waited for indexing to finish"
    );
}
