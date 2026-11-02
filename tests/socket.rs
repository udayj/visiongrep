mod model_cache;

use std::ffi::OsString;
use std::fs;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::os::unix::net::UnixListener;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant, UNIX_EPOCH};

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

    fn search(&self, flags: &[&str]) -> Output {
        self.command()
            .arg(QUERY)
            .arg("./photos")
            .args(flags)
            .output()
            .unwrap()
    }

    fn seed_cached_image(&self, name: &std::ffi::OsStr) -> bool {
        // Let the application create the schema and its model contract metadata.
        assert_eq!(self.search(&[]).status.code(), Some(1));
        let image = self.corpus.join(name);
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
        let connection = Connection::open(self.corpus.join(".visiongrep.db")).unwrap();
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
        self.start_server_with_index(None)
    }

    fn start_server_with_index(&self, index: Option<&Path>) -> Server {
        let mut command = self.command();
        command.current_dir("/").args(["serve", "--socket"]);
        if let Some(index) = index {
            command.arg("--index-path").arg(index);
        }
        let child = command
            .arg(&self.corpus)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let mut server = Server { child };
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(path) = self.socket_path()
                && std::os::unix::net::UnixStream::connect(&path).is_ok()
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

    fn socket_path(&self) -> Option<PathBuf> {
        let run = self.cache.join("visiongrep/run");
        fs::read_dir(run)
            .ok()?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.extension().is_some_and(|ext| ext == "sock"))
    }
}

struct Server {
    child: Child,
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
    let found = fixture.search(&["--paths-only", "--threshold", "-1"]);
    let empty = fixture.search(&["--paths-only", "--threshold", "1"]);
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
    let one_shot: Vec<_> = flags.iter().map(|flags| fixture.search(flags)).collect();
    let _server = fixture.start_server();
    for (flags, expected) in flags.iter().zip(one_shot) {
        let actual = fixture.search(flags);
        assert_eq!(actual.status.code(), expected.status.code());
        assert_eq!(actual.stdout, expected.stdout);
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
fn stale_socket_falls_back_and_next_server_replaces_it() {
    let fixture = Fixture::new();
    let server = fixture.start_server();
    let path = fixture.socket_path().unwrap();
    drop(server);
    assert!(path.exists());
    let fallback = fixture.search(&[]);
    assert_eq!(fallback.status.code(), Some(1));
    assert!(fallback.stderr.is_empty());
    let _server = fixture.start_server();
    let duplicate = fixture
        .command()
        .args(["serve", "--socket"])
        .arg(&fixture.corpus)
        .output()
        .unwrap();
    assert_eq!(duplicate.status.code(), Some(2));
    assert!(duplicate.stdout.is_empty());
    assert!(String::from_utf8_lossy(&duplicate.stderr).contains("already running"));
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
                Ok((stream, _)) => drop(stream),
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(5))
                }
                Err(error) => panic!("accept failed: {error}"),
            }
        }
    });
    // A normal search connects and fails, proving that the fake server is reachable.
    assert_eq!(fixture.search(&[]).status.code(), Some(2));
    for flag in ["--no-cache", "--verify-models", "--timing"] {
        assert_eq!(fixture.search(&[flag]).status.code(), Some(1), "{flag}");
    }
    stop.store(true, Ordering::Relaxed);
    worker.join().unwrap();
}

#[test]
fn reindex_refuses_to_replace_the_index_while_socket_server_is_running() {
    let fixture = Fixture::new();
    assert!(fixture.seed_cached_image(std::ffi::OsStr::new("photo.png")));
    let _server = fixture.start_server();
    let index = fixture.corpus.join(".visiongrep.db");
    let before = fs::read(&index).unwrap();

    let output = fixture.search(&["--reindex", "--timing", "--verify-models"]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert_eq!(
        output.stderr,
        b"visiongrep: stop the socket server before --reindex\n"
    );
    assert_eq!(fs::read(&index).unwrap(), before);
    assert_eq!(fixture.search(&["--paths-only"]).status.code(), Some(0));
}

#[test]
fn reindex_only_refuses_the_index_used_by_the_socket_server() {
    let fixture = Fixture::new();
    let server_index = fixture.scratch.path().join("server.db");
    let other_index = fixture.scratch.path().join("other.db");
    let _server = fixture.start_server_with_index(Some(&server_index));
    let search = |index: &Path| {
        fixture
            .command()
            .arg(QUERY)
            .arg("./photos")
            .arg("--reindex")
            .arg("--index-path")
            .arg(index)
            .output()
            .unwrap()
    };

    let blocked = search(&server_index);
    assert_eq!(blocked.status.code(), Some(2));
    assert_eq!(
        blocked.stderr,
        b"visiongrep: stop the socket server before --reindex\n"
    );
    let allowed = search(&other_index);
    assert_eq!(allowed.status.code(), Some(1));
    assert!(other_index.exists());
}

#[test]
#[ignore = "requires installed model artifacts"]
fn novel_text_query_through_server_matches_one_shot() {
    use image::{Rgb, RgbImage};
    let fixture = Fixture::new();
    RgbImage::from_pixel(16, 16, Rgb([240, 20, 30]))
        .save(fixture.corpus.join("red.png"))
        .unwrap();
    model_cache::link_models(
        &fixture.cache,
        &[
            "datacomp_vision.onnx",
            "datacomp_text.onnx",
            "datacomp_tokenizer.json",
        ],
    );
    let _server = fixture.start_server();
    let served = fixture.search(&["--json", "--threshold", "-1"]);
    let one_shot = fixture.search(&["--json", "--threshold", "-1", "--timing"]);
    assert_eq!(served.status.code(), Some(0));
    assert_eq!(one_shot.status.code(), Some(0));
    assert_eq!(served.stdout, one_shot.stdout);
}
