use std::ffi::OsString;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, params};
use tempfile::{Builder as TempFileBuilder, NamedTempFile};

use crate::embedding::{EmbeddingContract, NormalizedEmbedding};
use crate::error::VisionGrepError;

use super::scan::ImageFile;

const EMBEDDING_CACHE_VERSION: i64 = 3;
const INDEX_FILE_NAME: &str = ".visiongrep.db";
const REINDEX_FILE_PREFIX: &str = ".visiongrep.db.reindex-";
const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(5);
const ROOT_METADATA_KEY: &str = "search_root";
const IMAGE_CONTRACT_METADATA_KEY: &str = "image_embedding_contract";
const QUERY_CONTRACT_METADATA_KEY: &str = "query_embedding_contract";

#[cfg(test)]
fn test_contract() -> EmbeddingContract {
    EmbeddingContract {
        image: "test-image-contract",
        query: "test-query-contract",
    }
}

#[derive(Debug, Clone)]
pub(crate) struct IndexLocation {
    path: PathBuf,
}

impl IndexLocation {
    /// Resolves custom relative paths against the process working directory.
    ///
    /// The parent is canonicalized while the database filename is preserved because the database
    /// need not exist yet. The default remains search-root-local for compatibility.
    pub(crate) fn resolve(
        search_root: &Path,
        requested: Option<&Path>,
    ) -> Result<Self, VisionGrepError> {
        let Some(requested) = requested else {
            return Ok(Self {
                path: search_root.join(INDEX_FILE_NAME),
            });
        };
        if requested.file_name().is_none() {
            return Err(VisionGrepError::IndexPathWithoutFileName {
                path: requested.to_owned(),
            });
        }
        let path = if requested.is_absolute() {
            requested.to_owned()
        } else {
            std::env::current_dir()
                .map_err(VisionGrepError::Io)?
                .join(requested)
        };
        if path.is_dir() {
            return Err(VisionGrepError::IndexPathIsDirectory { path });
        }
        let parent = path
            .parent()
            .ok_or_else(|| VisionGrepError::IndexPathWithoutFileName { path: path.clone() })?;
        let canonical_parent =
            parent
                .canonicalize()
                .map_err(|source| VisionGrepError::IndexFile {
                    operation: "resolving index parent directory",
                    path: parent.to_owned(),
                    source,
                })?;
        let file_name = path
            .file_name()
            .ok_or_else(|| VisionGrepError::IndexPathWithoutFileName { path: path.clone() })?;
        Ok(Self {
            path: canonical_parent.join(file_name),
        })
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ImageRecord {
    pub(crate) path: PathBuf,
    pub(crate) embedding: NormalizedEmbedding,
}

pub(super) enum ImageUpdate {
    Upsert {
        file: ImageFile,
        embedding: NormalizedEmbedding,
    },
    Delete {
        relative_path: PathBuf,
    },
}

pub(crate) struct ImageIndex {
    conn: Connection,
    path: PathBuf,
}

pub(crate) struct ReconciliationPlan {
    missing: Vec<ImageFile>,
    stale_paths: Vec<Vec<u8>>,
}

impl ReconciliationPlan {
    pub(crate) fn missing(&self) -> &[ImageFile] {
        &self.missing
    }
}

struct CachedImageMetadata {
    path: Vec<u8>,
    mtime_ns: i64,
    size: i64,
}

/// A complete replacement index that remains invisible until it is verified and installed.
///
/// The temporary file lives beside the active database, so installing it is a same-filesystem
/// rename. Dropping this value after any build error removes the staged database and leaves the
/// active index untouched.
pub(crate) struct StagedImageIndex {
    index: ImageIndex,
    temporary: NamedTempFile,
    destination: PathBuf,
}

impl ImageIndex {
    pub(crate) fn exists(location: &IndexLocation) -> bool {
        location.path.exists()
    }

    pub(crate) fn open(
        location: &IndexLocation,
        search_root: &Path,
        contract: EmbeddingContract,
    ) -> Result<Self, VisionGrepError> {
        let conn = Connection::open(location.path())?;
        Self::from_connection(conn, location.path().to_owned(), search_root, contract)
    }

    #[cfg(test)]
    fn in_memory() -> Result<Self, VisionGrepError> {
        let conn = Connection::open_in_memory()?;
        Self::from_connection(
            conn,
            PathBuf::from("<memory>"),
            Path::new("/test-root"),
            test_contract(),
        )
    }

    /// Plans incremental reconciliation from one ordered cache read and one merge walk.
    ///
    /// Discovery and cached paths are both ordered by exact native Unix bytes. Content hashing and
    /// rename reuse are deliberately outside the cache contract; nanosecond mtime and size remain
    /// the change detector.
    pub(crate) fn plan_reconciliation(
        &self,
        files: &[ImageFile],
    ) -> Result<ReconciliationPlan, VisionGrepError> {
        let cached = self.cached_image_metadata()?;
        let mut missing = Vec::new();
        let mut stale_paths = Vec::new();
        let mut discovered_index = 0;
        let mut cached_index = 0;

        while let (Some(file), Some(cached_file)) =
            (files.get(discovered_index), cached.get(cached_index))
        {
            match file
                .relative_path
                .as_os_str()
                .as_bytes()
                .cmp(&cached_file.path)
            {
                std::cmp::Ordering::Less => {
                    missing.push(file.clone());
                    discovered_index += 1;
                }
                std::cmp::Ordering::Equal => {
                    if (file.mtime_ns, file.size) != (cached_file.mtime_ns, cached_file.size) {
                        missing.push(file.clone());
                    }
                    discovered_index += 1;
                    cached_index += 1;
                }
                std::cmp::Ordering::Greater => {
                    stale_paths.push(cached_file.path.clone());
                    cached_index += 1;
                }
            }
        }
        missing.extend_from_slice(&files[discovered_index..]);
        stale_paths.extend(
            cached[cached_index..]
                .iter()
                .map(|cached_file| cached_file.path.clone()),
        );

        Ok(ReconciliationPlan {
            missing,
            stale_paths,
        })
    }

    /// Applies the stale portion of a previously computed reconciliation plan atomically.
    pub(crate) fn apply_reconciliation(
        &mut self,
        plan: &ReconciliationPlan,
    ) -> Result<(), VisionGrepError> {
        let transaction = self.conn.transaction()?;
        {
            let mut stmt = transaction.prepare("DELETE FROM images WHERE path = ?1")?;
            for path in &plan.stale_paths {
                stmt.execute([path])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    fn cached_image_metadata(&self) -> Result<Vec<CachedImageMetadata>, VisionGrepError> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, mtime_ns, size FROM images ORDER BY path")?;
        stmt.query_map([], |row| {
            Ok(CachedImageMetadata {
                path: row.get(0)?,
                mtime_ns: row.get(1)?,
                size: row.get(2)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()
        .map_err(VisionGrepError::Index)
    }

    pub(super) fn apply_updates(
        &mut self,
        updates: Vec<ImageUpdate>,
    ) -> Result<(), VisionGrepError> {
        let transaction = self.conn.transaction()?;
        {
            let mut upsert = transaction.prepare(
                "INSERT INTO images (path, mtime_ns, size, embedding)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(path) DO UPDATE SET
                   mtime_ns = excluded.mtime_ns,
                   size = excluded.size,
                   embedding = excluded.embedding",
            )?;
            let mut delete = transaction.prepare("DELETE FROM images WHERE path = ?1")?;
            for update in updates {
                match update {
                    ImageUpdate::Upsert { file, embedding } => {
                        upsert.execute(params![
                            file.relative_path.as_os_str().as_bytes(),
                            file.mtime_ns,
                            file.size,
                            embedding.to_le_bytes(),
                        ])?;
                    }
                    ImageUpdate::Delete { relative_path } => {
                        delete.execute([relative_path.as_os_str().as_bytes()])?;
                    }
                }
            }
        }
        transaction.commit()?;
        Ok(())
    }

    /// Loads image embeddings while restoring exact native Unix path bytes and validating blobs.
    pub(crate) fn all_embeddings(
        &self,
        display_root: &Path,
    ) -> Result<Vec<ImageRecord>, VisionGrepError> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, embedding FROM images ORDER BY path")?;
        let records = stmt
            .query_map([], |row| {
                let path: Vec<u8> = row.get(0)?;
                let bytes: Vec<u8> = row.get(1)?;
                Ok((PathBuf::from(OsString::from_vec(path)), bytes))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        records
            .into_iter()
            .map(|(relative_path, bytes)| {
                let path = display_root.join(relative_path);
                let embedding = NormalizedEmbedding::from_le_bytes(&bytes).map_err(|source| {
                    VisionGrepError::InvalidCachedImageEmbedding {
                        path: path.clone(),
                        source,
                    }
                })?;
                Ok(ImageRecord { path, embedding })
            })
            .collect()
    }

    /// Looks up a query by exact text; case and whitespace differences are distinct cache keys.
    pub(crate) fn query_embedding(
        &self,
        query: &str,
    ) -> Result<Option<NormalizedEmbedding>, VisionGrepError> {
        let bytes = self
            .conn
            .query_row(
                "SELECT embedding FROM queries WHERE query = ?1",
                [query],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;

        bytes
            .map(|bytes| {
                NormalizedEmbedding::from_le_bytes(&bytes)
                    .map_err(|source| VisionGrepError::InvalidCachedQueryEmbedding { source })
            })
            .transpose()
    }

    pub(crate) fn upsert_query_embedding(
        &mut self,
        query: &str,
        embedding: &NormalizedEmbedding,
    ) -> Result<(), VisionGrepError> {
        self.conn.execute(
            "INSERT INTO queries (query, embedding)
             VALUES (?1, ?2)
             ON CONFLICT(query) DO UPDATE SET embedding = excluded.embedding",
            params![query, embedding.to_le_bytes()],
        )?;
        Ok(())
    }

    fn from_connection(
        conn: Connection,
        path: PathBuf,
        search_root: &Path,
        contract: EmbeddingContract,
    ) -> Result<Self, VisionGrepError> {
        conn.busy_timeout(SQLITE_BUSY_TIMEOUT)?;
        let mut index = Self { conn, path };
        index.init(search_root, contract)?;
        Ok(index)
    }

    /// Initializes or migrates the persisted embedding contract transactionally.
    ///
    /// Version 2 is migrated without losing its known current embeddings. Contract changes clear
    /// only the affected vector family, while a wrong-root reuse is rejected before mutation.
    fn init(
        &mut self,
        search_root: &Path,
        contract: EmbeddingContract,
    ) -> Result<(), VisionGrepError> {
        let version = self
            .conn
            .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))?;
        if version > EMBEDDING_CACHE_VERSION {
            return Err(VisionGrepError::IndexVersionTooNew {
                found: version,
                supported: EMBEDDING_CACHE_VERSION,
            });
        }

        if version == EMBEDDING_CACHE_VERSION {
            let stored_root = self.required_metadata(ROOT_METADATA_KEY)?;
            let stored_root = PathBuf::from(OsString::from_vec(stored_root));
            if stored_root != search_root {
                return Err(VisionGrepError::IndexRootMismatch {
                    index: self.path.clone(),
                    expected: search_root.to_owned(),
                    found: stored_root,
                });
            }
        }

        let index_path = self.path.clone();
        let transaction = self.conn.transaction()?;
        if version < 2 {
            transaction.execute_batch(
                "DROP TABLE IF EXISTS images;
                 DROP TABLE IF EXISTS queries;
                 DROP TABLE IF EXISTS metadata;",
            )?;
        }
        transaction.execute_batch(
            "CREATE TABLE IF NOT EXISTS images (
                path       BLOB PRIMARY KEY,
                mtime_ns   INTEGER NOT NULL,
                size       INTEGER NOT NULL,
                embedding  BLOB NOT NULL CHECK(length(embedding) = 2048)
            );
             CREATE TABLE IF NOT EXISTS queries (
                query      TEXT PRIMARY KEY,
                embedding  BLOB NOT NULL CHECK(length(embedding) = 2048)
            );
             CREATE TABLE IF NOT EXISTS metadata (
                key        TEXT PRIMARY KEY,
                value      BLOB NOT NULL
            );",
        )?;

        if version < EMBEDDING_CACHE_VERSION {
            set_metadata(
                &transaction,
                ROOT_METADATA_KEY,
                search_root.as_os_str().as_bytes(),
            )?;
            set_metadata(
                &transaction,
                IMAGE_CONTRACT_METADATA_KEY,
                contract.image.as_bytes(),
            )?;
            set_metadata(
                &transaction,
                QUERY_CONTRACT_METADATA_KEY,
                contract.query.as_bytes(),
            )?;
        } else {
            let stored_image_contract = required_transaction_metadata(
                &transaction,
                &index_path,
                IMAGE_CONTRACT_METADATA_KEY,
            )?;
            if stored_image_contract != contract.image.as_bytes() {
                transaction.execute("DELETE FROM images", [])?;
                set_metadata(
                    &transaction,
                    IMAGE_CONTRACT_METADATA_KEY,
                    contract.image.as_bytes(),
                )?;
            }

            let stored_query_contract = required_transaction_metadata(
                &transaction,
                &index_path,
                QUERY_CONTRACT_METADATA_KEY,
            )?;
            if stored_query_contract != contract.query.as_bytes() {
                transaction.execute("DELETE FROM queries", [])?;
                set_metadata(
                    &transaction,
                    QUERY_CONTRACT_METADATA_KEY,
                    contract.query.as_bytes(),
                )?;
            }
        }
        if version < EMBEDDING_CACHE_VERSION {
            transaction.pragma_update(None, "user_version", EMBEDDING_CACHE_VERSION)?;
        }
        transaction.commit()?;
        Ok(())
    }

    fn required_metadata(&self, key: &'static str) -> Result<Vec<u8>, VisionGrepError> {
        self.conn
            .query_row("SELECT value FROM metadata WHERE key = ?1", [key], |row| {
                row.get(0)
            })
            .optional()?
            .ok_or_else(|| VisionGrepError::IndexMetadataMissing {
                path: self.path.clone(),
                key,
            })
    }

    fn close(self) -> Result<(), VisionGrepError> {
        self.conn
            .close()
            .map_err(|(_, source)| VisionGrepError::Index(source))
    }
}

fn set_metadata(
    transaction: &rusqlite::Transaction<'_>,
    key: &str,
    value: &[u8],
) -> Result<(), rusqlite::Error> {
    transaction.execute(
        "INSERT INTO metadata (key, value) VALUES (?1, ?2)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

fn required_transaction_metadata(
    transaction: &rusqlite::Transaction<'_>,
    index_path: &Path,
    key: &'static str,
) -> Result<Vec<u8>, VisionGrepError> {
    transaction
        .query_row("SELECT value FROM metadata WHERE key = ?1", [key], |row| {
            row.get(0)
        })
        .optional()?
        .ok_or_else(|| VisionGrepError::IndexMetadataMissing {
            path: index_path.to_owned(),
            key,
        })
}

impl StagedImageIndex {
    pub(crate) fn create(
        location: &IndexLocation,
        search_root: &Path,
        contract: EmbeddingContract,
    ) -> Result<Self, VisionGrepError> {
        let parent =
            location
                .path()
                .parent()
                .ok_or_else(|| VisionGrepError::IndexPathWithoutFileName {
                    path: location.path().to_owned(),
                })?;
        let temporary = TempFileBuilder::new()
            .prefix(REINDEX_FILE_PREFIX)
            .tempfile_in(parent)
            .map_err(|source| VisionGrepError::IndexFile {
                operation: "creating staged reindex",
                path: parent.to_owned(),
                source,
            })?;
        let conn = Connection::open(temporary.path())?;
        let index =
            ImageIndex::from_connection(conn, temporary.path().to_owned(), search_root, contract)?;

        Ok(Self {
            index,
            temporary,
            destination: location.path().to_owned(),
        })
    }

    pub(crate) fn index_mut(&mut self) -> &mut ImageIndex {
        &mut self.index
    }

    /// Verifies the completed database before making it visible at the active index path.
    pub(crate) fn install(self) -> Result<(), VisionGrepError> {
        let Self {
            index,
            temporary,
            destination,
        } = self;
        let check = index
            .conn
            .query_row("PRAGMA quick_check(1)", [], |row| row.get::<_, String>(0))?;
        if check != "ok" {
            return Err(VisionGrepError::IndexIntegrity {
                path: temporary.path().to_owned(),
                detail: check,
            });
        }

        // Reading through the typed API also validates every persisted embedding blob.
        index.all_embeddings(Path::new(""))?;
        index.close()?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|source| VisionGrepError::IndexFile {
                operation: "syncing staged reindex",
                path: temporary.path().to_owned(),
                source,
            })?;
        temporary
            .persist(&destination)
            .map_err(|error| VisionGrepError::IndexFile {
                operation: "installing staged reindex",
                path: destination.clone(),
                source: error.error,
            })?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::time::Instant;

    use super::*;

    fn local_location(root: &Path) -> IndexLocation {
        IndexLocation::resolve(root, None).unwrap()
    }

    fn open_disk_index(root: &Path) -> ImageIndex {
        ImageIndex::open(&local_location(root), root, test_contract()).unwrap()
    }

    fn create_staged_index(root: &Path) -> StagedImageIndex {
        StagedImageIndex::create(&local_location(root), root, test_contract()).unwrap()
    }

    fn embedding() -> NormalizedEmbedding {
        let mut values = vec![0.0; crate::embedding::EMBEDDING_DIM];
        values[0] = 0.6;
        values[1] = 0.8;
        NormalizedEmbedding::from_model_output(values).unwrap()
    }

    fn image_file(path: PathBuf) -> ImageFile {
        ImageFile {
            relative_path: path,
            mtime_ns: 10,
            size: 20,
        }
    }

    fn insert(index: &mut ImageIndex, file: &ImageFile) {
        index
            .apply_updates(vec![ImageUpdate::Upsert {
                file: file.clone(),
                embedding: embedding(),
            }])
            .unwrap();
    }

    fn populate_reconciliation_benchmark(index: &mut ImageIndex, files: &[ImageFile]) {
        let transaction = index.conn.transaction().unwrap();
        let bytes = embedding().to_le_bytes();
        {
            let mut statement = transaction
                .prepare(
                    "INSERT INTO images (path, mtime_ns, size, embedding)
                     VALUES (?1, ?2, ?3, ?4)",
                )
                .unwrap();
            for file in files {
                statement
                    .execute(params![
                        file.relative_path.as_os_str().as_bytes(),
                        file.mtime_ns,
                        file.size,
                        &bytes,
                    ])
                    .unwrap();
            }
        }
        transaction.commit().unwrap();
    }

    fn per_file_query_changed_count(index: &ImageIndex, files: &[ImageFile]) -> usize {
        let mut statement = index
            .conn
            .prepare("SELECT mtime_ns, size FROM images WHERE path = ?1")
            .unwrap();
        files
            .iter()
            .filter(|file| {
                let cached = statement
                    .query_row([file.relative_path.as_os_str().as_bytes()], |row| {
                        Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
                    })
                    .optional()
                    .unwrap();
                cached != Some((file.mtime_ns, file.size))
            })
            .count()
    }

    fn duration_summary(mut samples: Vec<f64>) -> serde_json::Value {
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        let p95 = samples[(samples.len() * 95).div_ceil(100) - 1];
        serde_json::json!({"samples": samples.len(), "median_ms": median, "p95_ms": p95})
    }

    #[test]
    fn embedding_round_trips_through_bytes() {
        let embedding = embedding();
        let bytes = embedding.to_le_bytes();

        assert_eq!(
            NormalizedEmbedding::from_le_bytes(&bytes).unwrap(),
            embedding
        );
    }

    #[test]
    fn index_round_trips_embeddings() {
        let mut index = ImageIndex::in_memory().unwrap();
        let file = image_file(PathBuf::from("image.jpg"));

        insert(&mut index, &file);

        let records = index.all_embeddings(Path::new("photos")).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].path, PathBuf::from("photos/image.jpg"));
        assert_eq!(records[0].embedding, embedding());
        assert!(
            index
                .plan_reconciliation(&[file])
                .unwrap()
                .missing()
                .is_empty()
        );
    }

    #[test]
    fn removes_entries_for_renamed_or_deleted_images() {
        let mut index = ImageIndex::in_memory().unwrap();
        let old_file = image_file(PathBuf::from("old-name.jpg"));
        let renamed_file = image_file(PathBuf::from("new-name.jpg"));

        insert(&mut index, &old_file);
        let plan = index
            .plan_reconciliation(std::slice::from_ref(&renamed_file))
            .unwrap();
        index.apply_reconciliation(&plan).unwrap();

        assert!(index.all_embeddings(Path::new("")).unwrap().is_empty());
        assert_eq!(plan.missing().len(), 1);
    }

    #[test]
    fn bulk_reconciliation_preserves_unchanged_and_detects_metadata_changes() {
        let mut index = ImageIndex::in_memory().unwrap();
        let unchanged = image_file(PathBuf::from("a.jpg"));
        let changed = image_file(PathBuf::from("b.jpg"));
        insert(&mut index, &unchanged);
        insert(&mut index, &changed);
        let changed = ImageFile {
            relative_path: changed.relative_path,
            mtime_ns: changed.mtime_ns + 1,
            size: changed.size,
        };

        let plan = index
            .plan_reconciliation(&[unchanged, changed.clone()])
            .unwrap();

        assert_eq!(plan.missing().len(), 1);
        assert_eq!(plan.missing()[0].relative_path, changed.relative_path);
        index.apply_reconciliation(&plan).unwrap();
        assert_eq!(index.all_embeddings(Path::new("")).unwrap().len(), 2);
    }

    #[test]
    fn bulk_reconciliation_handles_discovered_paths_before_and_after_cached_paths() {
        let mut index = ImageIndex::in_memory().unwrap();
        let cached = image_file(PathBuf::from("middle.jpg"));
        insert(&mut index, &cached);
        let first = image_file(PathBuf::from("first.jpg"));
        let last = image_file(PathBuf::from("z-last.jpg"));

        let plan = index
            .plan_reconciliation(&[first.clone(), cached, last.clone()])
            .unwrap();

        assert_eq!(plan.missing().len(), 2);
        assert_eq!(plan.missing()[0].relative_path, first.relative_path);
        assert_eq!(plan.missing()[1].relative_path, last.relative_path);
    }

    #[test]
    #[ignore = "release benchmark for bulk index reconciliation"]
    fn reconciliation_performance_matrix() {
        let mut results = Vec::new();
        for corpus_size in [10_000, 100_000] {
            let files = (0..corpus_size)
                .map(|number| image_file(PathBuf::from(format!("image-{number:06}.jpg"))))
                .collect::<Vec<_>>();
            let mut changed = files.clone();
            for file in changed.iter_mut().step_by(100) {
                file.mtime_ns += 1;
            }

            let mut index = ImageIndex::in_memory().unwrap();
            populate_reconciliation_benchmark(&mut index, &files);
            for (scenario, discovered, expected_changed) in [
                ("unchanged", &files, 0),
                ("one_percent_changed", &changed, corpus_size / 100),
            ] {
                let mut per_file_samples = Vec::new();
                let mut bulk_samples = Vec::new();
                for _ in 0..21 {
                    let started = Instant::now();
                    let actual = per_file_query_changed_count(&index, discovered);
                    per_file_samples.push(started.elapsed().as_secs_f64() * 1_000.0);
                    assert_eq!(actual, expected_changed);

                    let started = Instant::now();
                    let plan = index.plan_reconciliation(discovered).unwrap();
                    bulk_samples.push(started.elapsed().as_secs_f64() * 1_000.0);
                    assert_eq!(plan.missing().len(), expected_changed);
                    assert!(plan.stale_paths.is_empty());
                }
                results.push(serde_json::json!({
                    "corpus_size": corpus_size,
                    "scenario": scenario,
                    "per_file_queries": duration_summary(per_file_samples),
                    "ordered_bulk_pass": duration_summary(bulk_samples),
                }));
            }
        }
        println!("{}", serde_json::to_string(&results).unwrap());
    }

    #[test]
    fn image_and_query_contracts_are_invalidated_independently() {
        let directory = tempfile::tempdir().unwrap();
        let location = local_location(directory.path());
        let file = image_file(PathBuf::from("image.jpg"));
        let query_embedding = embedding();
        {
            let mut index = ImageIndex::open(&location, directory.path(), test_contract()).unwrap();
            insert(&mut index, &file);
            index
                .upsert_query_embedding("cached query", &query_embedding)
                .unwrap();
        }

        let image_changed = EmbeddingContract {
            image: "changed-image-contract",
            query: test_contract().query,
        };
        {
            let mut index = ImageIndex::open(&location, directory.path(), image_changed).unwrap();
            assert!(index.all_embeddings(Path::new("")).unwrap().is_empty());
            assert_eq!(
                index.query_embedding("cached query").unwrap(),
                Some(query_embedding.clone())
            );
            insert(&mut index, &file);
        }

        let query_changed = EmbeddingContract {
            image: image_changed.image,
            query: "changed-query-contract",
        };
        let index = ImageIndex::open(&location, directory.path(), query_changed).unwrap();
        assert_eq!(index.all_embeddings(Path::new("")).unwrap().len(), 1);
        assert!(index.query_embedding("cached query").unwrap().is_none());
    }

    #[test]
    fn custom_index_rejects_reuse_for_another_search_root() {
        let directory = tempfile::tempdir().unwrap();
        let first_root = directory.path().join("first");
        let second_root = directory.path().join("second");
        fs::create_dir(&first_root).unwrap();
        fs::create_dir(&second_root).unwrap();
        let index_path = directory.path().join("shared.db");
        let location = IndexLocation::resolve(&first_root, Some(&index_path)).unwrap();
        drop(ImageIndex::open(&location, &first_root, test_contract()).unwrap());

        let error = match ImageIndex::open(&location, &second_root, test_contract()) {
            Ok(_) => panic!("wrong-root index reuse unexpectedly succeeded"),
            Err(error) => error,
        };

        assert!(matches!(error, VisionGrepError::IndexRootMismatch { .. }));
    }

    #[test]
    fn custom_index_does_not_write_inside_search_root() {
        let root = tempfile::tempdir().unwrap();
        let index_directory = tempfile::tempdir().unwrap();
        let index_path = index_directory.path().join("external.db");
        let location = IndexLocation::resolve(root.path(), Some(&index_path)).unwrap();

        drop(ImageIndex::open(&location, root.path(), test_contract()).unwrap());

        assert!(index_path.exists());
        assert!(!root.path().join(INDEX_FILE_NAME).exists());
    }

    #[test]
    fn relative_custom_index_is_resolved_from_working_directory() {
        let root = tempfile::tempdir().unwrap();
        let working_directory = std::env::current_dir().unwrap().canonicalize().unwrap();

        let location = IndexLocation::resolve(root.path(), Some(Path::new("relative.db"))).unwrap();

        assert_eq!(location.path(), working_directory.join("relative.db"));
    }

    #[test]
    fn staged_custom_index_lives_beside_its_destination() {
        let root = tempfile::tempdir().unwrap();
        let index_directory = tempfile::tempdir().unwrap();
        let location =
            IndexLocation::resolve(root.path(), Some(&index_directory.path().join("index.db")))
                .unwrap();
        let staged = StagedImageIndex::create(&location, root.path(), test_contract()).unwrap();

        assert_eq!(
            staged.temporary.path().parent(),
            Some(index_directory.path())
        );
        staged.install().unwrap();
        assert!(location.path().exists());
    }

    #[test]
    fn query_embeddings_round_trip() {
        let mut index = ImageIndex::in_memory().unwrap();

        assert!(index.query_embedding("cable drums").unwrap().is_none());
        let embedding = embedding();
        index
            .upsert_query_embedding("cable drums", &embedding)
            .unwrap();

        assert_eq!(
            index.query_embedding("cable drums").unwrap(),
            Some(embedding)
        );
    }

    #[test]
    fn incompatible_cache_version_clears_embeddings() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            "CREATE TABLE images (
                path TEXT PRIMARY KEY,
                mtime INTEGER NOT NULL,
                size INTEGER NOT NULL,
                embedding BLOB NOT NULL
             );
             INSERT INTO images VALUES ('old.jpg', 10, 20, X'00000000');",
        )
        .unwrap();

        let index = ImageIndex::from_connection(
            conn,
            PathBuf::from("<memory>"),
            Path::new("/test-root"),
            test_contract(),
        )
        .unwrap();

        assert!(index.all_embeddings(Path::new("")).unwrap().is_empty());
        let version = index
            .conn
            .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
            .unwrap();
        assert_eq!(version, EMBEDDING_CACHE_VERSION);
    }

    #[test]
    fn version_two_index_migrates_without_losing_known_embeddings() {
        let directory = tempfile::tempdir().unwrap();
        let location = local_location(directory.path());
        let conn = Connection::open(location.path()).unwrap();
        conn.execute_batch(
            "CREATE TABLE images (
                path BLOB PRIMARY KEY,
                mtime_ns INTEGER NOT NULL,
                size INTEGER NOT NULL,
                embedding BLOB NOT NULL CHECK(length(embedding) = 2048)
             );
             CREATE TABLE queries (
                query TEXT PRIMARY KEY,
                embedding BLOB NOT NULL CHECK(length(embedding) = 2048)
             );
             PRAGMA user_version = 2;",
        )
        .unwrap();
        conn.execute(
            "INSERT INTO images (path, mtime_ns, size, embedding) VALUES (?1, 10, 20, ?2)",
            params![b"image.jpg", embedding().to_le_bytes()],
        )
        .unwrap();
        drop(conn);

        let index = ImageIndex::open(&location, directory.path(), test_contract()).unwrap();

        assert_eq!(index.all_embeddings(Path::new("")).unwrap().len(), 1);
        assert_eq!(
            index
                .conn
                .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
                .unwrap(),
            EMBEDDING_CACHE_VERSION
        );
    }

    #[test]
    fn current_schema_without_root_metadata_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let location = local_location(directory.path());
        let conn = Connection::open(location.path()).unwrap();
        conn.execute_batch(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value BLOB NOT NULL);
             PRAGMA user_version = 3;",
        )
        .unwrap();
        drop(conn);

        let error = match ImageIndex::open(&location, directory.path(), test_contract()) {
            Ok(_) => panic!("index without root metadata unexpectedly opened"),
            Err(error) => error,
        };

        assert!(matches!(
            error,
            VisionGrepError::IndexMetadataMissing { .. }
        ));
    }

    #[test]
    fn index_preserves_non_utf8_paths() {
        let mut index = ImageIndex::in_memory().unwrap();
        let path = PathBuf::from(OsString::from_vec(vec![
            b'i', b'm', b'g', 0xff, b'.', b'j', b'p', b'g',
        ]));
        let file = image_file(path.clone());

        insert(&mut index, &file);

        assert_eq!(index.all_embeddings(Path::new("")).unwrap()[0].path, path);
    }

    #[test]
    fn invalid_cached_embedding_is_rejected() {
        let index = ImageIndex::in_memory().unwrap();
        index
            .conn
            .execute(
                "INSERT INTO images (path, mtime_ns, size, embedding) VALUES (?1, 10, 20, ?2)",
                params![b"bad.jpg", vec![0_u8; 2048]],
            )
            .unwrap();

        let error = index.all_embeddings(Path::new("")).unwrap_err();

        assert!(matches!(
            error,
            VisionGrepError::InvalidCachedImageEmbedding { .. }
        ));
    }

    #[test]
    fn update_batch_is_atomic() {
        let mut index = ImageIndex::in_memory().unwrap();
        index
            .conn
            .execute_batch(
                "CREATE TRIGGER reject_bad_image
                 BEFORE INSERT ON images
                 WHEN NEW.path = X'6261642E6A7067'
                 BEGIN
                   SELECT RAISE(ABORT, 'rejected test image');
                 END;",
            )
            .unwrap();
        let good = image_file(PathBuf::from("good.jpg"));
        let bad = image_file(PathBuf::from("bad.jpg"));

        assert!(
            index
                .apply_updates(vec![
                    ImageUpdate::Upsert {
                        file: good,
                        embedding: embedding(),
                    },
                    ImageUpdate::Upsert {
                        file: bad,
                        embedding: embedding(),
                    },
                ])
                .is_err()
        );
        assert!(index.all_embeddings(Path::new("")).unwrap().is_empty());
    }

    #[test]
    fn failed_staged_reindex_preserves_the_complete_active_index() {
        let directory = tempfile::tempdir().unwrap();
        let original_file = image_file(PathBuf::from("original.jpg"));
        let original_query = embedding();
        {
            let mut original = open_disk_index(directory.path());
            insert(&mut original, &original_file);
            original
                .upsert_query_embedding("cached query", &original_query)
                .unwrap();
        }

        let mut staged = create_staged_index(directory.path());
        let first_batch = (0..super::super::ingest::INDEX_BATCH_SIZE)
            .map(|number| ImageUpdate::Upsert {
                file: image_file(PathBuf::from(format!("staged-{number}.jpg"))),
                embedding: embedding(),
            })
            .collect();
        staged.index_mut().apply_updates(first_batch).unwrap();

        // A reader opening the active path during the rebuild still sees the complete old index.
        let concurrent_reader = open_disk_index(directory.path());
        assert_eq!(
            concurrent_reader
                .all_embeddings(Path::new(""))
                .unwrap()
                .len(),
            1
        );
        drop(concurrent_reader);

        staged
            .index_mut()
            .conn
            .execute_batch(
                "CREATE TRIGGER reject_later_batch
                 BEFORE INSERT ON images
                 WHEN NEW.path = X'6661696C2E6A7067'
                 BEGIN
                   SELECT RAISE(ABORT, 'injected later-batch failure');
                 END;",
            )
            .unwrap();
        let later_batch = vec![ImageUpdate::Upsert {
            file: image_file(PathBuf::from("fail.jpg")),
            embedding: embedding(),
        }];

        assert!(staged.index_mut().apply_updates(later_batch).is_err());
        drop(staged);

        let original = open_disk_index(directory.path());
        let records = original.all_embeddings(Path::new("")).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].path, original_file.relative_path);
        assert_eq!(
            original.query_embedding("cached query").unwrap(),
            Some(original_query)
        );
    }

    #[test]
    fn completed_staged_reindex_replaces_the_active_index() {
        let directory = tempfile::tempdir().unwrap();
        let old_file = image_file(PathBuf::from("old.jpg"));
        {
            let mut original = open_disk_index(directory.path());
            insert(&mut original, &old_file);
        }

        let new_file = image_file(PathBuf::from("new.jpg"));
        let mut staged = create_staged_index(directory.path());
        insert(staged.index_mut(), &new_file);
        staged.install().unwrap();
        let installed = open_disk_index(directory.path());

        let records = installed.all_embeddings(Path::new("")).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].path, new_file.relative_path);
    }
}
