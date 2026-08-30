use std::collections::HashSet;
use std::ffi::OsString;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::{Connection, OptionalExtension, params};

use crate::embedding::NormalizedEmbedding;
use crate::error::VisionGrepError;

use super::scan::ImageFile;

const EMBEDDING_CACHE_VERSION: i64 = 2;
const SQLITE_BUSY_TIMEOUT: Duration = Duration::from_secs(5);

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
}

impl ImageIndex {
    pub(crate) fn open(root: &Path) -> Result<Self, VisionGrepError> {
        let db_path = root.join(".visiongrep.db");
        let conn = Connection::open(db_path)?;
        Self::from_connection(conn)
    }

    #[cfg(test)]
    fn in_memory() -> Result<Self, VisionGrepError> {
        let conn = Connection::open_in_memory()?;
        Self::from_connection(conn)
    }

    /// Returns files whose path is absent or whose nanosecond mtime or size no longer matches.
    ///
    /// This is intentionally metadata-based; content hashing and rename reuse are not part of the
    /// current cache contract.
    pub(crate) fn images_needing_embedding(
        &self,
        files: &[ImageFile],
    ) -> Result<Vec<ImageFile>, VisionGrepError> {
        let mut stmt = self
            .conn
            .prepare("SELECT mtime_ns, size FROM images WHERE path = ?1")?;
        let mut missing = Vec::new();

        for file in files {
            let cached: Option<(i64, i64)> = stmt
                .query_row([file.relative_path.as_os_str().as_bytes()], |row| {
                    Ok((row.get(0)?, row.get(1)?))
                })
                .optional()?;

            if cached != Some((file.mtime_ns, file.size)) {
                missing.push(file.clone());
            }
        }

        Ok(missing)
    }

    /// Removes cached paths that no longer appear in the current recursive discovery result.
    pub(crate) fn remove_stale_entries(
        &mut self,
        files: &[ImageFile],
    ) -> Result<(), VisionGrepError> {
        let current_paths = files
            .iter()
            .map(|file| file.relative_path.as_os_str().as_bytes().to_vec())
            .collect::<HashSet<_>>();
        let cached_paths = {
            let mut stmt = self.conn.prepare("SELECT path FROM images")?;
            stmt.query_map([], |row| row.get::<_, Vec<u8>>(0))?
                .collect::<Result<Vec<_>, _>>()?
        };
        let stale_paths: Vec<Vec<u8>> = cached_paths
            .into_iter()
            .filter(|path| !current_paths.contains(path))
            .collect();

        let transaction = self.conn.transaction()?;
        {
            let mut stmt = transaction.prepare("DELETE FROM images WHERE path = ?1")?;
            for path in stale_paths {
                stmt.execute([path])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    pub(crate) fn clear(&mut self) -> Result<(), VisionGrepError> {
        let transaction = self.conn.transaction()?;
        transaction.execute_batch("DELETE FROM images; DELETE FROM queries;")?;
        transaction.commit()?;
        Ok(())
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

    fn from_connection(conn: Connection) -> Result<Self, VisionGrepError> {
        conn.busy_timeout(SQLITE_BUSY_TIMEOUT)?;
        let mut index = Self { conn };
        index.init()?;
        Ok(index)
    }

    /// Initializes or migrates the persisted embedding contract transactionally.
    ///
    /// Older cache versions are incompatible and rebuilt. A newer version is rejected so an older
    /// binary cannot silently corrupt data whose schema or embedding semantics it does not know.
    fn init(&mut self) -> Result<(), VisionGrepError> {
        let version = self
            .conn
            .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))?;
        if version > EMBEDDING_CACHE_VERSION {
            return Err(VisionGrepError::IndexVersionTooNew {
                found: version,
                supported: EMBEDDING_CACHE_VERSION,
            });
        }

        let transaction = self.conn.transaction()?;
        if version < EMBEDDING_CACHE_VERSION {
            transaction.execute_batch(
                "DROP TABLE IF EXISTS images;
                 DROP TABLE IF EXISTS queries;",
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
            );",
        )?;
        if version < EMBEDDING_CACHE_VERSION {
            transaction.pragma_update(None, "user_version", EMBEDDING_CACHE_VERSION)?;
        }
        transaction.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        assert!(index.images_needing_embedding(&[file]).unwrap().is_empty());
    }

    #[test]
    fn removes_entries_for_renamed_or_deleted_images() {
        let mut index = ImageIndex::in_memory().unwrap();
        let old_file = image_file(PathBuf::from("old-name.jpg"));
        let renamed_file = image_file(PathBuf::from("new-name.jpg"));

        insert(&mut index, &old_file);
        index
            .remove_stale_entries(std::slice::from_ref(&renamed_file))
            .unwrap();

        assert!(index.all_embeddings(Path::new("")).unwrap().is_empty());
        assert_eq!(
            index
                .images_needing_embedding(&[renamed_file])
                .unwrap()
                .len(),
            1
        );
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

        let index = ImageIndex::from_connection(conn).unwrap();

        assert!(index.all_embeddings(Path::new("")).unwrap().is_empty());
        let version = index
            .conn
            .pragma_query_value(None, "user_version", |row| row.get::<_, i64>(0))
            .unwrap();
        assert_eq!(version, EMBEDDING_CACHE_VERSION);
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
}
