use std::collections::HashSet;
use std::ffi::OsString;
use std::fs;
use std::os::unix::ffi::{OsStrExt, OsStringExt};
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OptionalExtension, params};

use crate::error::VisionGrepError;

use super::scan::ImageFile;

const EMBEDDING_CACHE_VERSION: i64 = 1;

#[derive(Debug, Clone)]
pub(crate) struct ImageRecord {
    pub(crate) path: PathBuf,
    pub(crate) embedding: Vec<f32>,
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

    pub(crate) fn reindex(root: &Path) -> Result<Self, VisionGrepError> {
        let db_path = root.join(".visiongrep.db");
        if db_path.exists() {
            fs::remove_file(&db_path)?;
        }
        Self::open(root)
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
                .query_row([file.path.as_os_str().as_bytes()], |row| {
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
            .map(|file| file.path.as_os_str().as_bytes().to_vec())
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

    pub(crate) fn remove_entries(&mut self, files: &[ImageFile]) -> Result<(), VisionGrepError> {
        let transaction = self.conn.transaction()?;
        {
            let mut stmt = transaction.prepare("DELETE FROM images WHERE path = ?1")?;
            for file in files {
                stmt.execute([file.path.as_os_str().as_bytes()])?;
            }
        }
        transaction.commit()?;
        Ok(())
    }

    pub(crate) fn upsert_embedding(
        &mut self,
        file: &ImageFile,
        embedding: &[f32],
    ) -> Result<(), VisionGrepError> {
        let bytes = embedding_to_bytes(embedding);
        self.conn.execute(
            "INSERT INTO images (path, mtime_ns, size, embedding)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(path) DO UPDATE SET
               mtime_ns = excluded.mtime_ns,
               size = excluded.size,
               embedding = excluded.embedding",
            params![
                file.path.as_os_str().as_bytes(),
                file.mtime_ns,
                file.size,
                bytes
            ],
        )?;
        Ok(())
    }

    /// Loads image embeddings while restoring exact native Unix path bytes and validating blobs.
    pub(crate) fn all_embeddings(&self) -> Result<Vec<ImageRecord>, VisionGrepError> {
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
            .map(|(path, bytes)| {
                let embedding = bytes_to_embedding(&path, &bytes)?;
                Ok(ImageRecord { path, embedding })
            })
            .collect()
    }

    /// Looks up a query by exact text; case and whitespace differences are distinct cache keys.
    pub(crate) fn query_embedding(&self, query: &str) -> Result<Option<Vec<f32>>, VisionGrepError> {
        let bytes = self
            .conn
            .query_row(
                "SELECT embedding FROM queries WHERE query = ?1",
                [query],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .optional()?;

        bytes
            .map(|bytes| bytes_to_query_embedding(&bytes))
            .transpose()
    }

    pub(crate) fn upsert_query_embedding(
        &mut self,
        query: &str,
        embedding: &[f32],
    ) -> Result<(), VisionGrepError> {
        let bytes = embedding_to_bytes(embedding);
        self.conn.execute(
            "INSERT INTO queries (query, embedding)
             VALUES (?1, ?2)
             ON CONFLICT(query) DO UPDATE SET embedding = excluded.embedding",
            params![query, bytes],
        )?;
        Ok(())
    }

    fn from_connection(conn: Connection) -> Result<Self, VisionGrepError> {
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
                embedding  BLOB NOT NULL
            );
             CREATE TABLE IF NOT EXISTS queries (
                query      TEXT PRIMARY KEY,
                embedding  BLOB NOT NULL
            );",
        )?;
        if version < EMBEDDING_CACHE_VERSION {
            transaction.pragma_update(None, "user_version", EMBEDDING_CACHE_VERSION)?;
        }
        transaction.commit()?;
        Ok(())
    }
}

/// Encodes floats explicitly as little-endian bytes to make the SQLite representation deliberate.
fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(embedding));
    for value in embedding {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Decodes a stored image vector after validating byte alignment and finite values.
fn bytes_to_embedding(path: &Path, bytes: &[u8]) -> Result<Vec<f32>, VisionGrepError> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(VisionGrepError::InvalidEmbeddingBlob {
            path: path.to_owned(),
            len: bytes.len(),
        });
    }

    let embedding = decode_embedding(bytes);
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err(VisionGrepError::NonFiniteEmbedding {
            path: path.to_owned(),
        });
    }
    Ok(embedding)
}

fn bytes_to_query_embedding(bytes: &[u8]) -> Result<Vec<f32>, VisionGrepError> {
    if bytes.len() % std::mem::size_of::<f32>() != 0 {
        return Err(VisionGrepError::InvalidQueryEmbeddingBlob { len: bytes.len() });
    }

    let embedding = decode_embedding(bytes);
    if embedding.iter().any(|value| !value.is_finite()) {
        return Err(VisionGrepError::NonFiniteQueryEmbedding);
    }
    Ok(embedding)
}

fn decode_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_round_trips_through_bytes() {
        let embedding = vec![0.1, -0.2, 0.3];
        let bytes = embedding_to_bytes(&embedding);

        assert_eq!(
            bytes_to_embedding(Path::new("image.jpg"), &bytes).unwrap(),
            embedding
        );
    }

    #[test]
    fn index_round_trips_embeddings() {
        let mut index = ImageIndex::in_memory().unwrap();
        let file = ImageFile {
            path: PathBuf::from("image.jpg"),
            mtime_ns: 10,
            size: 20,
        };

        index.upsert_embedding(&file, &[0.6, 0.8]).unwrap();

        let records = index.all_embeddings().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].path, PathBuf::from("image.jpg"));
        assert_eq!(records[0].embedding, vec![0.6, 0.8]);
        assert!(index.images_needing_embedding(&[file]).unwrap().is_empty());
    }

    #[test]
    fn removes_entries_for_renamed_or_deleted_images() {
        let mut index = ImageIndex::in_memory().unwrap();
        let old_file = ImageFile {
            path: PathBuf::from("old-name.jpg"),
            mtime_ns: 10,
            size: 20,
        };
        let renamed_file = ImageFile {
            path: PathBuf::from("new-name.jpg"),
            mtime_ns: 10,
            size: 20,
        };

        index.upsert_embedding(&old_file, &[0.6, 0.8]).unwrap();
        index
            .remove_stale_entries(std::slice::from_ref(&renamed_file))
            .unwrap();

        assert!(index.all_embeddings().unwrap().is_empty());
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
        index
            .upsert_query_embedding("cable drums", &[0.6, 0.8])
            .unwrap();

        assert_eq!(
            index.query_embedding("cable drums").unwrap(),
            Some(vec![0.6, 0.8])
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

        assert!(index.all_embeddings().unwrap().is_empty());
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
        let file = ImageFile {
            path: path.clone(),
            mtime_ns: 10,
            size: 20,
        };

        index.upsert_embedding(&file, &[0.6, 0.8]).unwrap();

        assert_eq!(index.all_embeddings().unwrap()[0].path, path);
    }
}
