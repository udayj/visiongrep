use std::path::PathBuf;

use crate::error::VisionGrepError;
use crate::index::ImageRecord;

pub(crate) const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.25;

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct SearchResult {
    pub(crate) score: f32,
    pub(crate) path: PathBuf,
}

pub(crate) struct Ranker<'a> {
    query_embedding: &'a [f32],
    top: usize,
    threshold: f32,
}

impl<'a> Ranker<'a> {
    pub(crate) fn new(query_embedding: &'a [f32], top: usize, threshold: f32) -> Self {
        Self {
            query_embedding,
            top,
            threshold,
        }
    }

    /// Scores every image, filters weak matches, and returns a deterministic top-K ranking.
    ///
    /// Equal dimensions are checked before scoring. Score ties are resolved by path so repeated
    /// searches over the same index produce stable output.
    pub(crate) fn rank(
        &self,
        images: Vec<ImageRecord>,
    ) -> Result<Vec<SearchResult>, VisionGrepError> {
        let mut results = Vec::with_capacity(images.len().min(self.top));
        for image in images {
            if image.embedding.len() != self.query_embedding.len() {
                return Err(VisionGrepError::EmbeddingDimensionMismatch {
                    path: image.path,
                    query_len: self.query_embedding.len(),
                    image_len: image.embedding.len(),
                });
            }

            let score = cosine_similarity(self.query_embedding, &image.embedding);
            if score >= self.threshold {
                results.push(SearchResult {
                    score,
                    path: image.path,
                });
            }
        }

        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.path.cmp(&right.path))
        });
        results.truncate(self.top);
        Ok(results)
    }
}

/// Computes cosine similarity as a dot product under the embedding layer's unit-norm invariant.
fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_is_dot_product() {
        let left = [0.6, 0.8];
        let right = [0.8, 0.6];

        assert_eq!(cosine_similarity(&left, &right), 0.96000004);
    }

    #[test]
    fn ranking_filters_sorts_and_truncates() {
        let query = [1.0, 0.0];
        let ranker = Ranker::new(&query, 2, 0.5);
        let results = ranker
            .rank(vec![
                ImageRecord {
                    path: PathBuf::from("b.jpg"),
                    embedding: vec![0.7, 0.0],
                },
                ImageRecord {
                    path: PathBuf::from("a.jpg"),
                    embedding: vec![0.9, 0.0],
                },
                ImageRecord {
                    path: PathBuf::from("c.jpg"),
                    embedding: vec![0.4, 0.0],
                },
            ])
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].path, PathBuf::from("a.jpg"));
        assert_eq!(results[1].path, PathBuf::from("b.jpg"));
    }

    #[test]
    fn ranking_rejects_mismatched_embedding_dimensions() {
        let query = [1.0, 0.0];
        let ranker = Ranker::new(&query, 1, 0.0);

        let error = ranker
            .rank(vec![ImageRecord {
                path: PathBuf::from("bad.jpg"),
                embedding: vec![1.0],
            }])
            .unwrap_err();

        assert!(matches!(
            error,
            VisionGrepError::EmbeddingDimensionMismatch { .. }
        ));
    }
}
