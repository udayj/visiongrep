use std::path::PathBuf;

use crate::embedding::NormalizedEmbedding;
use crate::index::ImageRecord;
use crate::timing::{Phase, TimingRecorder};

pub(crate) const DEFAULT_SIMILARITY_THRESHOLD: f32 = 0.25;

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct SearchResult {
    pub(crate) score: f32,
    pub(crate) path: PathBuf,
}

pub(crate) struct Ranker<'a> {
    query_embedding: &'a NormalizedEmbedding,
    top: usize,
    threshold: f32,
}

impl<'a> Ranker<'a> {
    pub(crate) fn new(
        query_embedding: &'a NormalizedEmbedding,
        top: usize,
        threshold: f32,
    ) -> Self {
        Self {
            query_embedding,
            top,
            threshold,
        }
    }

    /// Scores every image, filters weak matches, and returns a deterministic top-K ranking.
    ///
    /// Score ties are resolved by path so repeated searches over the same index produce stable
    /// output.
    pub(crate) fn rank(
        &self,
        images: Vec<ImageRecord>,
        timing: &mut TimingRecorder,
    ) -> Vec<SearchResult> {
        let scoring_started = timing.start();
        let mut results = Vec::with_capacity(images.len().min(self.top));
        for image in images {
            let score = cosine_similarity(self.query_embedding, &image.embedding);
            if score >= self.threshold {
                results.push(SearchResult {
                    score,
                    path: image.path,
                });
            }
        }
        timing.record(Phase::SimilarityScoring, scoring_started);

        let selection_started = timing.start();
        results.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.path.cmp(&right.path))
        });
        results.truncate(self.top);
        timing.record(Phase::TopKSelection, selection_started);
        results
    }
}

/// Computes cosine similarity as a dot product under the embedding layer's unit-norm invariant.
fn cosine_similarity(left: &NormalizedEmbedding, right: &NormalizedEmbedding) -> f32 {
    left.dot(right)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedding::EMBEDDING_DIM;

    fn embedding(first: f32, second: f32) -> NormalizedEmbedding {
        let mut values = vec![0.0; EMBEDDING_DIM];
        values[0] = first;
        values[1] = second;
        NormalizedEmbedding::from_model_output(values).unwrap()
    }

    #[test]
    fn cosine_similarity_is_dot_product() {
        let left = embedding(0.6, 0.8);
        let right = embedding(0.8, 0.6);

        assert_eq!(cosine_similarity(&left, &right), 0.96000004);
    }

    #[test]
    fn ranking_filters_sorts_and_truncates() {
        let query = embedding(1.0, 0.0);
        let ranker = Ranker::new(&query, 2, 0.5);
        let mut timing = TimingRecorder::disabled(crate::model::timing_metadata());
        let results = ranker.rank(
            vec![
                ImageRecord {
                    path: PathBuf::from("b.jpg"),
                    embedding: embedding(0.7, (1.0_f32 - 0.7_f32.powi(2)).sqrt()),
                },
                ImageRecord {
                    path: PathBuf::from("a.jpg"),
                    embedding: embedding(0.9, (1.0_f32 - 0.9_f32.powi(2)).sqrt()),
                },
                ImageRecord {
                    path: PathBuf::from("c.jpg"),
                    embedding: embedding(0.4, (1.0_f32 - 0.4_f32.powi(2)).sqrt()),
                },
            ],
            &mut timing,
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].path, PathBuf::from("a.jpg"));
        assert_eq!(results[1].path, PathBuf::from("b.jpg"));
    }
}
