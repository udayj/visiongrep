use std::cmp::Ordering;
use std::collections::BinaryHeap;
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

#[derive(Debug)]
struct HeapEntry(SearchResult);

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.score.to_bits() == other.0.score.to_bits() && self.0.path == other.0.path
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    /// Reverses score order so the worst retained result is the max-heap root.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .0
            .score
            .total_cmp(&self.0.score)
            .then_with(|| self.0.path.cmp(&other.0.path))
    }
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
        if self.top == 0 {
            return Vec::new();
        }

        let mut heap = BinaryHeap::with_capacity(self.top);
        for image in images {
            let scoring_started = timing.start();
            let score = cosine_similarity(self.query_embedding, &image.embedding);
            timing.record(Phase::SimilarityScoring, scoring_started);
            if score < self.threshold {
                continue;
            }

            let selection_started = timing.start();
            let candidate = SearchResult {
                score,
                path: image.path,
            };
            if heap.len() < self.top {
                heap.push(HeapEntry(candidate));
            } else {
                let should_replace = heap
                    .peek()
                    .is_some_and(|worst| final_rank_order(&candidate, &worst.0).is_lt());
                if should_replace {
                    heap.pop();
                    heap.push(HeapEntry(candidate));
                }
            }
            timing.record(Phase::TopKSelection, selection_started);
        }

        let selection_started = timing.start();
        let mut results = heap.into_iter().map(|entry| entry.0).collect::<Vec<_>>();
        results.sort_by(final_rank_order);
        timing.record(Phase::TopKSelection, selection_started);
        results
    }
}

/// Computes cosine similarity as a dot product under the embedding layer's unit-norm invariant.
fn cosine_similarity(left: &NormalizedEmbedding, right: &NormalizedEmbedding) -> f32 {
    left.dot(right)
}

fn final_rank_order(left: &SearchResult, right: &SearchResult) -> Ordering {
    right
        .score
        .total_cmp(&left.score)
        .then_with(|| left.path.cmp(&right.path))
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

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

    #[test]
    fn bounded_heap_preserves_path_ascending_tie_breaking() {
        let query = embedding(1.0, 0.0);
        let tied = embedding(1.0, 0.0);
        let mut timing = TimingRecorder::disabled(crate::model::timing_metadata());

        let results = Ranker::new(&query, 2, -1.0).rank(
            vec![
                ImageRecord {
                    path: PathBuf::from("c.jpg"),
                    embedding: tied.clone(),
                },
                ImageRecord {
                    path: PathBuf::from("a.jpg"),
                    embedding: tied.clone(),
                },
                ImageRecord {
                    path: PathBuf::from("b.jpg"),
                    embedding: tied,
                },
            ],
            &mut timing,
        );

        assert_eq!(results[0].path, PathBuf::from("a.jpg"));
        assert_eq!(results[1].path, PathBuf::from("b.jpg"));
    }

    #[test]
    fn threshold_is_inclusive_at_the_boundary() {
        let query = embedding(1.0, 0.0);
        let image = ImageRecord {
            path: PathBuf::from("boundary.jpg"),
            embedding: embedding(0.5, 3.0_f32.sqrt() / 2.0),
        };
        let mut timing = TimingRecorder::disabled(crate::model::timing_metadata());

        let accepted = Ranker::new(&query, 1, 0.5).rank(vec![image.clone()], &mut timing);
        let rejected = Ranker::new(&query, 1, 0.500_001).rank(vec![image], &mut timing);

        assert_eq!(accepted.len(), 1);
        assert!(rejected.is_empty());
    }

    fn rank_full_sort(
        query: &NormalizedEmbedding,
        images: &[ImageRecord],
        top: usize,
        threshold: f32,
    ) -> Vec<SearchResult> {
        let mut results = images
            .iter()
            .filter_map(|image| {
                let score = cosine_similarity(query, &image.embedding);
                (score >= threshold).then(|| SearchResult {
                    score,
                    path: image.path.clone(),
                })
            })
            .collect::<Vec<_>>();
        results.sort_by(final_rank_order);
        results.truncate(top);
        results
    }

    fn rank_bounded_heap(
        query: &NormalizedEmbedding,
        images: &[ImageRecord],
        top: usize,
        threshold: f32,
    ) -> Vec<SearchResult> {
        let mut heap = BinaryHeap::with_capacity(top);
        for image in images {
            let score = cosine_similarity(query, &image.embedding);
            if score < threshold {
                continue;
            }
            let candidate = SearchResult {
                score,
                path: image.path.clone(),
            };
            if heap.len() < top {
                heap.push(HeapEntry(candidate));
                continue;
            }
            let should_replace = heap
                .peek()
                .is_some_and(|worst| final_rank_order(&candidate, &worst.0).is_lt());
            if should_replace {
                heap.pop();
                heap.push(HeapEntry(candidate));
            }
        }
        let mut results = heap.into_iter().map(|entry| entry.0).collect::<Vec<_>>();
        results.sort_by(final_rank_order);
        results
    }

    fn synthetic_images(count: usize) -> Vec<ImageRecord> {
        (0..count)
            .map(|index| {
                let bucket = i32::try_from(index % 2_001).unwrap() - 1_000;
                let first = f32::from(i16::try_from(bucket).unwrap()) / 1_000.0;
                let second = (1.0 - first * first).max(0.0).sqrt();
                ImageRecord {
                    path: PathBuf::from(format!("image-{index:06}.jpg")),
                    embedding: embedding(first, second),
                }
            })
            .collect()
    }

    fn percentile(durations: &mut [Duration], percentile: usize) -> Duration {
        durations.sort_unstable();
        let position = durations
            .len()
            .saturating_mul(percentile)
            .div_ceil(100)
            .saturating_sub(1)
            .min(durations.len() - 1);
        durations[position]
    }

    /// Run with `cargo test ranking_selection_matrix -- --ignored --nocapture`.
    /// Set `VISIONGREP_RANKING_SAMPLES=21` when a meaningful p95 is required.
    #[test]
    #[ignore = "measurement harness for deterministic ranking selection"]
    fn ranking_selection_matrix() {
        let samples = std::env::var("VISIONGREP_RANKING_SAMPLES")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(5)
            .max(1);
        let query = embedding(1.0, 0.0);
        for corpus_size in [10_000, 100_000] {
            let images = synthetic_images(corpus_size);
            for threshold in [DEFAULT_SIMILARITY_THRESHOLD, -1.0] {
                for top in [5, 10, 100] {
                    let expected = rank_full_sort(&query, &images, top, threshold);
                    let actual = rank_bounded_heap(&query, &images, top, threshold);
                    assert_eq!(actual.len(), expected.len());
                    for (actual, expected) in actual.iter().zip(&expected) {
                        assert_eq!(actual.path, expected.path);
                        assert_eq!(actual.score.to_bits(), expected.score.to_bits());
                    }

                    let mut full_sort = Vec::with_capacity(samples);
                    let mut bounded_heap = Vec::with_capacity(samples);
                    for _ in 0..samples {
                        let started = Instant::now();
                        std::hint::black_box(rank_full_sort(&query, &images, top, threshold));
                        full_sort.push(started.elapsed());

                        let started = Instant::now();
                        std::hint::black_box(rank_bounded_heap(&query, &images, top, threshold));
                        bounded_heap.push(started.elapsed());
                    }
                    let full_sort_median = percentile(&mut full_sort.clone(), 50);
                    let heap_median = percentile(&mut bounded_heap.clone(), 50);
                    let report = serde_json::json!({
                        "schema_version": 1,
                        "corpus_size": corpus_size,
                        "threshold": threshold,
                        "top": top,
                        "samples": samples,
                        "full_sort_median_ms": full_sort_median.as_secs_f64() * 1_000.0,
                        "bounded_heap_median_ms": heap_median.as_secs_f64() * 1_000.0,
                        "median_speedup": full_sort_median.as_secs_f64() / heap_median.as_secs_f64(),
                        "full_sort_p95_ms": (samples >= 20).then(|| percentile(&mut full_sort, 95).as_secs_f64() * 1_000.0),
                        "bounded_heap_p95_ms": (samples >= 20).then(|| percentile(&mut bounded_heap, 95).as_secs_f64() * 1_000.0),
                    });
                    println!("{report}");
                }
            }
        }
    }
}
