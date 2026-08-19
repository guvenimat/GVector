//! Measurement infrastructure: exact ground-truth generation, recall@k, and
//! latency percentiles.
//!
//! `exact_top_k` here is deliberately a plain linear scan, independent of the
//! indexes: it is the reference against which index correctness is checked at
//! every phase. (The brute-force index of phase 1 uses it, but behind the
//! trait.)

use crate::distance::Metric;
use crate::types::{SearchResult, VectorId};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

/// Returns the exact top-k of a query over the given base set (ascending
/// distance). Cosine contract: the caller normalizes both the base and the
/// query beforehand.
pub fn exact_top_k(
    base: &[Vec<f32>],
    query: &[f32],
    k: usize,
    metric: Metric,
) -> Vec<SearchResult> {
    // In a max-heap the WORST candidate sits on top; when a better candidate
    // arrives the top is popped. This keeps memory at O(k) and time at
    // O(n log k).
    let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(k + 1);
    for (i, v) in base.iter().enumerate() {
        let d = metric.distance(query, v);
        let cand = SearchResult::new(VectorId(i as u64), d);
        if heap.len() < k {
            heap.push(cand);
        } else if let Some(worst) = heap.peek() {
            if cand < *worst {
                heap.pop();
                heap.push(cand);
            }
        }
    }
    let mut out = heap.into_vec();
    out.sort();
    out
}

/// Generates the ground truth for all queries in parallel.
pub fn ground_truth(
    base: &[Vec<f32>],
    queries: &[Vec<f32>],
    k: usize,
    metric: Metric,
) -> Vec<Vec<VectorId>> {
    queries
        .par_iter()
        .map(|q| {
            exact_top_k(base, q, k, metric)
                .iter()
                .map(|r| r.id)
                .collect()
        })
        .collect()
}

/// recall@k: the intersection ratio of the results with the ground truth,
/// averaged over queries.
///
/// The denominator is `min(k, gt.len())` — on a small index (fewer elements
/// than k) an index that returns the correct results must score 1.0 rather
/// than being penalized.
pub fn recall_at_k(results: &[Vec<VectorId>], truth: &[Vec<VectorId>], k: usize) -> f64 {
    assert_eq!(results.len(), truth.len(), "query counts must match");
    if results.is_empty() {
        return 1.0; // perfect by definition on an empty query set
    }
    let total: f64 = results
        .iter()
        .zip(truth.iter())
        .map(|(res, gt)| {
            let denom = k.min(gt.len());
            if denom == 0 {
                return 1.0; // empty index: there is no correct result to return
            }
            let gt_set: std::collections::HashSet<_> = gt.iter().take(k).collect();
            let hit = res.iter().take(k).filter(|id| gt_set.contains(id)).count();
            hit as f64 / denom as f64
        })
        .sum();
    total / results.len() as f64
}

/// Summary of a latency measurement.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub p50: Duration,
    pub p99: Duration,
    pub mean: Duration,
    pub samples: usize,
}

/// Times a closure once per query and computes percentiles.
/// Criterion is for micro-benchmarks; this one is for end-to-end reports.
pub fn measure_latency<F: FnMut(&[f32])>(queries: &[Vec<f32>], mut f: F) -> LatencyStats {
    assert!(!queries.is_empty(), "latency needs at least one query");
    let mut times: Vec<Duration> = Vec::with_capacity(queries.len());
    for q in queries {
        let t = Instant::now();
        f(q);
        times.push(t.elapsed());
    }
    times.sort();
    let pct = |p: f64| times[((times.len() as f64 * p).ceil() as usize - 1).min(times.len() - 1)];
    let mean = times.iter().sum::<Duration>() / times.len() as u32;
    LatencyStats {
        p50: pct(0.50),
        p99: pct(0.99),
        mean,
        samples: times.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_top_k_orders_by_distance() {
        let base = vec![vec![0.0, 10.0], vec![0.0, 1.0], vec![0.0, 5.0]];
        let res = exact_top_k(&base, &[0.0, 0.0], 2, Metric::L2);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].id, VectorId(1));
        assert_eq!(res[1].id, VectorId(2));
        assert!(res[0].distance <= res[1].distance);
    }

    #[test]
    fn exact_top_k_k_larger_than_base() {
        let base = vec![vec![1.0], vec![2.0]];
        let res = exact_top_k(&base, &[0.0], 10, Metric::L2);
        assert_eq!(res.len(), 2);
    }

    #[test]
    fn exact_top_k_empty_base() {
        let res = exact_top_k(&[], &[0.0], 5, Metric::L2);
        assert!(res.is_empty());
    }

    #[test]
    fn exact_top_k_single_element() {
        let base = vec![vec![3.0]];
        let res = exact_top_k(&base, &[0.0], 1, Metric::L2);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId(0));
    }

    #[test]
    fn exact_top_k_duplicate_vectors_both_returned() {
        let base = vec![vec![1.0, 1.0], vec![1.0, 1.0], vec![9.0, 9.0]];
        let res = exact_top_k(&base, &[1.0, 1.0], 2, Metric::L2);
        let ids: Vec<_> = res.iter().map(|r| r.id).collect();
        assert!(ids.contains(&VectorId(0)) && ids.contains(&VectorId(1)));
    }

    #[test]
    fn recall_perfect_and_partial() {
        let truth = vec![vec![VectorId(1), VectorId(2), VectorId(3)]];
        assert_eq!(
            recall_at_k(&[vec![VectorId(1), VectorId(2), VectorId(3)]], &truth, 3),
            1.0
        );
        let r = recall_at_k(&[vec![VectorId(1), VectorId(9), VectorId(8)]], &truth, 3);
        assert!((r - 1.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn recall_small_index_not_penalized() {
        // with 2 elements in the index and k=10 requested, if both are correct
        // the recall is 1.0
        let truth = vec![vec![VectorId(0), VectorId(1)]];
        let res = vec![vec![VectorId(0), VectorId(1)]];
        assert_eq!(recall_at_k(&res, &truth, 10), 1.0);
    }

    #[test]
    fn latency_stats_sane() {
        let queries = vec![vec![0.0f32; 4]; 20];
        let stats = measure_latency(&queries, |q| {
            std::hint::black_box(crate::distance::dot(q, q));
        });
        assert_eq!(stats.samples, 20);
        assert!(stats.p99 >= stats.p50);
    }
}
