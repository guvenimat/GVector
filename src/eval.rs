//! Ölçüm altyapısı: exact ground truth üretimi, recall@k, latency percentile'ları.
//!
//! Buradaki `exact_top_k` bilinçli olarak indekslerden bağımsız, sade bir
//! doğrusal taramadır: her aşamada indekslerin doğruluğunu sınayacak referans.
//! (Aşama 1'deki brute-force indeks bunu kullanacak ama trait arkasında.)

use crate::distance::Metric;
use crate::types::{SearchResult, VectorId};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

/// Verilen taban kümede query'nin exact top-k'sını döndürür (artan mesafe).
/// Cosine sözleşmesi: çağıran hem tabanı hem query'yi önceden normalize eder.
pub fn exact_top_k(
    base: &[Vec<f32>],
    query: &[f32],
    k: usize,
    metric: Metric,
) -> Vec<SearchResult> {
    // Max-heap'te en KÖTÜ aday tepede durur; daha iyi bir aday gelince
    // tepe atılır. Böylece bellek O(k), zaman O(n log k).
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

/// Tüm query'ler için ground truth'u paralel üretir.
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

/// recall@k: sonuçların ground truth ile kesişim oranı, query'ler üzerinden ortalama.
///
/// Payda `min(k, gt.len())` — küçük indekslerde (eleman sayısı < k) doğru
/// sonuç dönen bir indeks 1.0 almalı, cezalandırılmamalı.
pub fn recall_at_k(results: &[Vec<VectorId>], truth: &[Vec<VectorId>], k: usize) -> f64 {
    assert_eq!(results.len(), truth.len(), "query sayıları uyuşmalı");
    if results.is_empty() {
        return 1.0; // boş sorgu kümesinde tanım gereği kusursuz
    }
    let total: f64 = results
        .iter()
        .zip(truth.iter())
        .map(|(res, gt)| {
            let denom = k.min(gt.len());
            if denom == 0 {
                return 1.0; // boş indeks: dönecek doğru sonuç yok
            }
            let gt_set: std::collections::HashSet<_> = gt.iter().take(k).collect();
            let hit = res.iter().take(k).filter(|id| gt_set.contains(id)).count();
            hit as f64 / denom as f64
        })
        .sum();
    total / results.len() as f64
}

/// Latency ölçüm özeti.
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub p50: Duration,
    pub p99: Duration,
    pub mean: Duration,
    pub samples: usize,
}

/// Bir kapanışı her query için tek tek zamanlayıp percentile hesaplar.
/// Criterion micro-bench için; bu ise uçtan uca rapor içindir.
pub fn measure_latency<F: FnMut(&[f32])>(queries: &[Vec<f32>], mut f: F) -> LatencyStats {
    assert!(!queries.is_empty(), "latency için en az bir query gerekli");
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
        // indekste 2 eleman varken k=10 istenirse ve ikisi de doğruysa recall 1.0
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
