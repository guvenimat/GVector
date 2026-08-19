//! Brute-force (linear scan) index.
//!
//! The correctness reference for the whole project: every new index (HNSW and
//! so on) is checked against it. That is why there is nothing "clever" here —
//! readability and correctness come before everything else.

use crate::distance::{normalize, Metric};
use crate::index::{IndexError, VectorIndex};
use crate::types::{SearchResult, VectorId};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::collections::HashMap;

/// The element count above which the search is parallelized with rayon.
/// On small indexes the cost of distributing work across threads exceeds the
/// scan itself; the threshold was chosen as roughly "what one core scans in
/// ~1 ms".
const PARALLEL_THRESHOLD: usize = 20_000;

pub struct BruteForceIndex {
    metric: Metric,
    dim: usize,
    /// Vector data in one contiguous block (row-major). A flat block rather
    /// than `Vec<Vec<f32>>`: cache-friendly scanning, and no 24-byte Vec
    /// header per vector.
    data: Vec<f32>,
    /// Slot -> external id. The i-th vector in `data` belongs to `ids[i]`.
    ids: Vec<VectorId>,
    /// External id -> slot, so that delete and duplicate checks are O(1).
    slot_of: HashMap<VectorId, usize>,
}

impl BruteForceIndex {
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            metric,
            dim,
            data: Vec::new(),
            ids: Vec::new(),
            slot_of: HashMap::new(),
        }
    }

    /// An empty index with capacity allocated UP FRONT (#61).
    ///
    /// For the write buffer: the buffer grows up to the sealing threshold, and
    /// near that threshold `Vec`'s incremental growth turns into a ~64 MB
    /// realloc + memcpy — a single insert taking milliseconds is the symptom.
    /// Capacity is allocated at the EXACT size (no growth margin): the buffer
    /// is sealed as soon as it reaches the threshold, so extra margin would not
    /// prevent a realloc, it would only inflate the memory peak.
    ///
    /// ALL THREE structures are pre-allocated, not just the vector block:
    /// `ids` and `slot_of` grow with the record count too, and rehashing the
    /// `HashMap` produces a spike of the same class.
    pub fn with_capacity(dim: usize, metric: Metric, records: usize) -> Self {
        Self {
            metric,
            dim,
            data: Vec::with_capacity(records * dim),
            ids: Vec::with_capacity(records),
            slot_of: HashMap::with_capacity(records),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The vector slice at slot i.
    #[inline]
    fn vector_at(&self, slot: usize) -> &[f32] {
        &self.data[slot * self.dim..(slot + 1) * self.dim]
    }

    /// Scans the given slot range and returns a local top-k heap.
    /// A separate function so that the parallel and serial paths share the same
    /// core.
    fn scan_range(
        &self,
        query: &[f32],
        k: usize,
        range: std::ops::Range<usize>,
    ) -> BinaryHeap<SearchResult> {
        let mut heap = BinaryHeap::with_capacity(k + 1);
        for slot in range {
            let d = self.metric.distance(query, self.vector_at(slot));
            let cand = SearchResult::new(self.ids[slot], d);
            if heap.len() < k {
                heap.push(cand);
            } else if let Some(worst) = heap.peek() {
                if cand < *worst {
                    heap.pop();
                    heap.push(cand);
                }
            }
        }
        heap
    }

    /// Filtered linear scan: top-k among those passing `allow(id)`.
    /// Filtering is free in brute force — non-matches are skipped during the
    /// scan, which makes this the reference source of truth.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        allow: &dyn Fn(VectorId) -> bool,
    ) -> Vec<SearchResult> {
        if k == 0 {
            return Vec::new();
        }
        let normalized_query;
        let query: &[f32] = if self.metric.requires_normalization() {
            normalized_query = crate::distance::normalized(query);
            &normalized_query
        } else {
            query
        };
        let mut all: Vec<SearchResult> = (0..self.ids.len())
            .filter(|&s| allow(self.ids[s]))
            .map(|s| SearchResult::new(self.ids[s], self.metric.distance(query, self.vector_at(s))))
            .collect();
        all.sort();
        all.truncate(k);
        all
    }

    /// Is this id present in the index?
    pub fn contains(&self, id: VectorId) -> bool {
        self.slot_of.contains_key(&id)
    }

    /// The vector for an id (normalized, under cosine).
    pub fn vector_of(&self, id: VectorId) -> Option<&[f32]> {
        self.slot_of.get(&id).map(|&s| self.vector_at(s))
    }

    /// (id, vector) pairs — used by segment sealing when draining the buffer.
    /// Under cosine the returned vectors are already normalized (idempotent).
    pub fn entries(&self) -> impl Iterator<Item = (VectorId, &[f32])> {
        self.ids
            .iter()
            .enumerate()
            .map(|(slot, &id)| (id, self.vector_at(slot)))
    }

    /// Approximate memory usage of the index (bytes) — for the BENCHMARKS report.
    pub fn memory_bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<f32>()
            + self.ids.capacity() * std::mem::size_of::<VectorId>()
            // Rough estimate per HashMap entry: (key + value) * load-factor slack
            + self.slot_of.capacity() * (std::mem::size_of::<(VectorId, usize)>() + 8)
    }
}

impl VectorIndex for BruteForceIndex {
    fn insert(&mut self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        if vector.len() != self.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if self.slot_of.contains_key(&id) {
            return Err(IndexError::DuplicateId(id));
        }
        let slot = self.ids.len();
        self.data.extend_from_slice(vector);
        // Cosine policy: normalize at insert time, in place (see DECISIONS.md)
        if self.metric.requires_normalization() {
            let start = slot * self.dim;
            normalize(&mut self.data[start..start + self.dim]);
        }
        self.ids.push(id);
        self.slot_of.insert(id, slot);
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if k == 0 || self.ids.is_empty() {
            return Vec::new();
        }
        // The query side of the cosine contract: normalize once per search.
        let normalized_query;
        let query: &[f32] = if self.metric.requires_normalization() {
            normalized_query = crate::distance::normalized(query);
            &normalized_query
        } else {
            query
        };

        let n = self.ids.len();
        let mut heap = if n < PARALLEL_THRESHOLD {
            self.scan_range(query, k, 0..n)
        } else {
            // Each rayon chunk produces its own local top-k, then the k-sized
            // heaps are merged: no shared mutable state, no locks.
            let chunk = n.div_ceil(rayon::current_num_threads().max(1));
            (0..n)
                .into_par_iter()
                .step_by(chunk)
                .map(|start| self.scan_range(query, k, start..(start + chunk).min(n)))
                .reduce(BinaryHeap::new, |mut a, b| {
                    for cand in b {
                        if a.len() < k {
                            a.push(cand);
                        } else if let Some(worst) = a.peek() {
                            if cand < *worst {
                                a.pop();
                                a.push(cand);
                            }
                        }
                    }
                    a
                })
        };
        let mut out = Vec::with_capacity(heap.len());
        while let Some(r) = heap.pop() {
            out.push(r);
        }
        out.reverse(); // the heap drains worst-to-best; we want ascending distance
        out
    }

    fn delete(&mut self, id: VectorId) -> Result<(), IndexError> {
        let slot = self.slot_of.remove(&id).ok_or(IndexError::NotFound(id))?;
        let last = self.ids.len() - 1;
        // swap-remove: move the last vector into the deleted slot, O(1)
        // deletion. Safe here because slot order carries no meaning in brute
        // force.
        if slot != last {
            let (head, tail) = self.data.split_at_mut(last * self.dim);
            head[slot * self.dim..(slot + 1) * self.dim].copy_from_slice(&tail[..self.dim]);
            let moved_id = self.ids[last];
            self.ids[slot] = moved_id;
            self.slot_of.insert(moved_id, slot);
        }
        self.ids.pop();
        self.data.truncate(last * self.dim);
        Ok(())
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::random_vectors;
    use crate::eval::exact_top_k;

    fn build(vecs: &[Vec<f32>], metric: Metric) -> BruteForceIndex {
        let mut idx = BruteForceIndex::new(vecs[0].len(), metric);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        idx
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = BruteForceIndex::new(4, Metric::L2);
        assert!(idx.is_empty());
        assert!(idx.search(&[0.0; 4], 5).is_empty());
    }

    #[test]
    fn k_zero_returns_empty() {
        let idx = build(&random_vectors(10, 4, 42), Metric::L2);
        assert!(idx.search(&[0.0; 4], 0).is_empty());
    }

    #[test]
    fn k_larger_than_len_returns_all() {
        let idx = build(&random_vectors(3, 4, 42), Metric::L2);
        assert_eq!(idx.search(&[0.0; 4], 10).len(), 3);
    }

    #[test]
    fn single_element() {
        let idx = build(&[vec![1.0, 2.0]], Metric::L2);
        let res = idx.search(&[0.0, 0.0], 1);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId(0));
    }

    #[test]
    fn duplicate_vectors_distinct_ids_both_found() {
        let idx = build(
            &[vec![1.0, 1.0], vec![1.0, 1.0], vec![5.0, 5.0]],
            Metric::L2,
        );
        let ids: Vec<_> = idx.search(&[1.0, 1.0], 2).iter().map(|r| r.id).collect();
        assert!(ids.contains(&VectorId(0)) && ids.contains(&VectorId(1)));
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut idx = BruteForceIndex::new(2, Metric::L2);
        idx.insert(VectorId(7), &[1.0, 2.0]).unwrap();
        assert_eq!(
            idx.insert(VectorId(7), &[3.0, 4.0]),
            Err(IndexError::DuplicateId(VectorId(7)))
        );
    }

    #[test]
    fn dimension_mismatch_rejected() {
        let mut idx = BruteForceIndex::new(3, Metric::L2);
        assert!(matches!(
            idx.insert(VectorId(0), &[1.0, 2.0]),
            Err(IndexError::DimensionMismatch {
                expected: 3,
                got: 2
            })
        ));
    }

    #[test]
    fn zero_vector_cosine_no_nan() {
        let mut idx = BruteForceIndex::new(3, Metric::Cosine);
        idx.insert(VectorId(0), &[0.0, 0.0, 0.0]).unwrap();
        idx.insert(VectorId(1), &[1.0, 0.0, 0.0]).unwrap();
        let res = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|r| !r.distance.is_nan()));
        assert_eq!(res[0].id, VectorId(1)); // the true match, ahead of the zero vector
    }

    #[test]
    fn delete_then_search_and_reuse_id() {
        let mut idx = build(&random_vectors(10, 4, 42), Metric::L2);
        idx.delete(VectorId(3)).unwrap();
        assert_eq!(idx.len(), 9);
        assert!(idx
            .search(&[0.0; 4], 10)
            .iter()
            .all(|r| r.id != VectorId(3)));
        assert_eq!(
            idx.delete(VectorId(3)),
            Err(IndexError::NotFound(VectorId(3)))
        );
        // silinen id yeniden eklenebilmeli
        idx.insert(VectorId(3), &[9.0, 9.0, 9.0, 9.0]).unwrap();
        assert_eq!(idx.len(), 10);
    }

    /// The main correctness test: results identical to the reference exact_top_k.
    #[test]
    fn matches_reference_exact_scan_all_metrics() {
        let vecs = random_vectors(500, 16, 42);
        let queries = random_vectors(20, 16, 43);
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let idx = build(&vecs, metric);
            // apply the cosine contract by hand on the reference side
            let base: Vec<Vec<f32>> = if metric.requires_normalization() {
                vecs.iter()
                    .map(|v| crate::distance::normalized(v))
                    .collect()
            } else {
                vecs.clone()
            };
            for q in &queries {
                let qn = if metric.requires_normalization() {
                    crate::distance::normalized(q)
                } else {
                    q.clone()
                };
                let expected = exact_top_k(&base, &qn, 10, metric);
                let got = idx.search(q, 10);
                let exp_ids: Vec<_> = expected.iter().map(|r| r.id).collect();
                let got_ids: Vec<_> = got.iter().map(|r| r.id).collect();
                assert_eq!(got_ids, exp_ids, "metric={metric:?}");
            }
        }
    }

    /// The parallel path (n > threshold) must give the same result as the serial one.
    #[test]
    fn parallel_path_matches_serial() {
        let n = PARALLEL_THRESHOLD + 5_000;
        let vecs = random_vectors(n, 8, 42);
        let idx = build(&vecs, Metric::L2);
        for q in random_vectors(5, 8, 43) {
            let expected = exact_top_k(&vecs, &q, 10, Metric::L2);
            let got = idx.search(&q, 10);
            assert_eq!(
                got.iter().map(|r| r.id).collect::<Vec<_>>(),
                expected.iter().map(|r| r.id).collect::<Vec<_>>()
            );
        }
    }
}
