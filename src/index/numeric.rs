//! Numeric field index: cardinality estimation for Range predicates plus
//! bounded counting (DECISIONS #31).
//!
//! Two components, two distinct jobs:
//! - **Equal-width histogram (64 buckets)**: the ŝ estimate for the large arm.
//!   The estimate is NOT A SINGLE NUMBER but an [lower, upper] interval: the
//!   fully contained buckets give the lower bound, adding the boundary buckets
//!   gives the upper one. No within-bucket uniformity is ever assumed —
//!   instead of an assumption, the uncertainty is carried explicitly and the
//!   planner picks the conservative side.
//! - **Value-ordered BTreeMap**: the small-match decision is NOT made from an
//!   estimate but via `enumerate_up_to(limit)` — a real count up to limit+1
//!   elements. The decision is exact, and the matching ids fall out for free
//!   for the scan arm. (In a histogram-only design the small-arm decision
//!   would rest on an estimate; picking the wrong arm at the boundary is
//!   exactly the pathology we want to avoid.)
//!
//! Maintenance: insert/remove is O(log distinct). On an out-of-range insert
//! the histogram is widened by 12.5% and rebuilt from the sorted map
//! (O(distinct)) — the margin amortizes constant rebuilding under a
//! monotonically increasing value stream.

use crate::meta::ordered_bits;
use crate::types::VectorId;
use std::collections::BTreeMap;

const BUCKETS: usize = 64;

#[derive(Debug)]
pub struct NumericFieldIndex {
    /// value(bits) → the ids at that value.
    sorted: BTreeMap<u64, Vec<VectorId>>,
    total: usize,
    lo: f64,
    hi: f64,
    hist: Vec<usize>,
}

impl Default for NumericFieldIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl NumericFieldIndex {
    pub fn new() -> Self {
        Self {
            sorted: BTreeMap::new(),
            total: 0,
            lo: f64::INFINITY,
            hi: f64::NEG_INFINITY,
            hist: vec![0; BUCKETS],
        }
    }

    #[inline]
    fn bucket(&self, v: f64) -> usize {
        if self.hi <= self.lo {
            return 0;
        }
        (((v - self.lo) / (self.hi - self.lo) * BUCKETS as f64) as usize).min(BUCKETS - 1)
    }

    pub fn insert(&mut self, v: f64, id: VectorId) {
        self.sorted.entry(ordered_bits(v)).or_default().push(id);
        self.total += 1;
        if v < self.lo || v > self.hi {
            // Widening margin: rather than fitting exactly to the boundary it
            // overshoots by 12.5% — this avoids a rebuild on every insert under
            // a monotone stream.
            let span = (self.hi - self.lo).max(v.abs().max(1.0) * 0.01);
            self.lo = self.lo.min(v - span * 0.125);
            self.hi = self.hi.max(v + span * 0.125);
            self.rebuild_hist();
        } else {
            let b = self.bucket(v);
            self.hist[b] += 1;
        }
    }

    pub fn remove(&mut self, v: f64, id: VectorId) {
        if let Some(ids) = self.sorted.get_mut(&ordered_bits(v)) {
            if let Some(pos) = ids.iter().position(|&x| x == id) {
                ids.swap_remove(pos);
                if ids.is_empty() {
                    self.sorted.remove(&ordered_bits(v));
                }
                self.total -= 1;
                // The range only ever grows; v is always within [lo, hi].
                let b = self.bucket(v);
                self.hist[b] = self.hist[b].saturating_sub(1);
            }
        }
    }

    fn rebuild_hist(&mut self) {
        self.hist.iter_mut().for_each(|c| *c = 0);
        // The alternative to converting bits → f64 would be storing the value
        // alongside for bucket assignment. Since bits are monotone, computing
        // the bucket in bits space MIGHT SEEM equivalent — but equal width is
        // defined in f64 space. So decode the value back from bits:
        // ordered_bits is invertible.
        for (&bits, ids) in &self.sorted {
            let v = Self::bits_to_f64(bits);
            let b = self.bucket(v);
            self.hist[b] += ids.len();
        }
    }

    fn bits_to_f64(bits: u64) -> f64 {
        let b = if bits >> 63 == 1 {
            bits & !(1 << 63)
        } else {
            !bits
        };
        f64::from_bits(b)
    }

    /// The cardinality INTERVAL for [qlo, qhi] (closed range): (lower, upper).
    /// Lower = sum of the fully contained buckets; upper = plus the boundary
    /// buckets.
    pub fn estimate(&self, qlo: f64, qhi: f64) -> (usize, usize) {
        if self.total == 0 || qhi < self.lo || qlo > self.hi {
            return (0, 0);
        }
        let qlo = qlo.max(self.lo);
        let qhi = qhi.min(self.hi);
        let b_lo = self.bucket(qlo);
        let b_hi = self.bucket(qhi);
        if b_lo == b_hi {
            return (0, self.hist[b_lo]);
        }
        let lower: usize = self.hist[b_lo + 1..b_hi].iter().sum();
        (lower, lower + self.hist[b_lo] + self.hist[b_hi])
    }

    /// Collects the ids within [qlo, qhi] by counting; returns None if the
    /// count exceeds `limit`. This is the exact path for the small-match
    /// decision: the histogram is only used when it says "large"; a "small"
    /// verdict is always a real count.
    pub fn enumerate_up_to(&self, qlo: f64, qhi: f64, limit: usize) -> Option<Vec<VectorId>> {
        let mut out = Vec::new();
        for ids in self
            .sorted
            .range(ordered_bits(qlo)..=ordered_bits(qhi))
            .map(|(_, ids)| ids)
        {
            out.extend_from_slice(ids);
            if out.len() > limit {
                return None;
            }
        }
        Some(out)
    }

    /// Unbounded full enumeration (for the post-filter fallback).
    pub fn enumerate_all(&self, qlo: f64, qhi: f64) -> Vec<VectorId> {
        self.sorted
            .range(ordered_bits(qlo)..=ordered_bits(qhi))
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect()
    }

    /// Computed memory cost (bytes). BTreeMap node overhead is modelled with
    /// a rough per-entry constant; not exact but consistent — which suffices
    /// because the 9c threshold is proportional (DECISIONS #40).
    pub fn memory_bytes(&self) -> usize {
        const BTREE_ENTRY_OVERHEAD: usize = 48; // node share + key
        let vec_headers = self.sorted.len() * std::mem::size_of::<Vec<VectorId>>();
        let ids = self.total * std::mem::size_of::<VectorId>();
        self.sorted.len() * BTREE_ENTRY_OVERHEAD + vec_headers + ids + self.hist.len() * 8
    }

    pub fn len(&self) -> usize {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_bits_monotonic() {
        let vals = [-1e9, -3.5, -0.0, 0.0, 1e-9, 2.0, 7.5, 1e12];
        for w in vals.windows(2) {
            assert!(
                ordered_bits(w[0]) <= ordered_bits(w[1]),
                "{} {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn estimate_brackets_truth() {
        let mut idx = NumericFieldIndex::new();
        for i in 0..1_000 {
            idx.insert(i as f64, VectorId(i));
        }
        for (lo, hi) in [(0.0, 99.0), (250.0, 749.0), (990.0, 2000.0), (-50.0, 10.0)] {
            let truth = (0..1_000)
                .filter(|&i| (i as f64) >= lo && (i as f64) <= hi)
                .count();
            let (l, u) = idx.estimate(lo, hi);
            assert!(
                l <= truth && truth <= u,
                "[{lo},{hi}]: {l} ≤ {truth} ≤ {u} does not hold"
            );
        }
    }

    #[test]
    fn enumerate_exact_and_limited() {
        let mut idx = NumericFieldIndex::new();
        for i in 0..100 {
            idx.insert((i % 10) as f64, VectorId(i)); // repeated values
        }
        let ids = idx.enumerate_up_to(2.0, 3.0, 100).unwrap();
        assert_eq!(ids.len(), 20);
        assert!(idx.enumerate_up_to(0.0, 9.0, 50).is_none()); // 100 > 50
        assert_eq!(idx.enumerate_all(0.0, 9.0).len(), 100);
    }

    #[test]
    fn remove_keeps_hist_consistent() {
        let mut idx = NumericFieldIndex::new();
        for i in 0..500 {
            idx.insert(i as f64, VectorId(i));
        }
        for i in (0..500).step_by(3) {
            idx.remove(i as f64, VectorId(i));
        }
        let truth = (0..500).filter(|i| i % 3 != 0).count();
        assert_eq!(idx.len(), truth);
        let (l, u) = idx.estimate(f64::NEG_INFINITY, f64::INFINITY);
        assert!(l <= truth && truth <= u);
        // the histogram sum must stay equal to total
        let hist_sum: usize = idx.hist.iter().sum();
        assert_eq!(hist_sum, truth);
    }

    #[test]
    fn monotonic_inserts_amortized_widening() {
        // Thanks to the widening margin a monotone stream must not blow up
        // (behavioural test: correctness only — the histogram sum is preserved).
        let mut idx = NumericFieldIndex::new();
        for i in 0..10_000 {
            idx.insert(i as f64, VectorId(i));
        }
        let hist_sum: usize = idx.hist.iter().sum();
        assert_eq!(hist_sum, 10_000);
        let (l, u) = idx.estimate(1000.0, 1999.0);
        assert!(l <= 1000 && 1000 <= u);
    }
}
