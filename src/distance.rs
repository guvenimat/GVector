//! Distance functions.
//!
//! Contract: every function returns "smaller = closer".
//! - L2: squared Euclidean distance (no sqrt — an unnecessary cost for
//!   ordering, since top-k ranking is invariant under monotone transforms).
//! - Dot: `-dot(a,b)` (negated so that a large similarity means a small
//!   distance).
//! - Cosine: by policy, vectors are normalized at INSERT/QUERY time and the
//!   cosine distance is computed as `-dot`. Normalization is paid once per
//!   vector; no norm is computed in the hot search loop. This policy is
//!   recorded in DECISIONS.md.

/// Supported metrics. Indexes take this as configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Squared L2. The caller takes the sqrt if the true distance is needed.
    L2,
    /// Negated inner product.
    Dot,
    /// Negated inner product over normalized vectors.
    /// An index built with this metric is responsible for calling `normalize`
    /// on insert and query.
    Cosine,
}

impl Metric {
    /// Distance between two vectors ("smaller = closer").
    ///
    /// For cosine the caller is assumed to honour the normalization contract;
    /// no norm is recomputed here (policy: normalize at insert time).
    #[inline]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "dimension mismatch");
        match self {
            Metric::L2 => l2_squared(a, b),
            Metric::Dot | Metric::Cosine => -dot(a, b),
        }
    }

    /// Does this metric require normalization at insert/query time?
    pub fn requires_normalization(&self) -> bool {
        matches!(self, Metric::Cosine)
    }
}

// SIMD note: because `map().sum()` must preserve float addition order, LLVM
// cannot vectorize the reduction (measured: 128d dot ~60 ns, even with
// target-cpu=native). An explicit f32x8 with two accumulators gives both
// 8-wide SIMD and instruction-level parallelism; the summation order changes,
// but a ~1 ulp difference is irrelevant for distance comparisons.
// The `wide` crate offers a safe API, so #![deny(unsafe_code)] is preserved.

use wide::f32x8;

#[inline]
fn as_f32x8(chunk: &[f32]) -> f32x8 {
    f32x8::from(<[f32; 8]>::try_from(chunk).expect("chunk of 8"))
}

/// Inner product (f32x8 SIMD, 16 elements per iteration).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc0 = f32x8::ZERO;
    let mut acc1 = f32x8::ZERO;
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (x, y) in (&mut ca).zip(&mut cb) {
        acc0 += as_f32x8(&x[..8]) * as_f32x8(&y[..8]);
        acc1 += as_f32x8(&x[8..]) * as_f32x8(&y[8..]);
    }
    let mut sum = (acc0 + acc1).reduce_add();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    sum
}

/// Squared L2 distance (f32x8 SIMD, 16 elements per iteration).
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    let mut acc0 = f32x8::ZERO;
    let mut acc1 = f32x8::ZERO;
    let mut ca = a.chunks_exact(16);
    let mut cb = b.chunks_exact(16);
    for (x, y) in (&mut ca).zip(&mut cb) {
        let d0 = as_f32x8(&x[..8]) - as_f32x8(&y[..8]);
        let d1 = as_f32x8(&x[8..]) - as_f32x8(&y[8..]);
        acc0 += d0 * d0;
        acc1 += d1 * d1;
    }
    let mut sum = (acc0 + acc1).reduce_add();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Scales a vector in place to unit norm.
///
/// Zero-vector edge case: if the norm is 0 the vector is left as is (rather
/// than producing 0/0 = NaN). The cosine similarity of the zero vector with
/// anything is taken to be 0, which is exactly what -dot yields.
pub fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Copying helper that normalizes (for the query path).
pub fn normalized(v: &[f32]) -> Vec<f32> {
    let mut out = v.to_vec();
    normalize(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-5
    }

    #[test]
    fn dot_basic() {
        assert!(approx(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0));
        assert!(approx(dot(&[0.0, 0.0], &[1.0, 1.0]), 0.0));
    }

    #[test]
    fn l2_basic() {
        assert!(approx(l2_squared(&[0.0, 0.0], &[3.0, 4.0]), 25.0));
        assert!(approx(l2_squared(&[1.0, 1.0], &[1.0, 1.0]), 0.0));
    }

    #[test]
    fn normalize_unit_norm() {
        let mut v = vec![3.0, 4.0];
        normalize(&mut v);
        assert!(approx(dot(&v, &v), 1.0));
        assert!(approx(v[0], 0.6));
        assert!(approx(v[1], 0.8));
    }

    #[test]
    fn normalize_zero_vector_stays_zero_no_nan() {
        let mut v = vec![0.0, 0.0, 0.0];
        normalize(&mut v);
        assert!(v.iter().all(|x| *x == 0.0));
        // the distance to a zero vector must not be NaN
        let d = Metric::Cosine.distance(&v, &[1.0, 0.0, 0.0]);
        assert!(!d.is_nan());
        assert!(approx(d, 0.0));
    }

    #[test]
    fn identical_vectors_are_closest_cosine() {
        let a = normalized(&[1.0, 2.0, 3.0]);
        let d_self = Metric::Cosine.distance(&a, &a);
        assert!(approx(d_self, -1.0)); // -cos(0) = -1, the smallest possible
    }

    #[test]
    fn metric_smaller_is_closer() {
        // a near pair must give a smaller distance than a far one — in every metric
        let q = [1.0, 0.0];
        let near = [0.9, 0.1];
        let far = [-1.0, 0.0];
        for m in [Metric::L2, Metric::Dot] {
            assert!(m.distance(&q, &near) < m.distance(&q, &far), "{m:?}");
        }
        let qn = normalized(&q);
        assert!(
            Metric::Cosine.distance(&qn, &normalized(&near))
                < Metric::Cosine.distance(&qn, &normalized(&far))
        );
    }

    proptest! {
        #[test]
        fn prop_l2_symmetric_nonnegative(
            a in proptest::collection::vec(-100.0f32..100.0, 8),
            b in proptest::collection::vec(-100.0f32..100.0, 8),
        ) {
            let d1 = l2_squared(&a, &b);
            let d2 = l2_squared(&b, &a);
            prop_assert!(d1 >= 0.0);
            prop_assert!((d1 - d2).abs() <= 1e-3 * d1.abs().max(1.0));
        }

        #[test]
        fn prop_normalized_has_unit_norm_or_zero(
            a in proptest::collection::vec(-100.0f32..100.0, 8),
        ) {
            let n = normalized(&a);
            let norm = dot(&n, &n).sqrt();
            prop_assert!(norm == 0.0 || (norm - 1.0).abs() < 1e-4);
        }

        #[test]
        fn prop_cosine_bounded(
            a in proptest::collection::vec(-100.0f32..100.0, 8),
            b in proptest::collection::vec(-100.0f32..100.0, 8),
        ) {
            let d = Metric::Cosine.distance(&normalized(&a), &normalized(&b));
            // -dot(of normalized vectors) ∈ [-1, 1] (within a small float margin)
            prop_assert!((-1.0001..=1.0001).contains(&d));
            prop_assert!(!d.is_nan());
        }
    }
}
