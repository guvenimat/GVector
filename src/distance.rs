//! Mesafe fonksiyonları.
//!
//! Sözleşme: tüm fonksiyonlar "küçük = daha yakın" döndürür.
//! - L2: kare Öklid mesafesi (sqrt alınmaz — sıralama için gereksiz maliyet,
//!   top-k sıralaması monoton dönüşümlerde değişmez).
//! - Dot: `-dot(a,b)` (benzerlik büyükken mesafe küçük olsun diye negatif).
//! - Cosine: politikamız gereği vektörler INSERT/QUERY anında normalize edilir
//!   ve cosine mesafesi `-dot` ile hesaplanır. Normalizasyon vektör başına bir
//!   kez ödenir; sıcak arama döngüsünde norm hesaplanmaz. Bu politika
//!   DECISIONS.md'de kayıtlıdır.

/// Desteklenen metrikler. İndeksler bunu konfigürasyon olarak alır.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Metric {
    /// Kare L2. Gerçek mesafe gerekiyorsa çağıran sqrt alır.
    L2,
    /// Negatif iç çarpım.
    Dot,
    /// Normalize edilmiş vektörler üzerinde negatif iç çarpım.
    /// İndeks bu metrikle kurulduysa insert ve query'de `normalize` çağırmakla yükümlüdür.
    Cosine,
}

impl Metric {
    /// İki vektör arasındaki mesafe ("küçük = yakın").
    ///
    /// Cosine için çağıranın normalizasyon sözleşmesine uyduğu varsayılır;
    /// burada tekrar norm hesaplanmaz (politika: insert anında normalize).
    #[inline]
    pub fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), b.len(), "boyut uyuşmazlığı");
        match self {
            Metric::L2 => l2_squared(a, b),
            Metric::Dot | Metric::Cosine => -dot(a, b),
        }
    }

    /// Bu metrik insert/query anında normalizasyon gerektiriyor mu?
    pub fn requires_normalization(&self) -> bool {
        matches!(self, Metric::Cosine)
    }
}

// SIMD notu: `map().sum()` float toplama sırasını korumak zorunda olduğundan
// LLVM reduction'ı vektörleştiremez (ölçüm: 128d dot ~60 ns, target-cpu=native
// ile bile). Açık f32x8 + iki akümülatör hem 8-yol SIMD hem ILP sağlar;
// toplama sırası değişir ama mesafe karşılaştırmalarında ~1 ulp fark önemsizdir.
// `wide` crate'i güvenli API sunar; #![deny(unsafe_code)] korunur.

use wide::f32x8;

#[inline]
fn as_f32x8(chunk: &[f32]) -> f32x8 {
    f32x8::from(<[f32; 8]>::try_from(chunk).expect("8'lik parça"))
}

/// İç çarpım (f32x8 SIMD, 16 eleman/iterasyon).
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

/// Kare L2 mesafesi (f32x8 SIMD, 16 eleman/iterasyon).
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

/// Vektörü yerinde birim norma getirir.
///
/// Sıfır vektör edge case'i: norm 0 ise vektör olduğu gibi bırakılır
/// (0/0 = NaN üretmek yerine). Sıfır vektörün her şeyle cosine benzerliği
/// 0 kabul edilir; -dot zaten bunu verir.
pub fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Kopyalayıp normalize eden yardımcı (query yolu için).
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
        // sıfır vektörle mesafe NaN olmamalı
        let d = Metric::Cosine.distance(&v, &[1.0, 0.0, 0.0]);
        assert!(!d.is_nan());
        assert!(approx(d, 0.0));
    }

    #[test]
    fn identical_vectors_are_closest_cosine() {
        let a = normalized(&[1.0, 2.0, 3.0]);
        let d_self = Metric::Cosine.distance(&a, &a);
        assert!(approx(d_self, -1.0)); // -cos(0) = -1, mümkün olan en küçük
    }

    #[test]
    fn metric_smaller_is_closer() {
        // yakın çift, uzak çifte göre daha küçük mesafe vermeli — her metrikte
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
            // -dot(normalize edilmiş) ∈ [-1, 1] (küçük float payıyla)
            prop_assert!((-1.0001..=1.0001).contains(&d));
            prop_assert!(!d.is_nan());
        }
    }
}
