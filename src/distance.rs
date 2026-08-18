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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

/// İç çarpım.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    // zip ile iterasyon: bounds check'leri optimizer'ın eleyebildiği,
    // auto-vectorize edilebilen kanonik form.
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Kare L2 mesafesi.
#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
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
