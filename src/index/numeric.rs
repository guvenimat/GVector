//! Sayısal alan indeksi: Range koşulları için kardinalite tahmini + sınırlı
//! sayım (DECISIONS #31).
//!
//! İki bileşen, iki ayrı görev:
//! - **Eşit genişlikli histogram (64 kova)**: büyük-kol ŝ tahmini. Tahmin
//!   TEK SAYI DEĞİL [alt, üst] aralığıdır: tam içerilen kovalar alt sınırı,
//!   sınır kovaları eklenince üst sınırı verir. Kova içi düzgün dağılım
//!   varsayımı hiç yapılmaz — varsayım yerine belirsizlik açıkça taşınır,
//!   planlayıcı muhafazakâr tarafı seçer.
//! - **Değer-sıralı BTreeMap**: küçük-matches kararı tahminle DEĞİL,
//!   `enumerate_up_to(limit)` ile verilir — limit+1 elemana kadar gerçek
//!   sayım. Karar kesindir ve eşleşen id'ler tarama koluna bedavaya çıkar.
//!   (Histogram-yalnız tasarımda küçük kol kararı tahmine kalırdı; sınırda
//!   yanlış kol seçimi tam da kaçınmak istediğimiz patoloji.)
//!
//! Bakım: insert/remove O(log distinct). Aralık dışı insert'te histogram
//! %12.5 payla genişletilip sorted map'ten yeniden kurulur (O(distinct)) —
//! pay, monoton artan değer akışında sürekli yeniden kurmayı amorti eder.

use crate::meta::ordered_bits;
use crate::types::VectorId;
use std::collections::BTreeMap;

const BUCKETS: usize = 64;

#[derive(Debug)]
pub struct NumericFieldIndex {
    /// değer(bits) → o değerdeki id'ler.
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
            // Genişleme payı: sınıra tam oturtmak yerine %12.5 taşır —
            // monoton akışta her insert'te yeniden kurulumu engeller.
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
                // Aralık yalnız büyür; v her zaman [lo, hi] içindedir.
                let b = self.bucket(v);
                self.hist[b] = self.hist[b].saturating_sub(1);
            }
        }
    }

    fn rebuild_hist(&mut self) {
        self.hist.iter_mut().for_each(|c| *c = 0);
        // bits → f64 geri dönüşü yerine kova ataması için değeri saklamak
        // gerekirdi; bits monoton olduğundan kovayı bits uzayında hesaplamak
        // eşdeğer SANILABİLİR ama eşit-genişlik f64 uzayında tanımlı.
        // Bu yüzden değeri bits'ten geri çöz: ordered_bits tersinir.
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

    /// [qlo, qhi] (kapalı aralık) için kardinalite ARALIĞI: (alt, üst).
    /// Alt = tamamen içerilen kovaların toplamı; üst = + sınır kovaları.
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

    /// [qlo, qhi] içindeki id'leri sayarak topla; `limit`'i aşarsa None.
    /// Küçük-matches kararının kesin yolu: histogram sadece "büyük" derse
    /// kullanılır, "küçük" kararı her zaman gerçek sayımdır.
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

    /// Sınırsız tam sayım (post-filter fallback'i için).
    pub fn enumerate_all(&self, qlo: f64, qhi: f64) -> Vec<VectorId> {
        self.sorted
            .range(ordered_bits(qlo)..=ordered_bits(qhi))
            .flat_map(|(_, ids)| ids.iter().copied())
            .collect()
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
                "[{lo},{hi}]: {l} ≤ {truth} ≤ {u} değil"
            );
        }
    }

    #[test]
    fn enumerate_exact_and_limited() {
        let mut idx = NumericFieldIndex::new();
        for i in 0..100 {
            idx.insert((i % 10) as f64, VectorId(i)); // tekrar eden değerler
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
        // histogram toplamı total ile eşit kalmalı
        let hist_sum: usize = idx.hist.iter().sum();
        assert_eq!(hist_sum, truth);
    }

    #[test]
    fn monotonic_inserts_amortized_widening() {
        // Genişleme payı sayesinde monoton akış patlamamalı (davranış testi:
        // sadece doğruluk — histogram toplamı korunur).
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
