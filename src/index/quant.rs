//! Scalar quantization (Aşama 6): f32 → u8, per-dimension min/max kalibrasyonu.
//!
//! Tasarım: graf f32 hassasiyetle inşa edilir (inşa kalitesi tam hassasiyetten
//! yararlanır), sonra `QuantizedHnsw::from_hnsw` ile DONDURULUR: vektör verisi
//! u8 kodlara çevrilir, f32 kopyası atılır. Arama ADC (asymmetric distance
//! computation) kullanır: query f32 kalır, kodlar mesafe hesabı sırasında
//! anlık dequantize edilir — iki taraf birden quantize edilseydi hata iki kat
//! birikirdi, ADC bunun yarısını bedavaya kurtarır.
//!
//! Rerank YOK (saf quantization): gerekçe DECISIONS.md #23'te. Kısaca:
//! per-dimension kalibrasyonla SIFT tipi veride recall kaybı zaten < 0.02
//! hedefinin çok altında; diskten f32 okuyup yeniden sıralamak bir IO yolu,
//! bir dosya formatı bağımlılığı ve latency belirsizliği ekler.
//!
//! Donmuş indeks salt-okunurdur: insert/delete `Unsupported` döner. Segment
//! modelinde (Aşama 5) doğal karşılığı "mühürlenmiş segmentin quantize hali"dir;
//! yazma zaten buffer'a gider.

use crate::distance::Metric;
use crate::index::hnsw::{Cand, HnswIndex};
use crate::index::{IndexError, VectorIndex};
use crate::types::{SearchResult, VectorId};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Per-dimension doğrusal quantizer: `değer ≈ min[d] + scale[d] * kod`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ScalarQuantizer {
    mins: Vec<f32>,
    /// (max-min)/255; sabit boyutlarda (max==min) 0 olur, kod hep 0 üretir
    /// ve decode min'i döndürür — bilgi kaybı yok.
    scales: Vec<f32>,
}

impl ScalarQuantizer {
    /// Kalibrasyon: veri kümesinin her boyutunda gözlenen min/max.
    pub fn fit<'a>(vectors: impl Iterator<Item = &'a [f32]>, dim: usize) -> Self {
        let mut mins = vec![f32::INFINITY; dim];
        let mut maxs = vec![f32::NEG_INFINITY; dim];
        let mut any = false;
        for v in vectors {
            any = true;
            for d in 0..dim {
                mins[d] = mins[d].min(v[d]);
                maxs[d] = maxs[d].max(v[d]);
            }
        }
        if !any {
            mins.fill(0.0);
            maxs.fill(0.0);
        }
        let scales = mins
            .iter()
            .zip(&maxs)
            .map(|(lo, hi)| (hi - lo) / 255.0)
            .collect();
        Self { mins, scales }
    }

    pub fn dim(&self) -> usize {
        self.mins.len()
    }

    /// f32 → u8 kod. Aralık dışı değerler (kalibrasyon kümesi dışından gelen
    /// query'lerde olabilir) uçlara kırpılır.
    pub fn encode(&self, v: &[f32], out: &mut Vec<u8>) {
        debug_assert_eq!(v.len(), self.dim());
        out.extend(
            v.iter()
                .zip(self.mins.iter().zip(&self.scales))
                .map(|(&x, (&lo, &s))| {
                    if s == 0.0 {
                        0u8
                    } else {
                        (((x - lo) / s).round().clamp(0.0, 255.0)) as u8
                    }
                }),
        );
    }

    /// ADC mesafe: f32 query vs u8 kod, kod elemanları anlık dequantize edilir.
    /// distance modülündeki "küçük = yakın" sözleşmesine uyar.
    ///
    /// SIMD: dequantize (min + scale·kod) ve mesafe birikimi f32x8 şeritlerde;
    /// u8→f32 dönüşümü 8'lik sabit boy diziyle yapılır ki derleyici cvt
    /// komutlarına indirebilsin.
    #[inline]
    pub fn dist(&self, metric: Metric, query: &[f32], code: &[u8]) -> f32 {
        use wide::f32x8;
        debug_assert_eq!(query.len(), code.len());
        #[inline]
        fn f8(chunk: &[f32]) -> f32x8 {
            f32x8::from(<[f32; 8]>::try_from(chunk).expect("8'lik parça"))
        }
        #[inline]
        fn u8_to_f8(chunk: &[u8]) -> f32x8 {
            let mut arr = [0.0f32; 8];
            for (o, &c) in arr.iter_mut().zip(chunk) {
                *o = c as f32;
            }
            f32x8::from(arr)
        }
        let mut acc = f32x8::ZERO;
        let mut cq = query.chunks_exact(8);
        let mut cc = code.chunks_exact(8);
        let mut cm = self.mins.chunks_exact(8);
        let mut cs = self.scales.chunks_exact(8);
        let l2 = matches!(metric, Metric::L2);
        for (((q, c), m), s) in (&mut cq).zip(&mut cc).zip(&mut cm).zip(&mut cs) {
            let x = f8(m) + f8(s) * u8_to_f8(c);
            if l2 {
                let d = f8(q) - x;
                acc += d * d;
            } else {
                acc += f8(q) * x;
            }
        }
        let mut sum = acc.reduce_add();
        for (((q, c), m), s) in cq
            .remainder()
            .iter()
            .zip(cc.remainder())
            .zip(cm.remainder())
            .zip(cs.remainder())
        {
            let x = m + s * *c as f32;
            if l2 {
                let d = q - x;
                sum += d * d;
            } else {
                sum += q * x;
            }
        }
        // Cosine sözleşmesi: kodlar normalize edilmiş vektörlerden üretildi,
        // query'yi çağıran normalize eder → benzerlikler için -dot.
        if l2 {
            sum
        } else {
            -sum
        }
    }
}

/// Donmuş, quantize edilmiş HNSW: graf aynen, vektörler u8.
pub struct QuantizedHnsw {
    quantizer: ScalarQuantizer,
    metric: Metric,
    dim: usize,
    ef_search: usize,
    /// Slot-major u8 kodlar (n * dim byte) — f32'nin 1/4'ü.
    codes: Vec<u8>,
    ids: Vec<VectorId>,
    links: Vec<Vec<Vec<usize>>>,
    entry: Option<usize>,
    deleted: Vec<bool>,
    live: usize,
}

impl QuantizedHnsw {
    /// f32 indeksten donmuş quantize kopya üretir. Kaynak indeks bırakılırsa
    /// (drop) bellekte yalnızca kodlar kalır — "orijinal f32'yi tutma" kuralı
    /// çağıranın kaynağı düşürmesiyle tamamlanır.
    pub fn from_hnsw(src: &HnswIndex) -> Self {
        let dim = src.dim();
        let data = src.raw_vectors();
        let n = src.graph_ids().len();
        let quantizer = ScalarQuantizer::fit((0..n).map(|s| &data[s * dim..(s + 1) * dim]), dim);
        let mut codes = Vec::with_capacity(n * dim);
        for s in 0..n {
            quantizer.encode(&data[s * dim..(s + 1) * dim], &mut codes);
        }
        let deleted = src.graph_deleted().to_vec();
        let live = n - deleted.iter().filter(|&&d| d).count();
        Self {
            quantizer,
            metric: src.metric(),
            dim,
            ef_search: src.params().ef_search,
            codes,
            ids: src.graph_ids().to_vec(),
            links: src.graph_links().to_vec(),
            entry: src.graph_entry(),
            deleted,
            live,
        }
    }

    #[inline]
    fn code_at(&self, slot: usize) -> &[u8] {
        &self.codes[slot * self.dim..(slot + 1) * self.dim]
    }

    #[inline]
    fn dist_to(&self, query: &[f32], slot: usize) -> f32 {
        self.quantizer.dist(self.metric, query, self.code_at(slot))
    }

    /// hnsw::search_layer'ın ADC'li ikizi. Bilinçli kopya: donmuş indeksin
    /// arama yolu f32 indeksten bağımsız evrilebilsin diye (ve HnswIndex'i
    /// storage üzerinden generic'leştirmenin karmaşıklığına değmediği için).
    fn search_layer(&self, query: &[f32], entry: usize, ef: usize, level: usize) -> Vec<Cand> {
        let mut visited = vec![false; self.links.len()];
        let mut candidates: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        let mut results: BinaryHeap<Cand> = BinaryHeap::new();
        visited[entry] = true;
        let c = Cand {
            dist: self.dist_to(query, entry),
            slot: entry,
        };
        candidates.push(Reverse(c));
        if !self.deleted[entry] {
            results.push(c);
        }
        while let Some(Reverse(cur)) = candidates.pop() {
            if let Some(worst) = results.peek() {
                if cur.dist > worst.dist && results.len() >= ef {
                    break;
                }
            }
            for &nb in &self.links[cur.slot][level] {
                if visited[nb] {
                    continue;
                }
                visited[nb] = true;
                let d = self.dist_to(query, nb);
                let within = results.len() < ef || results.peek().is_none_or(|w| d < w.dist);
                if within {
                    let c = Cand { dist: d, slot: nb };
                    candidates.push(Reverse(c));
                    if !self.deleted[nb] {
                        results.push(c);
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
        let mut out = results.into_vec();
        out.sort();
        out
    }

    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
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
        let top = self.links[entry].len() - 1;
        let mut ep = entry;
        for level in (1..=top).rev() {
            // inişte ef=1; tombstone'lar waypoint olabilir diye results yerine
            // en yakın gezilen aday üzerinden ilerliyoruz
            let step = self.search_layer(query, ep, 1, level);
            if let Some(best) = step.first() {
                ep = best.slot;
            }
        }
        let ef = ef.max(k);
        self.search_layer(query, ep, ef, 0)
            .into_iter()
            .take(k)
            .map(|c| SearchResult::new(self.ids[c.slot], c.dist))
            .collect()
    }

    /// (kod belleği, graf belleği) byte.
    pub fn memory_bytes(&self) -> (usize, usize) {
        let link_bytes: usize = self
            .links
            .iter()
            .map(|levels| {
                std::mem::size_of::<Vec<usize>>()
                    + levels
                        .iter()
                        .map(|l| std::mem::size_of::<Vec<usize>>() + l.capacity() * 8)
                        .sum::<usize>()
            })
            .sum();
        (self.codes.len(), link_bytes)
    }
}

impl VectorIndex for QuantizedHnsw {
    fn insert(&mut self, _id: VectorId, _vector: &[f32]) -> Result<(), IndexError> {
        Err(IndexError::Unsupported(
            "quantize indeks donmuştur; yazma segment buffer'ına gider",
        ))
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.search_with_ef(query, k, self.ef_search)
    }

    fn delete(&mut self, _id: VectorId) -> Result<(), IndexError> {
        Err(IndexError::Unsupported("quantize indeks donmuştur"))
    }

    fn len(&self) -> usize {
        self.live
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::random_vectors;
    use crate::index::hnsw::HnswParams;

    #[test]
    fn quantizer_roundtrip_error_bounded() {
        let vecs = random_vectors(100, 8, 42);
        let quant = ScalarQuantizer::fit(vecs.iter().map(|v| v.as_slice()), 8);
        for v in &vecs {
            let mut code = Vec::new();
            quant.encode(v, &mut code);
            // ADC ile kendine mesafe, adım payından (scale/2)² * dim küçük olmalı
            let d = quant.dist(Metric::L2, v, &code);
            let bound: f32 = quant.scales.iter().map(|s| (s / 2.0) * (s / 2.0)).sum();
            assert!(
                d <= bound * 1.01,
                "quantization hatası sınır aşımı: {d} > {bound}"
            );
        }
    }

    #[test]
    fn constant_dimension_no_nan() {
        // sabit boyut: max==min, scale=0
        let vecs = [vec![1.0f32, 5.0], vec![2.0, 5.0], vec![3.0, 5.0]];
        let quant = ScalarQuantizer::fit(vecs.iter().map(|v| v.as_slice()), 2);
        let mut code = Vec::new();
        quant.encode(&vecs[0], &mut code);
        let d = quant.dist(Metric::L2, &vecs[0], &code);
        assert!(!d.is_nan());
        assert!(d < 1e-3);
    }

    #[test]
    fn empty_and_single_and_k_over_len() {
        let empty = QuantizedHnsw::from_hnsw(&HnswIndex::new(4, Metric::L2, HnswParams::default()));
        assert!(empty.search(&[0.0; 4], 5).is_empty());

        let mut one = HnswIndex::new(2, Metric::L2, HnswParams::default());
        one.insert(VectorId(0), &[1.0, 2.0]).unwrap();
        let q = QuantizedHnsw::from_hnsw(&one);
        assert_eq!(q.search(&[0.0, 0.0], 10).len(), 1);
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn zero_vector_cosine_no_nan() {
        let mut idx = HnswIndex::new(3, Metric::Cosine, HnswParams::default());
        idx.insert(VectorId(0), &[0.0; 3]).unwrap();
        idx.insert(VectorId(1), &[1.0, 0.0, 0.0]).unwrap();
        let q = QuantizedHnsw::from_hnsw(&idx);
        let res = q.search(&[1.0, 0.0, 0.0], 2);
        assert!(res.iter().all(|r| !r.distance.is_nan()));
        assert_eq!(res[0].id, VectorId(1));
    }

    #[test]
    fn tombstones_carried_over() {
        let vecs = random_vectors(200, 8, 42);
        let mut idx = HnswIndex::new(
            8,
            Metric::L2,
            HnswParams {
                tombstone_threshold: 2.0,
                ..Default::default()
            },
        );
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        idx.delete(VectorId(3)).unwrap();
        let q = QuantizedHnsw::from_hnsw(&idx);
        assert_eq!(q.len(), 199);
        let res = q.search(&vecs[3].clone(), 10);
        assert!(res.iter().all(|r| r.id != VectorId(3)));
    }

    #[test]
    fn quantized_recall_close_to_f32() {
        let vecs = random_vectors(2_000, 16, 42);
        let queries = random_vectors(50, 16, 43);
        let mut idx = HnswIndex::new(16, Metric::L2, HnswParams::default());
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        let quant = QuantizedHnsw::from_hnsw(&idx);
        let mut f32_hits = 0usize;
        let mut q_hits = 0usize;
        for q in &queries {
            let truth: Vec<_> = crate::eval::exact_top_k(&vecs, q, 10, Metric::L2)
                .iter()
                .map(|r| r.id)
                .collect();
            f32_hits += idx
                .search_with_ef(q, 10, 100)
                .iter()
                .filter(|r| truth.contains(&r.id))
                .count();
            q_hits += quant
                .search_with_ef(q, 10, 100)
                .iter()
                .filter(|r| truth.contains(&r.id))
                .count();
        }
        let f32_recall = f32_hits as f64 / 500.0;
        let q_recall = q_hits as f64 / 500.0;
        assert!(
            f32_recall - q_recall < 0.02,
            "quantization recall kaybı çok büyük: {f32_recall} -> {q_recall}"
        );
    }
}
