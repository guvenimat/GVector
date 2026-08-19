//! Brute-force (doğrusal tarama) indeksi.
//!
//! Proje boyunca doğruluk referansı: her yeni indeks (HNSW vb.) buna karşı
//! sınanır. Bu yüzden burada "akıllı" hiçbir şey yok — okunabilirlik ve
//! doğruluk her şeyin önünde.

use crate::distance::{normalize, Metric};
use crate::index::{IndexError, VectorIndex};
use crate::types::{SearchResult, VectorId};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::collections::HashMap;

/// Kaç elemandan sonra aramayı rayon ile paralelleştireceğimiz eşiği.
/// Küçük indekslerde thread dağıtım maliyeti taramanın kendisinden pahalı;
/// eşik kabaca "tek çekirdeğin ~1 ms'de taradığı" boyut seçildi.
const PARALLEL_THRESHOLD: usize = 20_000;

pub struct BruteForceIndex {
    metric: Metric,
    dim: usize,
    /// Vektör verisi tek bitişik blokta (satır-major). `Vec<Vec<f32>>` yerine
    /// düz blok: cache dostu tarama + vektör başına 24 byte Vec başlığı yok.
    data: Vec<f32>,
    /// Slot -> dışa dönük id. `data`'daki i. vektörün sahibi `ids[i]`.
    ids: Vec<VectorId>,
    /// Dışa dönük id -> slot. Delete ve duplicate kontrolü O(1) olsun diye.
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

    /// Kapasitesi ÖNCEDEN ayrılmış boş indeks (#61).
    ///
    /// Yazma buffer'ı için: buffer mühürleme eşiğine kadar büyüyor ve
    /// `Vec`'in kademeli büyümesi eşiğe yakın yerlerde ~64 MB'lık bir
    /// realloc + memcpy'ye dönüşüyor — tek bir insert'in milisaniyelere
    /// çıkması bunun belirtisi. Kapasite TAM boyutta ayrılır (büyüme payı
    /// eklenmez): buffer zaten eşiğe varınca mühürleniyor, fazladan pay
    /// realloc'u önlemez ama bellek zirvesini şişirir.
    ///
    /// ÜÇ yapı da ayrılır — yalnız vektör bloğu değil: `ids` ve `slot_of`
    /// da kayıt sayısıyla büyüyor, `HashMap`'in yeniden hash'lenmesi de
    /// aynı sınıfta bir sıçrama üretir.
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

    /// i. slottaki vektör dilimi.
    #[inline]
    fn vector_at(&self, slot: usize) -> &[f32] {
        &self.data[slot * self.dim..(slot + 1) * self.dim]
    }

    /// Verilen slot aralığını tarayıp yerel top-k heap'i döndürür.
    /// Paralel ve seri yol aynı çekirdeği kullansın diye ayrık fonksiyon.
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

    /// Filtreli doğrusal tarama: `allow(id)` geçenler arasında top-k.
    /// Brute-force'ta filtre bedava — taramada atlanır, referans doğruluk kaynağı.
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

    /// id indekste kayıtlı mı?
    pub fn contains(&self, id: VectorId) -> bool {
        self.slot_of.contains_key(&id)
    }

    /// id'nin (cosine'da normalize edilmiş) vektörü.
    pub fn vector_of(&self, id: VectorId) -> Option<&[f32]> {
        self.slot_of.get(&id).map(|&s| self.vector_at(s))
    }

    /// (id, vektör) çiftleri — segment mühürleme buffer'ı boşaltırken kullanır.
    /// Cosine'da dönen vektörler normalize edilmiş halidir (idempotent).
    pub fn entries(&self) -> impl Iterator<Item = (VectorId, &[f32])> {
        self.ids
            .iter()
            .enumerate()
            .map(|(slot, &id)| (id, self.vector_at(slot)))
    }

    /// İndeksin yaklaşık bellek kullanımı (byte) — BENCHMARKS raporu için.
    pub fn memory_bytes(&self) -> usize {
        self.data.capacity() * std::mem::size_of::<f32>()
            + self.ids.capacity() * std::mem::size_of::<VectorId>()
            // HashMap girdisi başına kaba tahmin: (key + value) * doluluk payı
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
        // Cosine politikası: normalizasyon insert anında, yerinde (bkz. DECISIONS.md)
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
        // Cosine sözleşmesinin query tarafı: aramada bir kez normalize et.
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
            // Her rayon parçası kendi yerel top-k'sını üretir, sonra k'lık
            // heap'ler birleştirilir: paylaşılan mutable durum yok, kilit yok.
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
        out.reverse(); // heap kötüden iyiye boşalır; artan mesafe istiyoruz
        out
    }

    fn delete(&mut self, id: VectorId) -> Result<(), IndexError> {
        let slot = self.slot_of.remove(&id).ok_or(IndexError::NotFound(id))?;
        let last = self.ids.len() - 1;
        // swap-remove: son vektörü silinen slota taşı, O(1) silme.
        // Brute-force'ta slot sırası anlam taşımadığı için güvenli.
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
        assert_eq!(res[0].id, VectorId(1)); // gerçek eş, sıfır vektörden önce
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

    /// Ana doğruluk testi: referans exact_top_k ile birebir aynı sonuç.
    #[test]
    fn matches_reference_exact_scan_all_metrics() {
        let vecs = random_vectors(500, 16, 42);
        let queries = random_vectors(20, 16, 43);
        for metric in [Metric::L2, Metric::Dot, Metric::Cosine] {
            let idx = build(&vecs, metric);
            // referans tarafında cosine sözleşmesini elle uygula
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

    /// Paralel yol (n > eşik) seri yolla aynı sonucu vermeli.
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
