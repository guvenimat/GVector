//! HNSW indeksi — Malkov & Yashunin (2016), "Efficient and robust approximate
//! nearest neighbor search using Hierarchical Navigable Small World graphs".
//!
//! Temsil: Rc/RefCell yok; her node bir `usize` slot, komşuluklar
//! `links[slot][level] = Vec<usize>`. Bu hem borrow-checker sürtünmesini
//! sıfırlar hem serileştirmeyi (Aşama 3) trivial yapar.
//!
//! Algoritma haritası (makaledeki numaralarla):
//! - Algorithm 1 (INSERT): `insert`
//! - Algorithm 2 (SEARCH-LAYER): `search_layer`
//! - Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC): `select_neighbors_heuristic`
//! - Algorithm 5 (K-NN-SEARCH): `search`

use crate::distance::{normalize, Metric};
use crate::index::{IndexError, VectorIndex};
use crate::types::{SearchResult, VectorId};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// (mesafe, slot) çifti; mesafe üzerinden total_cmp sıralaması.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cand {
    dist: f32,
    slot: usize,
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.slot.cmp(&other.slot))
    }
}

#[derive(Debug, Clone)]
pub struct HnswParams {
    /// Üst katmanlarda hedef komşu sayısı (makaledeki M).
    pub m: usize,
    /// Taban katman (0) için komşu limiti; makale 2M önerir.
    pub m_max0: usize,
    /// İnşa sırasındaki arama genişliği (efConstruction).
    pub ef_construction: usize,
    /// Sorgu sırasındaki taban katman genişliği (ef).
    pub ef_search: usize,
    /// Seviye atama rastgeleliği için seed (tekrarlanabilirlik).
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 50,
            seed: crate::dataset::DEFAULT_SEED,
        }
    }
}

pub struct HnswIndex {
    params: HnswParams,
    metric: Metric,
    dim: usize,
    /// mL = 1/ln(M): seviye dağılım çarpanı (makale 4.1'deki optimum).
    ml: f64,
    /// Vektörler tek bitişik blokta, slot-major.
    data: Vec<f32>,
    ids: Vec<VectorId>,
    slot_of: HashMap<VectorId, usize>,
    /// links[slot][level] = komşu slot listesi. `links[slot].len()-1` node'un en üst seviyesi.
    links: Vec<Vec<Vec<usize>>>,
    /// Graf giriş noktası (en yüksek seviyeli node).
    entry: Option<usize>,
    rng: StdRng,
}

impl HnswIndex {
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Self {
        let ml = 1.0 / (params.m as f64).ln();
        Self {
            rng: StdRng::seed_from_u64(params.seed),
            params,
            metric,
            dim,
            ml,
            data: Vec::new(),
            ids: Vec::new(),
            slot_of: HashMap::new(),
            links: Vec::new(),
            entry: None,
        }
    }

    pub fn params(&self) -> &HnswParams {
        &self.params
    }

    /// ef_search'ü sonradan ayarlamak parametre süpürmesi için gerekli;
    /// grafı etkilemez, sadece sorgu genişliğini değiştirir.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.params.ef_search = ef;
    }

    #[inline]
    fn vector_at(&self, slot: usize) -> &[f32] {
        &self.data[slot * self.dim..(slot + 1) * self.dim]
    }

    #[inline]
    fn dist_to(&self, query: &[f32], slot: usize) -> f32 {
        self.metric.distance(query, self.vector_at(slot))
    }

    /// Üstel seviye ataması: floor(-ln(U) * mL). U=0 alt sınırı clamp'lenir.
    fn random_level(&mut self) -> usize {
        let u: f64 = self.rng.gen_range(f64::MIN_POSITIVE..1.0);
        (-u.ln() * self.ml).floor() as usize
    }

    /// Algorithm 2 — SEARCH-LAYER: `entry_points`'ten başlayıp `level`'da
    /// ef genişliğinde greedy best-first arama. Artan mesafeli ef sonuç döndürür.
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        level: usize,
    ) -> Vec<Cand> {
        // visited: slot başına bayrak. HashSet yerine Vec<bool>: n=100K'da bile
        // 100KB'lik tek allocation, dal başına hash maliyeti yok.
        let mut visited = vec![false; self.links.len()];
        // candidates: en YAKIN tepede (min-heap, Reverse ile).
        let mut candidates: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        // results: en UZAK tepede (max-heap) — kötüleri atmak için.
        let mut results: BinaryHeap<Cand> = BinaryHeap::new();

        for &ep in entry_points {
            if visited[ep] {
                continue;
            }
            visited[ep] = true;
            let c = Cand {
                dist: self.dist_to(query, ep),
                slot: ep,
            };
            candidates.push(Reverse(c));
            results.push(c);
        }

        while let Some(Reverse(cur)) = candidates.pop() {
            // Erken çıkış: en yakın aday bile sonuç kümesinin en kötüsünden
            // uzaksa bu katmanda daha iyisi bulunamaz (makaledeki durdurma koşulu).
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
                let should_add =
                    results.len() < ef || d < results.peek().expect("results dolu").dist;
                if should_add {
                    let c = Cand { dist: d, slot: nb };
                    candidates.push(Reverse(c));
                    results.push(c);
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }
        let mut out = results.into_vec();
        out.sort();
        out
    }

    /// Algorithm 4 — SELECT-NEIGHBORS-HEURISTIC.
    ///
    /// Naif "en yakın M" yerine: aday, seçilmiş herhangi bir komşuya
    /// query'den olduğundan daha yakınsa ELENİR. Bu, aynı kümeden gereksiz
    /// kenarları kırpar ve kümeler ARASI köprü kenarları korur — grafın
    /// bağlantılılığı (dolayısıyla recall) buna dayanır.
    ///
    /// keepPrunedConnections=true davranışı: elenenlerden en yakınlarıyla
    /// M'e tamamla (makaledeki opsiyonel adım; düşük dereceli node bırakmamak için).
    fn select_neighbors_heuristic(&self, candidates: &[Cand], m: usize) -> Vec<usize> {
        let mut selected: Vec<Cand> = Vec::with_capacity(m);
        let mut pruned: Vec<Cand> = Vec::new();
        for &c in candidates {
            if selected.len() >= m {
                break;
            }
            let c_vec = self.vector_at(c.slot);
            // c, seçilmişlerden birine query'ye olduğundan daha mı yakın?
            let dominated = selected
                .iter()
                .any(|s| self.metric.distance(c_vec, self.vector_at(s.slot)) < c.dist);
            if dominated {
                pruned.push(c);
            } else {
                selected.push(c);
            }
        }
        // keepPrunedConnections: boş kalan kontenjanı elenen en yakınlarla doldur
        for c in pruned {
            if selected.len() >= m {
                break;
            }
            selected.push(c);
        }
        selected.into_iter().map(|c| c.slot).collect()
    }

    /// Bir seviyedeki komşu limiti: taban katman daha yoğun (makale: M_max0 = 2M).
    #[inline]
    fn max_links(&self, level: usize) -> usize {
        if level == 0 {
            self.params.m_max0
        } else {
            self.params.m
        }
    }

    /// `node`'un `level`'daki komşu listesi limiti aştıysa heuristic ile kırp.
    fn shrink_links(&mut self, node: usize, level: usize) {
        let limit = self.max_links(level);
        if self.links[node][level].len() <= limit {
            return;
        }
        let node_vec_start = node * self.dim;
        // Aday listesi: mevcut komşular, node'a uzaklıklarıyla, artan sırada.
        let mut cands: Vec<Cand> = self.links[node][level]
            .iter()
            .map(|&nb| Cand {
                dist: self.metric.distance(
                    &self.data[node_vec_start..node_vec_start + self.dim],
                    self.vector_at(nb),
                ),
                slot: nb,
            })
            .collect();
        cands.sort();
        self.links[node][level] = self.select_neighbors_heuristic(&cands, limit);
    }

    /// Sorguyu ef genişliğiyle çalıştırıp SearchResult listesi döndürür
    /// (parametre süpürmesinde set_ef_search'süz kullanım için).
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
        // Üst katmanlarda greedy iniş (ef=1): her katmanda en yakın node'a atla.
        let top = self.links[entry].len() - 1;
        let mut ep = entry;
        for level in (1..=top).rev() {
            ep = self.search_layer(query, &[ep], 1, level)[0].slot;
        }
        // Taban katmanda geniş arama; ef en az k olmalı yoksa k sonuç çıkmaz.
        let ef = ef.max(k);
        let found = self.search_layer(query, &[ep], ef, 0);
        found
            .into_iter()
            .take(k)
            .map(|c| SearchResult::new(self.ids[c.slot], c.dist))
            .collect()
    }

    /// Graf kenar belleği dahil toplam indeks belleği (byte).
    pub fn memory_bytes(&self) -> (usize, usize) {
        let vec_bytes = self.data.capacity() * 4 + self.ids.capacity() * 8;
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
        (vec_bytes, link_bytes)
    }
}

impl VectorIndex for HnswIndex {
    /// Algorithm 1 — INSERT.
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
        if self.metric.requires_normalization() {
            let start = slot * self.dim;
            normalize(&mut self.data[start..start + self.dim]);
        }
        self.ids.push(id);
        self.slot_of.insert(id, slot);

        let level = self.random_level();
        self.links.push(vec![Vec::new(); level + 1]);

        let Some(entry) = self.entry else {
            // İlk eleman: doğrudan giriş noktası olur.
            self.entry = Some(slot);
            return Ok(());
        };

        let query = self.vector_at(slot).to_vec(); // borrow ayrımı için kopya
        let top = self.links[entry].len() - 1;
        let mut ep = entry;

        // 1. faz: yeni node'un seviyesinin ÜSTÜNDEKİ katmanlarda sadece
        // greedy iniş — buralara kenar eklenmeyecek, sadece yaklaşıyoruz.
        for lc in ((level + 1)..=top).rev() {
            ep = self.search_layer(&query, &[ep], 1, lc)[0].slot;
        }

        // 2. faz: level..0 arası her katmanda ef_construction genişliğinde ara,
        // heuristic ile komşu seç, çift yönlü bağla, limit aşan komşuları kırp.
        let mut eps = vec![ep];
        for lc in (0..=level.min(top)).rev() {
            let found = self.search_layer(&query, &eps, self.params.ef_construction, lc);
            let neighbors = self.select_neighbors_heuristic(&found, self.params.m);
            for &nb in &neighbors {
                self.links[slot][lc].push(nb);
                self.links[nb][lc].push(slot);
                self.shrink_links(nb, lc);
            }
            // Bir alt katmana, bu katmanda bulunanların tümünden in (makale W'yi taşır).
            eps = found.into_iter().map(|c| c.slot).collect();
        }

        // Yeni node herkesten yüksekse giriş noktası el değiştirir.
        if level > top {
            self.entry = Some(slot);
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.search_with_ef(query, k, self.params.ef_search)
    }

    fn delete(&mut self, _id: VectorId) -> Result<(), IndexError> {
        // Aşama 4'te tombstone tabanlı silme gelecek.
        Err(IndexError::Unsupported("delete Aşama 4'te eklenecek"))
    }

    fn len(&self) -> usize {
        self.ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::random_vectors;
    use crate::index::bruteforce::BruteForceIndex;
    use proptest::prelude::*;

    fn build(vecs: &[Vec<f32>], metric: Metric) -> HnswIndex {
        let mut idx = HnswIndex::new(vecs[0].len(), metric, HnswParams::default());
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        idx
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = HnswIndex::new(4, Metric::L2, HnswParams::default());
        assert!(idx.is_empty());
        assert!(idx.search(&[0.0; 4], 5).is_empty());
    }

    #[test]
    fn single_element() {
        let idx = build(&[vec![1.0, 2.0]], Metric::L2);
        let res = idx.search(&[0.0, 0.0], 1);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId(0));
    }

    #[test]
    fn k_larger_than_len_returns_all() {
        let idx = build(&random_vectors(5, 8, 42), Metric::L2);
        assert_eq!(idx.search(&[0.0; 8], 100).len(), 5);
    }

    #[test]
    fn k_zero_returns_empty() {
        let idx = build(&random_vectors(5, 8, 42), Metric::L2);
        assert!(idx.search(&[0.0; 8], 0).is_empty());
    }

    #[test]
    fn duplicate_vectors_distinct_ids_both_found() {
        let mut vecs = random_vectors(50, 8, 42);
        vecs[10] = vecs[3].clone(); // birebir kopya
        let idx = build(&vecs, Metric::L2);
        let res = idx.search(&vecs[3].clone(), 2);
        let ids: Vec<_> = res.iter().map(|r| r.id).collect();
        assert!(ids.contains(&VectorId(3)) && ids.contains(&VectorId(10)));
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut idx = HnswIndex::new(2, Metric::L2, HnswParams::default());
        idx.insert(VectorId(1), &[1.0, 2.0]).unwrap();
        assert!(matches!(
            idx.insert(VectorId(1), &[3.0, 4.0]),
            Err(IndexError::DuplicateId(_))
        ));
    }

    #[test]
    fn zero_vector_cosine_no_nan() {
        let mut idx = HnswIndex::new(3, Metric::Cosine, HnswParams::default());
        idx.insert(VectorId(0), &[0.0; 3]).unwrap();
        idx.insert(VectorId(1), &[1.0, 0.0, 0.0]).unwrap();
        let res = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|r| !r.distance.is_nan()));
    }

    #[test]
    fn deterministic_across_builds_same_seed() {
        let vecs = random_vectors(200, 16, 7);
        let a = build(&vecs, Metric::L2);
        let b = build(&vecs, Metric::L2);
        let q = random_vectors(1, 16, 8).pop().unwrap();
        assert_eq!(a.search(&q, 10), b.search(&q, 10));
    }

    #[test]
    fn link_counts_respect_limits() {
        let idx = build(&random_vectors(2_000, 16, 42), Metric::L2);
        for (slot, levels) in idx.links.iter().enumerate() {
            for (level, nbrs) in levels.iter().enumerate() {
                assert!(
                    nbrs.len() <= idx.max_links(level),
                    "slot {slot} level {level}: {} > limit",
                    nbrs.len()
                );
                // Not: kenarların çift yönlülüğü GARANTİ DEĞİL — shrink_links
                // bir tarafı kırptığında karşı kenar kalır (graf yönlüdür,
                // hnswlib ile aynı davranış). Bu yüzden sadece komşu slotların
                // geçerli ve node'un o seviyeye sahip olduğunu doğruluyoruz.
                for &nb in nbrs {
                    assert!(nb < idx.links.len());
                    assert!(
                        level < idx.links[nb].len(),
                        "komşu {nb} seviye {level}'a sahip değil"
                    );
                }
            }
        }
    }

    #[test]
    fn recall_on_small_set_vs_bruteforce() {
        let vecs = random_vectors(2_000, 32, 42);
        let queries = random_vectors(50, 32, 43);
        let hnsw = build(&vecs, Metric::L2);
        let mut bf = BruteForceIndex::new(32, Metric::L2);
        for (i, v) in vecs.iter().enumerate() {
            bf.insert(VectorId(i as u64), v).unwrap();
        }
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let truth: Vec<_> = bf.search(q, 10).iter().map(|r| r.id).collect();
            let got = hnsw.search_with_ef(q, 10, 100);
            hits += got.iter().filter(|r| truth.contains(&r.id)).count();
            total += truth.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.95, "recall {recall} < 0.95");
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]
        /// İnvariant: yüksek ef_search'te HNSW'nin İLK sonucu brute-force'un
        /// ilk sonucuyla büyük ölçüde örtüşmeli.
        #[test]
        fn prop_top1_matches_bruteforce_at_high_ef(seed in 0u64..1000) {
            let vecs = random_vectors(300, 8, seed);
            let queries = random_vectors(20, 8, seed.wrapping_add(1));
            let hnsw = build(&vecs, Metric::L2);
            let mut bf = BruteForceIndex::new(8, Metric::L2);
            for (i, v) in vecs.iter().enumerate() {
                bf.insert(VectorId(i as u64), v).unwrap();
            }
            let mut agree = 0;
            for q in &queries {
                let h = hnsw.search_with_ef(q, 1, 300);
                let b = bf.search(q, 1);
                if h[0].id == b[0].id {
                    agree += 1;
                }
            }
            // 20 sorgunun en az 18'inde top-1 aynı olmalı
            prop_assert!(agree >= 18, "top-1 örtüşme {agree}/20");
        }
    }
}
