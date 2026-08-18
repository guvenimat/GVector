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
/// `quant` modülü de aynı adaylık yapısını kullanır.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Cand {
    pub(crate) dist: f32,
    pub(crate) slot: usize,
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

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
    /// Tombstone oranı bu eşiği aşınca delete otomatik compaction tetikler.
    pub tombstone_threshold: f64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 50,
            seed: crate::dataset::DEFAULT_SEED,
            tombstone_threshold: 0.3,
        }
    }
}

/// Filtreli aramanın gezinti istatistikleri (ölçüm ve ileride planlayıcı
/// sinyali). `admitted/visited` oranının çökmesi, fallback tetiklenmeden
/// yaşanan "sessiz recall düşüşü"nün imzasıdır.
#[derive(Debug, Clone, Copy, Default)]
pub struct FilterSearchStats {
    /// Taban katmanda ziyaret edilen node sayısı.
    pub visited: usize,
    /// Sonuç kümesine kabul edilen aday sayısı (eviction öncesi).
    pub admitted: usize,
    /// Graf araması k'dan az sonuç bulup doğrusal taramaya düşüldü mü?
    pub fallback_used: bool,
    /// Ziyaret bütçesi doldu da arama erken kesildi mi? (Kabul/ziyaret
    /// oranının çöktüğü patolojik durumun canlı tespiti — ölçümdeki
    /// "kümelenmiş × uzak sorgu" hücresi. Kesilince fallback taramaya geçilir.)
    pub budget_exhausted: bool,
}

/// Vektör verisinin nerede durduğu: bellekte sahipli blok ya da diskten
/// memmap ile lazy yüklenmiş bölge. Mmap yoluna yazılamaz; ilk insert'te
/// veri sahipli Vec'e kopyalanır (copy-on-write).
enum VectorStorage {
    Owned(Vec<f32>),
    /// Şimdilik inşa edilmiyor: memmap2 açılışı unsafe gerektirir ve crate
    /// deny(unsafe_code) ile derlenir; izin çıkarsa lazy load bunu kullanacak.
    #[allow(dead_code)]
    Mmap {
        map: memmap2::Mmap,
        /// f32 verisinin dosya içindeki byte offset'i (4'e hizalı garanti).
        offset: usize,
        /// f32 eleman sayısı.
        len: usize,
    },
}

impl VectorStorage {
    #[inline]
    fn as_slice(&self) -> &[f32] {
        match self {
            VectorStorage::Owned(v) => v,
            // cast_slice hizayı runtime'da doğrular; offset'i 4'e hizalı
            // yazdığımız ve mmap tabanı sayfa hizalı olduğu için güvenli.
            VectorStorage::Mmap { map, offset, len } => {
                bytemuck::cast_slice(&map[*offset..*offset + *len * 4])
            }
        }
    }

    /// Yazma erişimi: mmap destekliyse önce sahipli kopyaya dönüştür.
    fn to_owned_mut(&mut self) -> &mut Vec<f32> {
        if let VectorStorage::Mmap { .. } = self {
            *self = VectorStorage::Owned(self.as_slice().to_vec());
        }
        match self {
            VectorStorage::Owned(v) => v,
            VectorStorage::Mmap { .. } => unreachable!("üstte dönüştürüldü"),
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
    storage: VectorStorage,
    ids: Vec<VectorId>,
    slot_of: HashMap<VectorId, usize>,
    /// links[slot][level] = komşu slot listesi. `links[slot].len()-1` node'un en üst seviyesi.
    links: Vec<Vec<Vec<usize>>>,
    /// Graf giriş noktası (en yüksek seviyeli node).
    entry: Option<usize>,
    /// Tombstone bayrakları: silinen node graf'ta GEZİLİR (bağlantılılık
    /// için köprü görevi sürer) ama sonuçlara girmez. Gerçek temizlik compaction'da.
    deleted: Vec<bool>,
    deleted_count: usize,
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
            storage: VectorStorage::Owned(Vec::new()),
            ids: Vec::new(),
            slot_of: HashMap::new(),
            links: Vec::new(),
            entry: None,
            deleted: Vec::new(),
            deleted_count: 0,
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
        &self.storage.as_slice()[slot * self.dim..(slot + 1) * self.dim]
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
    ///
    /// `exclude_deleted`: true ise tombstone'lu node'lar GEZİLİR (komşuları
    /// keşfedilir — bağlantılılık köprüsü olarak lazımlar) ama sonuç kümesine
    /// alınmazlar. İnşa sırasında false: yeni node tombstone'lara da bağlanabilir,
    /// compaction zaten onları toptan temizleyecek.
    ///
    /// `filter`: metadata filtresi aynı ilkeyle çalışır — eşleşmeyen node
    /// gezilir (bağlantılılık), sonuca girmez. None = filtre yok.
    ///
    /// Dönüş: (sonuçlar, ziyaret edilen node sayısı, sonuç kümesine kabul
    /// edilen aday sayısı). Sayaçlar iki increment'ten ibaret — üretim yolu
    /// bedavaya enstrümante olur; filtreli aramada kabul/ziyaret oranının
    /// çökmesi "sessiz recall düşüşü"nün imzasıdır (plan: filtre ölçümü).
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        level: usize,
        exclude_deleted: bool,
        filter: Option<&dyn Fn(usize) -> bool>,
        visited_budget: Option<usize>,
    ) -> (Vec<Cand>, usize, usize) {
        // visited: slot başına bayrak. HashSet yerine Vec<bool>: n=100K'da bile
        // 100KB'lik tek allocation, dal başına hash maliyeti yok.
        let mut visited = vec![false; self.links.len()];
        let mut visited_count = 0usize;
        let mut admitted_count = 0usize;
        // candidates: en YAKIN tepede (min-heap, Reverse ile).
        let mut candidates: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        // results: en UZAK tepede (max-heap) — kötüleri atmak için.
        let mut results: BinaryHeap<Cand> = BinaryHeap::new();

        for &ep in entry_points {
            if visited[ep] {
                continue;
            }
            visited[ep] = true;
            visited_count += 1;
            let c = Cand {
                dist: self.dist_to(query, ep),
                slot: ep,
            };
            candidates.push(Reverse(c));
            let admissible = !(exclude_deleted && self.deleted[ep]) && filter.is_none_or(|f| f(ep));
            if admissible {
                results.push(c);
                admitted_count += 1;
            }
        }

        while let Some(Reverse(cur)) = candidates.pop() {
            // Bütçe: filtreli aramada kabul oranı çökünce gezinti tüm grafa
            // yayılabilir; bütçe bunu keser, çağıran taramaya geçer.
            if visited_budget.is_some_and(|b| visited_count >= b) {
                break;
            }
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
                visited_count += 1;
                let d = self.dist_to(query, nb);
                let within =
                    results.len() < ef || results.peek().is_none_or(|worst| d < worst.dist);
                if within {
                    let c = Cand { dist: d, slot: nb };
                    candidates.push(Reverse(c));
                    // Tombstone / filtre-dışı node gezilir ama sonuçlara girmez.
                    let admissible =
                        !(exclude_deleted && self.deleted[nb]) && filter.is_none_or(|f| f(nb));
                    if admissible {
                        results.push(c);
                        admitted_count += 1;
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
        let mut out = results.into_vec();
        out.sort();
        (out, visited_count, admitted_count)
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
        // Aday listesi: mevcut komşular, node'a uzaklıklarıyla, artan sırada.
        let mut cands: Vec<Cand> = self.links[node][level]
            .iter()
            .map(|&nb| Cand {
                dist: self
                    .metric
                    .distance(self.vector_at(node), self.vector_at(nb)),
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
            // İnişte tombstone da geçerli durak: sadece yol gösteriyor.
            ep = self
                .search_layer(query, &[ep], 1, level, false, None, None)
                .0[0]
                .slot;
        }
        // Taban katmanda geniş arama; ef en az k olmalı yoksa k sonuç çıkmaz.
        // Tombstone'lar burada sonuç dışı.
        let ef = ef.max(k);
        let (found, _, _) = self.search_layer(query, &[ep], ef, 0, true, None, None);
        found
            .into_iter()
            .take(k)
            .map(|c| SearchResult::new(self.ids[c.slot], c.dist))
            .collect()
    }

    /// Filtreli arama: `allow(id)` true dönen kayıtlar aday olabilir.
    /// Eşleşmeyen node'lar gezinti köprüsü olarak kullanılır (bkz. meta modülü).
    ///
    /// Doğruluk garantisi: graf araması k'dan az sonuç bulursa (aşırı seçici
    /// filtre grafın gezilen bölgesinde az eşleşme bıraktıysa) tüm yaşayan
    /// kayıtlar üzerinde filtreli doğrusal taramaya düşülür — yavaş ama eksiksiz.
    pub fn search_filtered_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allow: &dyn Fn(VectorId) -> bool,
    ) -> Vec<SearchResult> {
        self.search_filtered_stats(query, k, ef, allow, None).0
    }

    /// `search_filtered_with_ef`'in enstrümante hali — üretim yolu bu
    /// fonksiyonu sarar, dönüş tipi ekstra alan gerektiğinde imza bozulmadan
    /// `FilterSearchStats`'a alan eklenerek genişler.
    pub fn search_filtered_stats(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allow: &dyn Fn(VectorId) -> bool,
        visited_budget: Option<usize>,
    ) -> (Vec<SearchResult>, FilterSearchStats) {
        let mut stats = FilterSearchStats::default();
        let Some(entry) = self.entry else {
            return (Vec::new(), stats);
        };
        if k == 0 {
            return (Vec::new(), stats);
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
            ep = self
                .search_layer(query, &[ep], 1, level, false, None, None)
                .0[0]
                .slot;
        }
        let ef = ef.max(k);
        let slot_allow = |slot: usize| allow(self.ids[slot]);
        let (found, visited, admitted) =
            self.search_layer(query, &[ep], ef, 0, true, Some(&slot_allow), visited_budget);
        stats.visited = visited;
        stats.admitted = admitted;
        stats.budget_exhausted = visited_budget.is_some_and(|b| visited >= b);
        // Bütçe dolduysa kısmi sonuçları OLDUĞU GİBİ döndür — ne yapılacağına
        // (posting-list taraması vb.) çağıran karar verir; buradaki O(n)
        // fallback'i koşmak bütçenin amacını boşa çıkarırdı.
        if stats.budget_exhausted {
            let out = found
                .into_iter()
                .take(k)
                .map(|c| SearchResult::new(self.ids[c.slot], c.dist))
                .collect();
            return (out, stats);
        }
        if found.len() >= k {
            let out = found
                .into_iter()
                .take(k)
                .map(|c| SearchResult::new(self.ids[c.slot], c.dist))
                .collect();
            return (out, stats);
        }
        // Fallback: seçicilik gezilen bölgeyi aştı — doğrusal tarama.
        stats.fallback_used = true;
        let mut all: Vec<SearchResult> = (0..self.ids.len())
            .filter(|&s| !self.deleted[s] && slot_allow(s))
            .map(|s| SearchResult::new(self.ids[s], self.dist_to(query, s)))
            .collect();
        all.sort();
        all.truncate(k);
        (all, stats)
    }

    /// Graf kenar belleği dahil toplam indeks belleği (byte).
    pub fn memory_bytes(&self) -> (usize, usize) {
        let vec_bytes = self.storage.as_slice().len() * 4 + self.ids.capacity() * 8;
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
        let data = self.storage.to_owned_mut();
        data.extend_from_slice(vector);
        if self.metric.requires_normalization() {
            let start = slot * self.dim;
            normalize(&mut data[start..start + self.dim]);
        }
        self.ids.push(id);
        self.slot_of.insert(id, slot);
        self.deleted.push(false);

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
            ep = self.search_layer(&query, &[ep], 1, lc, false, None, None).0[0].slot;
        }

        // 2. faz: level..0 arası her katmanda ef_construction genişliğinde ara,
        // heuristic ile komşu seç, çift yönlü bağla, limit aşan komşuları kırp.
        let mut eps = vec![ep];
        for lc in (0..=level.min(top)).rev() {
            let (found, _, _) = self.search_layer(
                &query,
                &eps,
                self.params.ef_construction,
                lc,
                false,
                None,
                None,
            );
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

    /// Tombstone tabanlı silme: node graf'ta kalır (köprü görevi), sonuçlardan
    /// düşer. `slot_of`'tan çıkarıldığı için aynı id yeniden eklenebilir.
    fn delete(&mut self, id: VectorId) -> Result<(), IndexError> {
        let slot = self.slot_of.remove(&id).ok_or(IndexError::NotFound(id))?;
        self.deleted[slot] = true;
        self.deleted_count += 1;
        // Kritik durum: giriş noktası silindi. Tombstone waypoint olarak
        // çalışmaya devam edebilirdi ama tüm aramaların ölü bir node'dan
        // başlaması hem kafa karıştırıcı hem compaction'ı zorlaştırır —
        // yaşayan en yüksek seviyeli node'u yeni giriş yap.
        if self.entry == Some(slot) {
            self.pick_new_entry();
        }
        let ratio = self.deleted_count as f64 / self.ids.len() as f64;
        if ratio >= self.params.tombstone_threshold {
            self.compact();
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.ids.len() - self.deleted_count
    }
}

impl HnswIndex {
    /// Yaşayan node'lar arasından en yüksek seviyeliyi giriş noktası yapar.
    /// Hepsi silinmişse entry None olur (arama boş döner, insert sıfırdan kurar).
    fn pick_new_entry(&mut self) {
        self.entry = (0..self.ids.len())
            .filter(|&s| !self.deleted[s])
            .max_by_key(|&s| self.links[s].len());
    }

    /// Tombstone oranı ne olursa olsun indeksi yaşayan elemanlardan yeniden
    /// kurar: vektör verisi, graf kenarları ve tombstone bellekleri gerçekten
    /// geri verilir. O(n · insert) maliyetli — eşikle tetiklenmesinin nedeni bu.
    pub fn compact(&mut self) {
        let mut fresh = HnswIndex::new(self.dim, self.metric, self.params.clone());
        for slot in 0..self.ids.len() {
            if !self.deleted[slot] {
                // Cosine: saklanan vektör zaten normalize; tekrar normalize idempotent.
                fresh
                    .insert(self.ids[slot], self.vector_at(slot))
                    .expect("compaction insert'i başarısız olamaz");
            }
        }
        *self = fresh;
    }

    // Quantization (Aşama 6) grafın kendisini yeniden kullanır; bu erişimciler
    // `quant` modülünün donmuş kopya çıkarabilmesi için crate-içi açık.
    pub(crate) fn graph_links(&self) -> &[Vec<Vec<usize>>] {
        &self.links
    }
    pub(crate) fn graph_ids(&self) -> &[VectorId] {
        &self.ids
    }
    pub(crate) fn graph_entry(&self) -> Option<usize> {
        self.entry
    }
    pub(crate) fn graph_deleted(&self) -> &[bool] {
        &self.deleted
    }
    pub(crate) fn raw_vectors(&self) -> &[f32] {
        self.storage.as_slice()
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// id bu indekste yaşıyor mu (tombstone'lular hariç)?
    pub fn contains(&self, id: VectorId) -> bool {
        self.slot_of.contains_key(&id)
    }

    /// Yaşayan bir kaydın (normalize edilmiş olabilecek) vektörü.
    /// Planlayıcının tarama kolu id listesinden doğrudan mesafe hesaplar.
    pub fn vector_of(&self, id: VectorId) -> Option<&[f32]> {
        self.slot_of.get(&id).map(|&s| self.vector_at(s))
    }

    /// Tombstone oranı (test ve gözlem için).
    pub fn tombstone_ratio(&self) -> f64 {
        if self.ids.is_empty() {
            0.0
        } else {
            self.deleted_count as f64 / self.ids.len() as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Kalıcılık (Aşama 3)
//
// Dosya düzeni (tüm sayılar little-endian):
//   [0..4)   magic  b"GVDB"
//   [4..8)   format versiyonu (u32) = 1
//   [8..16)  meta uzunluğu (u64)
//   [16..16+meta_len)  bincode(Meta)
//   ...pad (meta sonunu 4 byte'a hizalar; f32 bölümü cast edilebilsin diye)
//   [data_off..data_off+n*dim*4)  ham f32 vektör verisi
//   [son 4 byte)  crc32 (kendinden önceki her şeyin)
//
// Vektör bölümü meta'nın DIŞINDA tutulur: memmap2 ile dosyayı açıp bu bölgeyi
// kopyasız kullanmak (lazy load) mümkün olsun diye. Meta (graf, id'ler) her
// durumda belleğe deserialize edilir — graf gezinmesi zaten rastgele erişimli
// ve küçük; asıl yer kaplayan vektör verisidir.
// ---------------------------------------------------------------------------

const MAGIC: [u8; 4] = *b"GVDB";
const FORMAT_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    #[error("io hatası: {0}")]
    Io(#[from] std::io::Error),
    #[error("bozuk dosya: {0}")]
    Corrupt(String),
    #[error("desteklenmeyen format versiyonu: {0} (bu sürüm {FORMAT_VERSION} okur)")]
    UnsupportedVersion(u32),
    #[error("serileştirme hatası: {0}")]
    Encode(#[from] bincode::Error),
}

/// Diske yazılan graf metadata'sı. Vektör verisi bilinçli olarak burada değil.
#[derive(serde::Serialize, serde::Deserialize)]
struct Meta {
    params: HnswParams,
    metric: Metric,
    dim: u64,
    n: u64,
    ids: Vec<VectorId>,
    links: Vec<Vec<Vec<u64>>>,
    entry: Option<u64>,
    /// Tombstone bayrakları (Aşama 4). Silinip yeniden eklenen id'ler yüzünden
    /// `ids` içinde aynı id iki slotta görünebilir; yaşayan olan tekildir.
    deleted: Vec<bool>,
}

fn corrupt(msg: impl Into<String>) -> PersistError {
    PersistError::Corrupt(msg.into())
}

impl HnswIndex {
    pub fn save(&self, path: &std::path::Path) -> Result<(), PersistError> {
        let meta = Meta {
            params: self.params.clone(),
            metric: self.metric,
            dim: self.dim as u64,
            n: self.ids.len() as u64,
            ids: self.ids.clone(),
            links: self
                .links
                .iter()
                .map(|ls| {
                    ls.iter()
                        .map(|l| l.iter().map(|&s| s as u64).collect())
                        .collect()
                })
                .collect(),
            entry: self.entry.map(|e| e as u64),
            deleted: self.deleted.clone(),
        };
        let meta_bytes = bincode::serialize(&meta)?;

        let mut buf: Vec<u8> = Vec::new();
        buf.extend(MAGIC);
        buf.extend(FORMAT_VERSION.to_le_bytes());
        buf.extend((meta_bytes.len() as u64).to_le_bytes());
        buf.extend(&meta_bytes);
        while !buf.len().is_multiple_of(4) {
            buf.push(0); // f32 bölümü hizalaması
        }
        buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(self.storage.as_slice()));

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&buf);
        buf.extend(hasher.finalize().to_le_bytes());

        // Önce geçici dosyaya yaz, sonra atomik rename: yarım yazılmış dosya
        // asıl yolun üstüne hiç gelmesin.
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, &buf)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Baytlardan yükleme — fuzz hedefi ve testler bu yolu paylaşır.
    /// Vektör verisi kopyalanır (Owned). Dönen indeks aramaya hazırdır.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<HnswIndex, PersistError> {
        let (meta, data_range) = Self::parse(bytes)?;
        let data: Vec<f32> = bytes[data_range]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 byte")))
            .collect();
        Self::rebuild(meta, VectorStorage::Owned(data))
    }

    /// Dosyadan yükleme.
    ///
    /// `lazy=true` niyeti vektör bölümünü memmap2 ile kopyasız kullanmaktır
    /// (VectorStorage::Mmap). Ancak `memmap2::Mmap::map` bir `unsafe fn`dir
    /// (harita yaşarken dosyanın değişmemesi çağıranın sorumluluğudur) ve
    /// crate `#![deny(unsafe_code)]` ile derlenir. Bu izin verilene dek
    /// lazy parametre kabul edilir ama iki yol da güvenli tam-okumayla
    /// çalışır — davranış aynı, sadece bellek kopyası tasarrufu ertelenmiş
    /// durumda (bkz. DECISIONS.md, Aşama 3).
    pub fn load(path: &std::path::Path, _lazy: bool) -> Result<HnswIndex, PersistError> {
        let bytes = std::fs::read(path)?;
        Self::load_from_bytes(&bytes)
    }

    /// Header + crc + sınır doğrulaması. Başarıda (Meta, f32 bölümünün aralığı).
    fn parse(bytes: &[u8]) -> Result<(Meta, std::ops::Range<usize>), PersistError> {
        if bytes.len() < 20 {
            return Err(corrupt("dosya header için bile kısa"));
        }
        if bytes[0..4] != MAGIC {
            return Err(corrupt("magic uyuşmuyor (bu bir GVDB dosyası değil)"));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().expect("4 byte"));
        if version != FORMAT_VERSION {
            return Err(PersistError::UnsupportedVersion(version));
        }
        // Checksum önce: gövde sağlam değilse gerisini yorumlamaya çalışma.
        let body = &bytes[..bytes.len() - 4];
        let stored_crc = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().expect("4 byte"));
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(body);
        if hasher.finalize() != stored_crc {
            return Err(corrupt("crc32 uyuşmuyor (dosya bozulmuş/kesilmiş)"));
        }
        let meta_len = u64::from_le_bytes(bytes[8..16].try_into().expect("8 byte")) as usize;
        let meta_end = 16usize
            .checked_add(meta_len)
            .ok_or_else(|| corrupt("meta_len taşıyor"))?;
        if meta_end > body.len() {
            return Err(corrupt("meta_len dosya boyutunu aşıyor"));
        }
        let meta: Meta = bincode::deserialize(&bytes[16..meta_end])?;
        let data_off = meta_end.div_ceil(4) * 4;
        let expected = (meta.n as usize)
            .checked_mul(meta.dim as usize)
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| corrupt("n*dim taşıyor"))?;
        if body.len() < data_off || body.len() - data_off != expected {
            return Err(corrupt(format!(
                "vektör bölümü {} byte olmalıydı, {} var",
                expected,
                body.len().saturating_sub(data_off)
            )));
        }
        Ok((meta, data_off..data_off + expected))
    }

    /// Meta + storage'dan çalışır indeks kurar; iç tutarlılığı doğrular
    /// (fuzz'da çökmemek için her slot referansı sınır kontrolünden geçer).
    fn rebuild(meta: Meta, storage: VectorStorage) -> Result<HnswIndex, PersistError> {
        let n = meta.n as usize;
        let dim = meta.dim as usize;
        if dim == 0 || dim > 1 << 20 {
            return Err(corrupt("mantıksız dim"));
        }
        if meta.ids.len() != n || meta.links.len() != n {
            return Err(corrupt("ids/links uzunluğu n ile uyuşmuyor"));
        }
        let mut links = Vec::with_capacity(n);
        for ls in &meta.links {
            if ls.is_empty() {
                return Err(corrupt("node'un hiç seviyesi yok"));
            }
            let mut node_levels = Vec::with_capacity(ls.len());
            for level in ls {
                let l: Vec<usize> = level.iter().map(|&s| s as usize).collect();
                if l.iter().any(|&s| s >= n) {
                    return Err(corrupt("komşu slot sınır dışı"));
                }
                node_levels.push(l);
            }
            links.push(node_levels);
        }
        if meta.deleted.len() != n {
            return Err(corrupt("deleted uzunluğu n ile uyuşmuyor"));
        }
        let deleted_count = meta.deleted.iter().filter(|&&d| d).count();
        let entry = match meta.entry {
            Some(e) if (e as usize) < n => Some(e as usize),
            Some(_) => return Err(corrupt("entry point sınır dışı")),
            // Tüm elemanlar tombstone'sa entry meşru olarak None olabilir.
            None if deleted_count == n => None,
            None => return Err(corrupt("yaşayan eleman var ama entry yok")),
        };
        // Sadece yaşayan slotlar id haritasına girer; tombstone slotundaki id
        // yeniden eklenmiş olabilir ve yaşayan kopyası haritada olmalı.
        let mut slot_of = HashMap::with_capacity(n - deleted_count);
        for (slot, &id) in meta.ids.iter().enumerate() {
            if meta.deleted[slot] {
                continue;
            }
            if slot_of.insert(id, slot).is_some() {
                return Err(corrupt("yinelenen yaşayan VectorId"));
            }
        }
        let ml = 1.0 / (meta.params.m.max(2) as f64).ln();
        Ok(HnswIndex {
            // RNG durumu diske yazılmaz; yükleme sonrası seviye ataması
            // seed ⊕ n'den yeniden türetilir (deterministik ama inşa
            // ortasındaki durumla birebir aynı değil — bkz. DECISIONS.md).
            rng: StdRng::seed_from_u64(meta.params.seed ^ meta.n),
            params: meta.params,
            metric: meta.metric,
            dim,
            ml,
            storage,
            ids: meta.ids,
            slot_of,
            links,
            entry,
            deleted: meta.deleted,
            deleted_count,
        })
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

    // ---- Silme / compaction testleri (Aşama 4) ----

    /// Compaction tetiklenmesin diye yüksek eşikli parametre.
    fn no_compact_params() -> HnswParams {
        HnswParams {
            tombstone_threshold: 2.0, // asla otomatik tetiklenmez
            ..Default::default()
        }
    }

    fn build_with(vecs: &[Vec<f32>], params: HnswParams) -> HnswIndex {
        let mut idx = HnswIndex::new(vecs[0].len(), Metric::L2, params);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        idx
    }

    #[test]
    fn delete_removes_from_results_and_len() {
        let vecs = random_vectors(100, 8, 42);
        let mut idx = build_with(&vecs, no_compact_params());
        idx.delete(VectorId(7)).unwrap();
        assert_eq!(idx.len(), 99);
        let res = idx.search(&vecs[7].clone(), 10);
        assert!(res.iter().all(|r| r.id != VectorId(7)));
        assert_eq!(
            idx.delete(VectorId(7)),
            Err(IndexError::NotFound(VectorId(7)))
        );
    }

    #[test]
    fn delete_entry_point_picks_new_entry_and_search_works() {
        let vecs = random_vectors(500, 8, 42);
        let mut idx = build_with(&vecs, no_compact_params());
        // giriş noktasını bul ve sil — kritik senaryo
        let entry_slot = idx.entry.unwrap();
        let entry_id = idx.ids[entry_slot];
        idx.delete(entry_id).unwrap();
        let new_entry = idx.entry.expect("yeni entry seçilmeliydi");
        assert_ne!(new_entry, entry_slot);
        assert!(!idx.deleted[new_entry]);
        // yeni entry yaşayanların en yüksek seviyelisi olmalı
        let max_level = (0..idx.ids.len())
            .filter(|&s| !idx.deleted[s])
            .map(|s| idx.links[s].len())
            .max()
            .unwrap();
        assert_eq!(idx.links[new_entry].len(), max_level);
        // arama hâlâ çalışıyor ve silinen id dönmüyor
        let res = idx.search(&vecs[0].clone(), 10);
        assert_eq!(res.len(), 10);
        assert!(res.iter().all(|r| r.id != entry_id));
    }

    #[test]
    fn delete_all_then_reinsert() {
        let vecs = random_vectors(20, 4, 42);
        let mut idx = build_with(&vecs, no_compact_params());
        for i in 0..20 {
            idx.delete(VectorId(i)).unwrap();
        }
        assert_eq!(idx.len(), 0);
        assert!(idx.search(&[0.0; 4], 5).is_empty());
        assert!(idx.entry.is_none());
        // boşalmış indekse ekleme sıfırdan kurulum gibi çalışmalı
        idx.insert(VectorId(100), &[1.0; 4]).unwrap();
        assert_eq!(idx.search(&[1.0; 4], 1)[0].id, VectorId(100));
    }

    #[test]
    fn deleted_id_can_be_reinserted_with_new_vector() {
        let vecs = random_vectors(50, 4, 42);
        let mut idx = build_with(&vecs, no_compact_params());
        idx.delete(VectorId(5)).unwrap();
        idx.insert(VectorId(5), &[9.0; 4]).unwrap();
        assert_eq!(idx.len(), 50);
        let res = idx.search(&[9.0; 4], 1);
        assert_eq!(res[0].id, VectorId(5));
    }

    #[test]
    fn recall_stays_high_after_20pct_deletion() {
        let vecs = random_vectors(2_000, 16, 42);
        let queries = random_vectors(30, 16, 43);
        let mut idx = build_with(&vecs, no_compact_params());
        let mut bf = BruteForceIndex::new(16, Metric::L2);
        for (i, v) in vecs.iter().enumerate() {
            bf.insert(VectorId(i as u64), v).unwrap();
        }
        // her 5. elemanı sil (%20)
        for i in (0..2_000).step_by(5) {
            idx.delete(VectorId(i)).unwrap();
            bf.delete(VectorId(i)).unwrap();
        }
        let mut hits = 0;
        let mut total = 0;
        for q in &queries {
            let truth: Vec<_> = bf.search(q, 10).iter().map(|r| r.id).collect();
            let got = idx.search_with_ef(q, 10, 100);
            assert_eq!(got.len(), 10, "silme sonrası eksik sonuç");
            hits += got.iter().filter(|r| truth.contains(&r.id)).count();
            total += truth.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.95, "silme sonrası recall {recall} < 0.95");
    }

    #[test]
    fn compaction_triggers_at_threshold_and_frees_memory() {
        let vecs = random_vectors(1_000, 16, 42);
        let mut idx = build_with(
            &vecs,
            HnswParams {
                tombstone_threshold: 0.3,
                ..Default::default()
            },
        );
        let (vec_before, link_before) = idx.memory_bytes();
        // %30 eşiğin hemen altına kadar sil — compaction tetiklenmemeli
        for i in 0..299 {
            idx.delete(VectorId(i)).unwrap();
        }
        assert!(idx.tombstone_ratio() > 0.0);
        // eşiği aşan silme compaction tetikler
        idx.delete(VectorId(299)).unwrap();
        assert_eq!(
            idx.tombstone_ratio(),
            0.0,
            "compaction tombstone bırakmamalı"
        );
        assert_eq!(idx.len(), 700);
        let (vec_after, link_after) = idx.memory_bytes();
        assert!(
            vec_after < vec_before && link_after < link_before,
            "bellek düşmedi: vec {vec_before}->{vec_after}, link {link_before}->{link_after}"
        );
        // compaction sonrası arama sağlıklı
        let res = idx.search(&vecs[500].clone(), 5);
        assert_eq!(res[0].id, VectorId(500));
    }

    #[test]
    fn persist_roundtrip_with_tombstones() {
        let vecs = random_vectors(200, 8, 42);
        let mut idx = build_with(&vecs, no_compact_params());
        for i in (0..200).step_by(7) {
            idx.delete(VectorId(i)).unwrap();
        }
        // silinmiş bir id'yi yeniden ekle: ids'te çift kayıt senaryosu
        idx.insert(VectorId(0), &[0.5; 8]).unwrap();
        let loaded = HnswIndex::load_from_bytes(&save_to_bytes(&idx)).unwrap();
        assert_eq!(idx.len(), loaded.len());
        for q in random_vectors(10, 8, 43) {
            assert_eq!(idx.search(&q, 10), loaded.search(&q, 10));
        }
    }

    // ---- Kalıcılık testleri (Aşama 3) ----

    fn save_to_bytes(idx: &HnswIndex) -> Vec<u8> {
        let dir = std::env::temp_dir().join(format!("gvdb-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("t-{:p}.gvdb", idx as *const _));
        idx.save(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        bytes
    }

    #[test]
    fn persist_roundtrip_identical_results() {
        let vecs = random_vectors(1_000, 16, 42);
        let idx = build(&vecs, Metric::L2);
        let loaded = HnswIndex::load_from_bytes(&save_to_bytes(&idx)).unwrap();
        for q in random_vectors(20, 16, 43) {
            assert_eq!(idx.search(&q, 10), loaded.search(&q, 10));
        }
        assert_eq!(idx.len(), loaded.len());
    }

    #[test]
    fn persist_roundtrip_cosine_normalized_data_preserved() {
        let vecs = random_vectors(200, 8, 42);
        let idx = build(&vecs, Metric::Cosine);
        let loaded = HnswIndex::load_from_bytes(&save_to_bytes(&idx)).unwrap();
        for q in random_vectors(5, 8, 43) {
            assert_eq!(idx.search(&q, 5), loaded.search(&q, 5));
        }
    }

    #[test]
    fn persist_empty_index_roundtrip() {
        let idx = HnswIndex::new(4, Metric::L2, HnswParams::default());
        let loaded = HnswIndex::load_from_bytes(&save_to_bytes(&idx)).unwrap();
        assert!(loaded.is_empty());
        assert!(loaded.search(&[0.0; 4], 3).is_empty());
    }

    #[test]
    fn persist_loaded_index_accepts_inserts() {
        let vecs = random_vectors(100, 8, 42);
        let idx = build(&vecs, Metric::L2);
        let mut loaded = HnswIndex::load_from_bytes(&save_to_bytes(&idx)).unwrap();
        loaded.insert(VectorId(999), &[0.5; 8]).unwrap();
        assert_eq!(loaded.len(), 101);
        let res = loaded.search(&[0.5; 8], 1);
        assert_eq!(res[0].id, VectorId(999));
    }

    #[test]
    fn persist_truncated_file_is_error_not_panic() {
        let idx = build(&random_vectors(100, 8, 42), Metric::L2);
        let bytes = save_to_bytes(&idx);
        for cut in [0, 3, 10, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                HnswIndex::load_from_bytes(&bytes[..cut]).is_err(),
                "kesik dosya (cut={cut}) hata döndürmeliydi"
            );
        }
    }

    #[test]
    fn persist_bitflip_detected_by_crc() {
        let idx = build(&random_vectors(100, 8, 42), Metric::L2);
        let bytes = save_to_bytes(&idx);
        // header'dan sonra çeşitli noktalarda tek bit boz
        for pos in [8, 20, bytes.len() / 2, bytes.len() - 10] {
            let mut bad = bytes.clone();
            bad[pos] ^= 0x01;
            assert!(
                HnswIndex::load_from_bytes(&bad).is_err(),
                "bit flip @{pos} yakalanmalıydı"
            );
        }
    }

    #[test]
    fn persist_wrong_magic_and_version() {
        let idx = build(&random_vectors(10, 4, 42), Metric::L2);
        let bytes = save_to_bytes(&idx);
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(matches!(
            HnswIndex::load_from_bytes(&bad),
            Err(PersistError::Corrupt(_))
        ));
        // versiyonu değiştir + crc'yi düzelt ki versiyon kontrolüne ulaşsın
        let mut bad = bytes.clone();
        bad[4] = 99;
        let body_len = bad.len() - 4;
        let mut h = crc32fast::Hasher::new();
        h.update(&bad[..body_len]);
        let crc = h.finalize().to_le_bytes();
        bad[body_len..].copy_from_slice(&crc);
        assert!(matches!(
            HnswIndex::load_from_bytes(&bad),
            Err(PersistError::UnsupportedVersion(99))
        ));
    }

    proptest! {
        /// Mini-fuzz (cargo-fuzz'ın CI'siz muadili): rastgele baytlar asla
        /// panic üretmemeli. Gerçek fuzz hedefi fuzz/fuzz_targets/load_index.rs.
        #[test]
        fn prop_load_random_bytes_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..2048)) {
            let _ = HnswIndex::load_from_bytes(&bytes);
        }

        /// Geçerli bir dosyanın rastgele bir baytını bozmak ya hata vermeli
        /// ya da (pad baytı gibi checksum'a girmeyen yer yoktur — crc her şeyi
        /// kapsar) asla panic olmamalı.
        #[test]
        fn prop_corrupted_valid_file_no_panic(pos in 0usize..500, xor in 1u8..255) {
            let idx = build(&random_vectors(20, 4, 42), Metric::L2);
            let mut bytes = save_to_bytes(&idx);
            let p = pos % bytes.len();
            bytes[p] ^= xor;
            // crc her baytı kapsadığından bozulma Err olmalı
            prop_assert!(HnswIndex::load_from_bytes(&bytes).is_err());
        }
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
