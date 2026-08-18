//! Segment tabanlı eşzamanlı indeks (Aşama 5) — Lucene/Qdrant modeli.
//!
//! Yapı:
//! - **Mühürlü segmentler**: salt-okunur HNSW indeksleri (`Arc<Segment>`).
//!   İçerikleri asla değişmez; silmeler segment-yerel tombstone kümesine yazılır.
//! - **Yazma buffer'ı**: küçük bir brute-force indeks. Insert'ler buraya gider;
//!   eşik aşılınca buffer HNSW'ye "mühürlenir" ve segment listesine eklenir.
//! - Arama tüm segmentleri + buffer'ı gezip sonuçları id bazında birleştirir.
//!
//! Kilit disiplini (neden aramalar pratikte bloklanmaz):
//! - Okuyucu, segment listesinin read kilidini yalnızca `Vec<Arc<Segment>>`
//!   klonlayacak kadar tutar (birkaç pointer kopyası) ve HNSW aramasını
//!   kilitsiz yapar — Arc içeriği immutable.
//! - Buffer araması read kilidi altındadır ama buffer küçüktür (< eşik) ve
//!   brute-force taraması mikrosaniyeler sürer; yazıcının buffer write kilidi
//!   de O(1) append kadar kısadır. Pahalı iş olan HNSW inşası (mühürleme)
//!   HİÇBİR kilit tutulmadan yapılır; sadece sonucun listeye eklenmesi kilitli.
//! - Mühürleme sırası: önce segment eklenir, SONRA buffer boşaltılır. Aradaki
//!   anda bir okuyucu aynı id'yi iki kaynaktan görebilir; birleştirme id
//!   bazında tekilleştirdiği için bu güvenlidir (kayıp yerine kopya tercih edildi).
//!
//! Tek-yazar varsayımı: `insert`/`delete` `&mut self` ister (VectorIndex
//! sözleşmesi zaten böyle). Okuyucular `&self` ile herhangi bir thread'den
//! arayabilir; `SegmentedIndex: Sync` olduğundan `Arc<RwLock<...>>` yerine
//! doğrudan `Arc<SegmentedIndex>` + tek yazıcı thread'i yeterlidir... yazıcı
//! da `&self` üzerinden çalışabilsin diye mutasyonlar iç kilitlerle yazıldı ve
//! `insert_shared`/`delete_shared` olarak da açıldı (stres testi bunu kullanır).

use crate::distance::Metric;
use crate::index::bruteforce::BruteForceIndex;
use crate::index::hnsw::{HnswIndex, HnswParams};
use crate::index::numeric::NumericFieldIndex;
use crate::index::{IndexError, VectorIndex};
use crate::meta::{Filter, MetaKey, MetaValue, Metadata, Predicate};
use crate::storage::wal::{self, ReplayReport, SyncPolicy, Wal, WalRecord};
use crate::storage::{self, Manifest, SegmentRef, StorageError};
use crate::types::{SearchResult, VectorId};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

/// Segmentin diskteki karşılığı. Dosya adı yazıldığı generation'ı taşır ve
/// DEĞİŞMEZ: bir segment bir kez yazılır, sonraki checkpoint'ler onu yalnız
/// manifest'ten referanslar (bkz. storage modülü başlığı).
#[derive(Debug, Clone)]
struct StoredFile {
    name: String,
    crc32: u32,
}

/// Mühürlü, immutable HNSW + kendi tombstone kümesi.
///
/// Tombstone'lar segment-YEREL: bir id silinip yeniden eklendiğinde eski
/// kopya kendi segmentinde sonsuza dek gölgede kalır, yeni kopya buffer'da
/// (sonra başka segmentte) yaşar — global bir silinmiş-küme olsaydı yeniden
/// ekleme eski kopyayı hortlatırdı.
struct Segment {
    index: HnswIndex,
    tombstones: RwLock<HashSet<VectorId>>,
    /// Diske yazılmışsa dosya adı + CRC; yeni mühürlenen ve merge çıktısı
    /// segmentlerde None (bir sonraki checkpoint yazar).
    stored: RwLock<Option<StoredFile>>,
}

pub struct SegmentedIndex {
    dim: usize,
    metric: Metric,
    /// Mühürlenen segmentlerin HNSW parametreleri.
    hnsw_params: HnswParams,
    /// Buffer bu boyuta ulaşınca mühürlenir.
    seal_threshold: usize,
    segments: RwLock<Vec<Arc<Segment>>>,
    buffer: RwLock<BruteForceIndex>,
    /// Sorgu genişliği (mühürlü segmentlerde).
    ef_search: usize,
    /// id → metadata. Vektör verisinden ayrı tutulur: segmentler immutable
    /// ama metadata idare (silme, yeniden ekleme) id düzeyinde akar.
    metadata: RwLock<HashMap<VectorId, Metadata>>,
    /// Eq posting-list'leri: (alan, değer) → yaşayan id kümesi.
    /// Planlayıcının O(1) kardinalite tahmini ve tarama kolunun id kaynağı.
    /// Insert/delete'te bakımı yapılır; Range koşulları kapsam dışı (DECISIONS #28).
    postings: RwLock<HashMap<(String, MetaKey), HashSet<VectorId>>>,
    /// Sayısal alanlar için Range indeksi: histogram (ŝ aralığı) +
    /// değer-sıralı map (sınırlı sayım). Bkz. numeric modülü / DECISIONS #31.
    numeric: RwLock<HashMap<String, NumericFieldIndex>>,
    /// Planlayıcı eşikleri (sorgu planlama parametreleri; graf parametresi değil).
    /// Değerler seçicilik ölçümünden (BENCHMARKS, filtre süpürmesi) türetildi.
    planner: PlannerConfig,
    /// Tavan bekçisi: mühürleme sonrası segment sayısı bunu aşarsa en KÜÇÜK
    /// iki segment birleştirilir. Gerekçe latency kazancı değil (eşit-recall
    /// karşılaştırmasında tam merge ~%20 — BENCHMARKS segcurve), sınırsız
    /// büyümeyi kesmek: eğri doğrusal (~+45µs/segment), 40 segment = ~1.8ms.
    /// En küçük iki: yeniden inşa maliyeti n'e bağlı — en ucuz birleştirme,
    /// ve boyutlar dengelenir (en-eski politikası dev segmenti boşuna
    /// yeniden kurabilirdi).
    max_segments: usize,
    /// Kalıcılık dizini (bağlıysa). Bellek-içi kullanımda None.
    storage_dir: RwLock<Option<PathBuf>>,
    /// Monoton checkpoint sayacı; dosya adlarının benzersizliği buna dayanır.
    generation: AtomicU64,
    /// Son başarılı checkpoint'in unix zamanı (0 = hiç yapılmadı).
    last_checkpoint: AtomicU64,
    /// Sıcak kalıcılık (Aşama 7b). None = yalnız checkpoint dayanıklılığı.
    wal: RwLock<Option<Wal>>,
    /// Açılışta yapılan WAL replay'inin raporu (gözlem / /stats).
    replay_report: RwLock<ReplayReport>,
}

/// Planlayıcı yapılandırması. Değerler 10K + 100K seçicilik süpürmelerinden
/// (BENCHMARKS) türetildi.
///
/// Neden gezinti-içi filtre üretim yolundan çıktı: 100K ölçümü, kümelenmiş
/// eşleşme + uzak sorguda gezinti-içi filtrenin grafın tamamına yayıldığını
/// (35ms'e kadar) VE ölçekle sessiz recall düşüşü başladığını (0.948) gösterdi.
/// Filtresiz gezinti bu patolojiye yapısal olarak bağışık: gezinti filtreye
/// hiç bakmaz, aynı ~µs yolunu yürür; filtre sonuçlara over-fetch ile uygulanır.
#[derive(Debug, Clone)]
pub struct PlannerConfig {
    /// est ≤ scan_factor·k → tarama kolu (küçük mutlak eşleşme).
    pub scan_factor: usize,
    /// est ≤ scan_fraction·n → tarama kolu. 0.05: bu bandın altında
    /// over-fetch'in beklenen eşleşmesi k'yı garanti edemiyor; taramanın
    /// maliyeti est ile sınırlı ve her sorgu konumunda öngörülebilir.
    pub scan_fraction: f64,
    /// Post-filter kolunda over-fetch: ef'' = overfetch_beta·k/ŝ.
    /// β=5: beklenen eşleşme 5k. β=3 ile sonuç SAYISI yetiyordu ama orta-band
    /// kümelenmiş sorguda gerçek top-k'nın bir kısmı pencere dışında kalıp
    /// recall 0.979'a düşüyordu (10K ölçümü); β=5 pencereyi kalite için genişletir.
    pub overfetch_beta: f64,
    /// ef'' üst tavanı = overfetch_cap_factor·ef (tahmin hatası ef''e çarpan
    /// olarak girer; tavan bunu sınırlar — kullanıcı geri bildirimi).
    pub overfetch_cap_factor: usize,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            scan_factor: 16,
            scan_fraction: 0.05,
            overfetch_beta: 5.0,
            overfetch_cap_factor: 8,
        }
    }
}

/// MetaValue'nun sayısal izdüşümü (Range indeksine girenler).
fn numeric_value(v: &MetaValue) -> Option<f64> {
    match v {
        MetaValue::Int(i) => Some(*i as f64),
        MetaValue::Float(f) => Some(*f),
        _ => None,
    }
}

/// Planlayıcının kol kararı. `Scan` id'leri yanında taşır (bedava çıktılar);
/// `Post` fallback kaynağını taşır ki <2k durumunda exact sayım yapılabilsin.
enum Arm {
    /// Bir koşulun eşleşmesi kesin sıfır — sonuç boş.
    Empty,
    /// Kesin küçük eşleşme kümesi: doğrudan tarama.
    Scan(HashSet<VectorId>),
    /// Filtresiz gezinti + over-fetch; ŝ üst-sınır tahmininden.
    Post {
        s_hat: f64,
        fallback: FallbackSource,
    },
    /// Tahmin yok (Eq'suz ve sayısal-indekssiz): gezinti-içi filtre.
    Legacy,
}

enum FallbackSource {
    Ids(HashSet<VectorId>),
    Range { key: String, lo: f64, hi: f64 },
}

impl SegmentedIndex {
    pub fn new(dim: usize, metric: Metric, hnsw_params: HnswParams, seal_threshold: usize) -> Self {
        let ef_search = hnsw_params.ef_search;
        Self {
            dim,
            metric,
            hnsw_params,
            seal_threshold,
            segments: RwLock::new(Vec::new()),
            buffer: RwLock::new(BruteForceIndex::new(dim, metric)),
            ef_search,
            metadata: RwLock::new(HashMap::new()),
            postings: RwLock::new(HashMap::new()),
            numeric: RwLock::new(HashMap::new()),
            planner: PlannerConfig::default(),
            max_segments: 8,
            storage_dir: RwLock::new(None),
            generation: AtomicU64::new(0),
            last_checkpoint: AtomicU64::new(0),
            wal: RwLock::new(None),
            replay_report: RwLock::new(ReplayReport::default()),
        }
    }

    /// Segment tavanını değiştir (test/deney için).
    pub fn set_max_segments(&mut self, max: usize) {
        self.max_segments = max.max(2);
    }

    /// Metadata'lı insert. Metadata'sız `insert_shared` boş geçer.
    pub fn insert_with_meta(
        &self,
        id: VectorId,
        vector: &[f32],
        meta: Metadata,
    ) -> Result<(), IndexError> {
        // Write-ahead sırası (DECISIONS #36): (1) validasyon — mutasyon yok,
        // (2) WAL append + politikaya göre fsync, (3) belleğe uygula.
        // Ters sırada "istemciye hata döndük ama kayıt bellekte kaldı ve
        // sonraki checkpoint onu kalıcılaştırdı" durumu oluşurdu.
        self.validate_insert(id, vector)?;
        if let Some(w) = self.wal.write().expect("kilit").as_mut() {
            w.append(&WalRecord::insert(id, vector, &meta))
                .map_err(|e| IndexError::Storage(e.to_string()))?;
        }
        self.apply_insert(id, vector, meta)
    }

    /// Insert'in bellek tarafı (WAL'sız). Replay bu yolu kullanır — replay
    /// sırasında WAL bağlı olmadığı için kayıtlar ikiye katlanmaz.
    fn apply_insert(&self, id: VectorId, vector: &[f32], meta: Metadata) -> Result<(), IndexError> {
        let should_seal = {
            let mut buffer = self.buffer.write().expect("kilit");
            buffer.insert(id, vector)?;
            buffer.len() >= self.seal_threshold
        }; // write kilidi düşer; mühürleme kilitsiz çalışacak
        if !meta.is_empty() {
            self.index_metadata(id, meta);
        }
        if should_seal {
            self.seal(); // tavan bekçisi seal'ın içinde
        }
        Ok(())
    }

    /// Insert doğrulaması: boyut ve id çakışması. Hiçbir şeyi değiştirmez —
    /// WAL'a yazmadan önce çağrılabilsin diye ayrı.
    fn validate_insert(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        if vector.len() != self.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if self.buffer.read().expect("kilit").contains(id) {
            return Err(IndexError::DuplicateId(id));
        }
        let segments = self.segments.read().expect("kilit");
        for seg in segments.iter() {
            if seg.index.contains(id) && !seg.tombstones.read().expect("kilit").contains(&id) {
                return Err(IndexError::DuplicateId(id));
            }
        }
        Ok(())
    }

    /// Metadata'yı depoya + türetilmiş indekslere (posting-list'ler, sayısal
    /// alanlar) işler. Insert yolu ile snapshot'tan yeniden kurma yolu bunu
    /// paylaşır: türetilmiş yapılar diske yazılmadığı için tek kaynak burası.
    fn index_metadata(&self, id: VectorId, meta: Metadata) {
        let mut postings = self.postings.write().expect("kilit");
        for (key, value) in &meta {
            postings
                .entry((key.clone(), value.key()))
                .or_default()
                .insert(id);
        }
        drop(postings);
        // Sayısal değerler Range indeksine de girer (Int/Float).
        let mut numeric = self.numeric.write().expect("kilit");
        for (key, value) in &meta {
            if let Some(v) = numeric_value(value) {
                numeric.entry(key.clone()).or_default().insert(v, id);
            }
        }
        drop(numeric);
        self.metadata.write().expect("kilit").insert(id, meta);
    }

    /// Kardinalite tahmini: Eq koşullarının posting sayılarının minimumu
    /// (VE bağlacı için üst sınır — kesişim daha küçük olabilir, büyük olamaz).
    /// Eq koşulu yoksa None (Range için histogram tutmuyoruz).
    /// Dönen küme: en küçük posting listesi (tarama kolunun aday kaynağı).
    fn estimate(&self, filter: &Filter) -> Option<(usize, HashSet<VectorId>)> {
        let keys = filter.eq_keys();
        if keys.is_empty() {
            return None;
        }
        let postings = self.postings.read().expect("kilit");
        let mut best: Option<&HashSet<VectorId>> = None;
        for (k, mk) in keys {
            match postings.get(&(k.to_string(), mk)) {
                // Herhangi bir Eq koşulunun hiç eşleşmesi yoksa sonuç boştur.
                None => return Some((0, HashSet::new())),
                Some(set) => {
                    if best.is_none_or(|b| set.len() < b.len()) {
                        best = Some(set);
                    }
                }
            }
        }
        best.map(|s| (s.len(), s.clone()))
    }

    /// Kol kararı (DECISIONS #29 + #31). Aralık tahmini muhafazakâr kullanılır:
    /// - Küçük kol kararı asla tahminle verilmez: Eq'te sayım zaten kesin,
    ///   Range'de sınırlı sayım (`enumerate_up_to`) kesinleştirir. Sınırlı
    ///   sayım yalnız alt sınır ≤ limit iken denenir (alt sınır bile büyükse
    ///   kesin büyüktür, sayım israf olur).
    /// - Büyük kolun ŝ'ı ÜST sınırların minimumu (VE bağlacı Fréchet üst
    ///   sınırı). Üst sınır küçük ŝ → büyük ef'' yönünde hata yapar; yanlışsa
    ///   bedeli recall değil latency (ve <2k fallback'i zaten var).
    fn plan(&self, filter: &Filter, k: usize) -> Arm {
        let n = self.len_shared().max(1);
        let scan_limit =
            (self.planner.scan_factor * k).max((self.planner.scan_fraction * n as f64) as usize);
        let eq = self.estimate(filter);
        if let Some((0, _)) = eq {
            return Arm::Empty;
        }
        let mut best_upper: Option<usize> = eq.as_ref().map(|(e, _)| *e);
        let mut best_small: Option<HashSet<VectorId>> = eq
            .as_ref()
            .filter(|(e, _)| *e <= scan_limit)
            .map(|(_, set)| set.clone());
        let mut range_fallback: Option<(String, f64, f64, usize)> = None;
        {
            let numeric = self.numeric.read().expect("kilit");
            for p in &filter.must {
                if let Predicate::Range { key, min, max } = p {
                    if let Some(fi) = numeric.get(key) {
                        let (lower, upper) = fi.estimate(*min, *max);
                        if upper == 0 {
                            return Arm::Empty;
                        }
                        if best_upper.is_none_or(|b| upper < b) {
                            best_upper = Some(upper);
                        }
                        if range_fallback
                            .as_ref()
                            .is_none_or(|(_, _, _, u)| upper < *u)
                        {
                            range_fallback = Some((key.clone(), *min, *max, upper));
                        }
                        if best_small.is_none() && lower <= scan_limit {
                            if let Some(ids) = fi.enumerate_up_to(*min, *max, scan_limit) {
                                best_small = Some(ids.into_iter().collect());
                            }
                        }
                    }
                }
            }
        }
        if let Some(ids) = best_small {
            return Arm::Scan(ids);
        }
        match best_upper {
            Some(upper) => Arm::Post {
                s_hat: (upper as f64 / n as f64).clamp(1e-6, 1.0),
                fallback: match eq {
                    Some((_, set)) => FallbackSource::Ids(set),
                    None => {
                        let (key, lo, hi, _) =
                            range_fallback.expect("upper varsa range kaynağı var");
                        FallbackSource::Range { key, lo, hi }
                    }
                },
            },
            None => Arm::Legacy,
        }
    }

    /// Ölçüm/test için: planlayıcının seçtiği kolun adı.
    pub fn debug_plan_arm(&self, filter: &Filter, k: usize) -> &'static str {
        match self.plan(filter, k) {
            Arm::Empty => "empty",
            Arm::Scan(_) => "scan",
            Arm::Post { .. } => "post",
            Arm::Legacy => "legacy",
        }
    }

    /// Ölçüm için: bir sayısal alanın [lo, hi] kardinalite aralığı tahmini.
    pub fn debug_range_estimate(&self, key: &str, lo: f64, hi: f64) -> (usize, usize) {
        self.numeric
            .read()
            .expect("kilit")
            .get(key)
            .map(|fi| fi.estimate(lo, hi))
            .unwrap_or((0, 0))
    }

    /// Tarama kolu: aday id'ler (en küçük posting listesi) üzerinde tam filtre
    /// + doğrudan mesafe. Exact — graf hiç açılmaz.
    fn scan_candidates(
        &self,
        query: &[f32],
        k: usize,
        candidates: &HashSet<VectorId>,
        filter: &Filter,
    ) -> Vec<SearchResult> {
        let normalized_query;
        let query: &[f32] = if self.metric.requires_normalization() {
            normalized_query = crate::distance::normalized(query);
            &normalized_query
        } else {
            query
        };
        let meta = self.metadata.read().expect("kilit");
        let buffer = self.buffer.read().expect("kilit");
        let segments: Vec<Arc<Segment>> = self
            .segments
            .read()
            .expect("kilit")
            .iter()
            .cloned()
            .collect();
        // Maliyet notları (ilk sürüm brute-force taramanın ~4 katıydı):
        // - Tek Eq'li filtrede posting listesi ZATEN kesin eşleşme kümesi:
        //   id başına metadata haritası sorgusu atlanır.
        // - Kaynak-dışı döngü: id başına "tüm kaynakları dene" yerine kaynak
        //   başına kalan id'ler elenir — aynı haritaya ardışık erişim,
        //   bulunan id bir daha denenmez.
        // - Top-k heap: O(est·log k), tüm listeyi sıralamak yerine.
        let mut heap: std::collections::BinaryHeap<SearchResult> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        let push = |id: VectorId, d: f32, heap: &mut std::collections::BinaryHeap<SearchResult>| {
            let cand = SearchResult::new(id, d);
            if heap.len() < k {
                heap.push(cand);
            } else if let Some(worst) = heap.peek() {
                if cand < *worst {
                    heap.pop();
                    heap.push(cand);
                }
            }
        };
        let mut remaining: Vec<VectorId> = if filter.must.len() > 1 {
            candidates
                .iter()
                .copied()
                .filter(|&id| filter.matches_id(&meta, id))
                .collect()
        } else {
            candidates.iter().copied().collect()
        };
        // En yeni kaynaktan eskiye: yaşayan kopya her zaman en yeni konumda
        // (sil→yeniden-ekle zinciri buffer'a, oradan daha yeni segmente gider).
        remaining.retain(|&id| {
            if let Some(v) = buffer.vector_of(id) {
                push(id, self.metric.distance(query, v), &mut heap);
                false
            } else {
                true
            }
        });
        for seg in segments.iter().rev() {
            if remaining.is_empty() {
                break;
            }
            let tombs = seg.tombstones.read().expect("kilit");
            remaining.retain(|&id| {
                if let Some(v) = seg.index.vector_of(id) {
                    if !tombs.contains(&id) {
                        push(id, self.metric.distance(query, v), &mut heap);
                    }
                    false // bu segmentte bulundu (canlı ya da gölge) — arama biter
                } else {
                    true
                }
            });
        }
        let mut out = heap.into_vec();
        out.sort();
        out
    }

    /// Filtreli arama — üç kollu planlayıcı (gerekçe: BENCHMARKS filtre
    /// süpürmesi; DECISIONS #28–29):
    /// 1. Eq tahmini küçükse (≤ max(scan_factor·k, scan_fraction·n)):
    ///    grafı açmadan posting listesinde doğrudan tarama — exact ve ucuz.
    /// 2. Aksi halde FİLTRESİZ gezinti + over-fetch (ef'' = β·k/ŝ, tavanlı),
    ///    filtre sonuçlara uygulanır; sonuç < 2k kalırsa exact taramaya düşer.
    ///    Filtresiz gezinti, gezinti-içi filtrenin kümelenmiş-eşleşme
    ///    patolojisine (grafın tamamını gezme) yapısal olarak bağışıktır.
    /// 3. Eq koşulu yoksa (tahmin yok) gezinti-içi filtre + found<k güvenlik
    ///    ağı — tek seçenek.
    pub fn search_filtered(&self, query: &[f32], k: usize, filter: &Filter) -> Vec<SearchResult> {
        if filter.must.is_empty() {
            return self.search_shared(query, k);
        }
        if k == 0 {
            return Vec::new();
        }
        // Kısayol: tek Eq ve her yaşayan kayıt eşleşiyor → filtre davranışsal
        // boş; filtresiz yol birebir eşdeğer (UI'dan varsayılan filtre durumu).
        if filter.must.len() == 1 {
            if let Some((est, _)) = self.estimate(filter) {
                if est >= self.len_shared().max(1) {
                    return self.search_shared(query, k);
                }
            }
        }
        match self.plan(filter, k) {
            Arm::Empty => Vec::new(),
            Arm::Scan(candidates) => self.scan_candidates(query, k, &candidates, filter),
            Arm::Post { s_hat, fallback } => {
                // Post-filter kolu: FİLTRESİZ gezinti (patolojiye bağışık) +
                // over-fetch + sonuçta filtre. ŝ üst sınır; gerçek seçicilik
                // küçükse sonuç 2k'nın altında kalır ve exact fallback koşar.
                let ef_over = ((self.planner.overfetch_beta * k as f64 / s_hat) as usize).clamp(
                    self.ef_search.max(2 * k),
                    (self.planner.overfetch_cap_factor * self.ef_search).max(4 * k),
                );
                let mut all: Vec<SearchResult> = Vec::new();
                {
                    let meta = self.metadata.read().expect("kilit");
                    let allow = |id: VectorId| filter.matches_id(&meta, id);
                    let segments: Vec<Arc<Segment>> = self
                        .segments
                        .read()
                        .expect("kilit")
                        .iter()
                        .cloned()
                        .collect();
                    for seg in &segments {
                        let tombs = seg.tombstones.read().expect("kilit");
                        let res = seg.index.search_with_ef(query, ef_over, ef_over);
                        all.extend(
                            res.into_iter()
                                .filter(|r| !tombs.contains(&r.id) && allow(r.id)),
                        );
                    }
                    let buffer = self.buffer.read().expect("kilit");
                    all.extend(buffer.search_filtered(query, k, &allow));
                } // kilitler düşer — fallback scan_candidates yeniden alacak
                all.sort();
                let mut seen = HashSet::with_capacity(all.len());
                all.retain(|r| seen.insert(r.id));
                if all.len() < 2 * k {
                    // Over-fetch penceresi eşleşme bölgesini ıskaladı (sorgu
                    // eşleşmelerden uzak) ya da üst-sınır tahmini şişkindi
                    // (VE korelasyonu): exact taramaya düş. Fallback adayları
                    // kaynaktan gelir — Eq posting'i ya da Range tam sayımı.
                    let candidates = match fallback {
                        FallbackSource::Ids(ids) => ids,
                        FallbackSource::Range { key, lo, hi } => self
                            .numeric
                            .read()
                            .expect("kilit")
                            .get(&key)
                            .map(|fi| fi.enumerate_all(lo, hi).into_iter().collect())
                            .unwrap_or_default(),
                    };
                    return self.scan_candidates(query, k, &candidates, filter);
                }
                all.truncate(k);
                all
            }
            // Tahmin yok (Eq'suz ve sayısal-indekssiz Range): gezinti-içi
            // filtre + found<k güvenlik ağı tek seçenek.
            Arm::Legacy => {
                let meta = self.metadata.read().expect("kilit");
                let allow = |id: VectorId| filter.matches_id(&meta, id);
                let segments: Vec<Arc<Segment>> = self
                    .segments
                    .read()
                    .expect("kilit")
                    .iter()
                    .cloned()
                    .collect();
                let mut all: Vec<SearchResult> = Vec::new();
                for seg in &segments {
                    let tombs = seg.tombstones.read().expect("kilit");
                    let allow_live = |id: VectorId| !tombs.contains(&id) && allow(id);
                    let want = k + tombs.len().min(k);
                    all.extend(seg.index.search_filtered_with_ef(
                        query,
                        want,
                        self.ef_search.max(want),
                        &allow_live,
                    ));
                }
                {
                    let buffer = self.buffer.read().expect("kilit");
                    all.extend(buffer.search_filtered(query, k, &allow));
                }
                all.sort();
                let mut seen = HashSet::with_capacity(all.len());
                all.retain(|r| seen.insert(r.id));
                all.truncate(k);
                all
            }
        }
    }

    /// Paylaşımlı (&self) insert — tek yazıcı thread'inden çağrılmalı.
    /// Birden çok yazıcı data race üretmez (her şey kilitli) ama duplicate id
    /// kontrolü iki yazıcı arasında yarışabilir; sözleşme tek yazıcıdır.
    pub fn insert_shared(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        self.insert_with_meta(id, vector, Metadata::new())
    }

    /// Buffer'ı HNSW segmentine dönüştürür. Pahalı inşa kilitsiz yapılır;
    /// okuyucular bu süre boyunca eski segmentler + dolu buffer'ı görmeye
    /// devam eder (hiçbir vektör görünmez olmaz).
    fn seal(&self) {
        // 1. Buffer'ın anlık kopyasını al (read kilidi kısa tutulur).
        let entries: Vec<(VectorId, Vec<f32>)> = {
            let buffer = self.buffer.read().expect("kilit");
            buffer.entries().map(|(id, v)| (id, v.to_vec())).collect()
        };
        if entries.is_empty() {
            return;
        }
        // 2. HNSW'yi kilit dışında inşa et (asıl maliyet burada).
        let mut params = self.hnsw_params.clone();
        // Her segment farklı ama deterministik seed alsın: aynı inşa sırası
        // aynı grafı üretsin, segmentler arası graf yapısı korelasyonsuz olsun.
        params.seed = params.seed.wrapping_add(entries.len() as u64);
        let mut hnsw = HnswIndex::new(self.dim, self.metric, params);
        for (id, v) in &entries {
            hnsw.insert(*id, v)
                .expect("mühürleme insert'i başarısız olamaz");
        }
        let segment = Arc::new(Segment {
            index: hnsw,
            tombstones: RwLock::new(HashSet::new()),
            stored: RwLock::new(None), // ilk checkpoint yazacak
        });
        // 3. Önce segmenti yayınla, SONRA buffer'ı boşalt. Aradaki pencerede
        // kopya görünür (id bazlı dedupe emer); ters sıra veri kaybettirirdi.
        self.segments.write().expect("kilit").push(segment);
        {
            let mut buffer = self.buffer.write().expect("kilit");
            // Tek yazıcı sözleşmesi: seal ile bu satır arasında insert olamaz,
            // buffer içeriği hâlâ `entries` ile birebir aynı — komple sıfırla.
            *buffer = BruteForceIndex::new(self.dim, self.metric);
        }
        // Tavan bekçisi: mühürleme mekanizmasının "iki girdi, bir çıktı"
        // varyantı. Yazıcıyı inşa süresince meşgul eder (seal ile aynı
        // sözleşme); okuyucular takas anına dek eski iki segmenti aramaya
        // devam eder — o pencerede 3 kopya yaşar (bellek tepe noktası,
        // BENCHMARKS'ta ölçülü).
        while self.segments.read().expect("kilit").len() > self.max_segments {
            self.merge_smallest_pair();
        }
    }

    /// En küçük (canlı sayıya göre) iki segmenti tek segmentte yeniden inşa
    /// eder. Merge doğal compaction: tombstone'lular yeni segmente taşınmaz,
    /// merged segmentin tombstone kümesi boş başlar.
    fn merge_smallest_pair(&self) {
        // 1. Kurbanları seç (read kilidi kısa; Arc klonları inşa boyunca yaşar).
        let (a, b) = {
            let segments = self.segments.read().expect("kilit");
            if segments.len() < 2 {
                return;
            }
            let live = |s: &Arc<Segment>| s.index.len() - s.tombstones.read().expect("kilit").len();
            let mut order: Vec<usize> = (0..segments.len()).collect();
            order.sort_by_key(|&i| live(&segments[i]));
            (segments[order[0]].clone(), segments[order[1]].clone())
        };
        // 2. Kilitsiz yeniden inşa. Tek yazıcı sözleşmesi: bu sırada delete
        // gelemez, tombstone kümeleri donmuş sayılır.
        let mut params = self.hnsw_params.clone();
        let total = a.index.len() + b.index.len();
        params.seed = params.seed.wrapping_add(total as u64).wrapping_add(1);
        let mut merged = HnswIndex::new(self.dim, self.metric, params);
        for seg in [&a, &b] {
            let tombs = seg.tombstones.read().expect("kilit");
            for (id, v) in seg.index.live_entries() {
                if !tombs.contains(&id) {
                    merged
                        .insert(id, v)
                        .expect("merge insert'i başarısız olamaz");
                }
            }
        }
        let merged = Arc::new(Segment {
            index: merged,
            tombstones: RwLock::new(HashSet::new()),
            stored: RwLock::new(None), // birleşik yeni dosyaya yazılacak
        });
        // 3. Atomik takas: iki kaynağı çıkar, birleşiği ekle. Arc kimliğiyle
        // eşle — indeksler inşa sırasında kaymış olabilir (tek yazıcıda
        // kaymaz ama kimlik eşleme varsayım taşımaz).
        let mut segments = self.segments.write().expect("kilit");
        segments.retain(|s| !Arc::ptr_eq(s, &a) && !Arc::ptr_eq(s, &b));
        segments.push(merged);
    }

    /// Paylaşımlı (&self) silme — tek yazıcı thread'inden.
    /// Metadata da düşürülür (yeniden eklemede eski metadata sızmasın).
    pub fn delete_shared(&self, id: VectorId) -> Result<(), IndexError> {
        // Write-ahead: önce "bu id yaşıyor mu" (mutasyonsuz kontrol), sonra
        // WAL, sonra gerçek silme.
        if !self.contains_live(id) {
            return Err(IndexError::NotFound(id));
        }
        if let Some(w) = self.wal.write().expect("kilit").as_mut() {
            w.append(&WalRecord::delete(id))
                .map_err(|e| IndexError::Storage(e.to_string()))?;
        }
        self.apply_delete(id)
    }

    /// id yaşayan bir kayda mı ait? (buffer ya da tombstone'suz segment)
    fn contains_live(&self, id: VectorId) -> bool {
        if self.buffer.read().expect("kilit").contains(id) {
            return true;
        }
        self.segments.read().expect("kilit").iter().any(|seg| {
            seg.index.contains(id) && !seg.tombstones.read().expect("kilit").contains(&id)
        })
    }

    /// Silmenin bellek tarafı (WAL'sız); replay bu yolu kullanır.
    fn apply_delete(&self, id: VectorId) -> Result<(), IndexError> {
        let res = self.delete_vector_only(id);
        if res.is_ok() {
            // Posting-list'ler yalnız yaşayan id'leri içerir: metadata'yı
            // düşürmeden önce anahtarlarını okuyup listelerden çıkar.
            if let Some(meta) = self.metadata.write().expect("kilit").remove(&id) {
                let mut postings = self.postings.write().expect("kilit");
                for (key, value) in &meta {
                    if let Some(set) = postings.get_mut(&(key.clone(), value.key())) {
                        set.remove(&id);
                    }
                }
                drop(postings);
                let mut numeric = self.numeric.write().expect("kilit");
                for (key, value) in &meta {
                    if let Some(v) = numeric_value(value) {
                        if let Some(fi) = numeric.get_mut(key) {
                            fi.remove(v, id);
                        }
                    }
                }
            }
        }
        res
    }

    fn delete_vector_only(&self, id: VectorId) -> Result<(), IndexError> {
        // Önce buffer: oradaysa gerçek silme (brute-force swap-remove).
        {
            let mut buffer = self.buffer.write().expect("kilit");
            match buffer.delete(id) {
                Ok(()) => return Ok(()),
                Err(IndexError::NotFound(_)) => {}
                Err(e) => return Err(e),
            }
        }
        // Sonra segmentler (yeniden ekleme zincirinde yaşayan kopya en yenidedir,
        // yeniden ekleme buffer'a gittiği için segmentlerde en fazla bir canlı kopya olur).
        let segments = self.segments.read().expect("kilit");
        for seg in segments.iter().rev() {
            if seg.index.contains(id) {
                let mut tombs = seg.tombstones.write().expect("kilit");
                if tombs.insert(id) {
                    return Ok(());
                }
                // zaten tombstone'luysa daha eski segmentlere bakmaya gerek yok:
                // canlı kopya olsaydı bu tombstone atılırken silinmiş olurdu
                return Err(IndexError::NotFound(id));
            }
        }
        Err(IndexError::NotFound(id))
    }

    /// Kilitsiz-esaslı arama: segment listesi klonlanır, HNSW aramaları
    /// hiçbir kilit tutulmadan koşar; buffer araması kısa read kilidiyle.
    pub fn search_shared(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        if k == 0 {
            return Vec::new();
        }
        let segments: Vec<Arc<Segment>> = self
            .segments
            .read()
            .expect("kilit")
            .iter()
            .cloned()
            .collect();

        let mut all: Vec<SearchResult> = Vec::new();
        for seg in &segments {
            let tombs = seg.tombstones.read().expect("kilit");
            // Tombstone'lar sonuçları eleyebileceği için fazladan aday iste;
            // tombstone sayısı k'yı aşarsa bile ef zaten üst sınırı belirler.
            let want = k + tombs.len().min(k);
            let res = seg
                .index
                .search_with_ef(query, want, self.ef_search.max(want));
            all.extend(res.into_iter().filter(|r| !tombs.contains(&r.id)));
        }
        {
            let buffer = self.buffer.read().expect("kilit");
            all.extend(buffer.search(query, k));
        }
        // id bazında tekilleştir (mühürleme penceresindeki kopyalar için):
        // aynı id'nin kopyaları aynı vektördür, hangisi kalırsa kalsın.
        all.sort();
        let mut seen = HashSet::with_capacity(all.len());
        all.retain(|r| seen.insert(r.id));
        all.truncate(k);
        all
    }

    pub fn len_shared(&self) -> usize {
        let segments = self.segments.read().expect("kilit");
        let seg_live: usize = segments
            .iter()
            .map(|s| s.index.len() - s.tombstones.read().expect("kilit").len())
            .sum();
        seg_live + self.buffer.read().expect("kilit").len()
    }

    // ---- Kalıcılık: soğuk yol (Aşama 7a) ----

    /// Kalıcılık dizinini bağlar (bellek-içi indeksi kalıcı hale getirir).
    pub fn attach_storage(&self, dir: PathBuf) {
        *self.storage_dir.write().expect("kilit") = Some(dir);
    }

    pub fn storage_dir(&self) -> Option<PathBuf> {
        self.storage_dir.read().expect("kilit").clone()
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Son başarılı checkpoint'in unix zamanı; 0 = henüz yok.
    pub fn last_checkpoint_unix(&self) -> u64 {
        self.last_checkpoint.load(Ordering::Relaxed)
    }

    /// Checkpoint: buffer'ı mühürle → yeni segmentleri yaz → metadata
    /// snapshot'ı yaz → manifest'i atomik takas et → referanssızları temizle.
    ///
    /// Yazma sırası kritik: manifest EN SON yazılır, GC ondan SONRA çalışır.
    /// Böylece her an diskteki manifest, referansladığı tüm dosyalar var
    /// olacak şekilde tutarlıdır; kesinti hangi adımda olursa olsun eski
    /// manifest geçerli kalır (yeni dosyalar yetim kalır, sonraki GC toplar).
    ///
    /// Tek yazar sözleşmesi gereği yazıcı task'inden çağrılır.
    pub fn checkpoint(&self) -> Result<u64, StorageError> {
        let dir = self.storage_dir().ok_or_else(|| {
            StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "kalıcılık dizini bağlı değil (attach_storage/open_or_create)",
            ))
        })?;
        std::fs::create_dir_all(&dir)?;
        // Buffer'ı mühürle: checkpoint sonrası tüm veri segmentlerde olsun ki
        // WAL rotasyonu (7b) hiçbir kaydı sahipsiz bırakmasın.
        self.seal();
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;

        let segments: Vec<Arc<Segment>> = self
            .segments
            .read()
            .expect("kilit")
            .iter()
            .cloned()
            .collect();
        let mut refs = Vec::with_capacity(segments.len());
        for (i, seg) in segments.iter().enumerate() {
            let existing = seg.stored.read().expect("kilit").clone();
            let stored = match existing {
                // Değişmez: bir kez yazılan segment bir daha yazılmaz.
                Some(s) => s,
                None => {
                    let bytes = seg.index.to_bytes()?;
                    let name = Manifest::segment_file_name(generation, i);
                    storage::write_file_durable(&dir.join(&name), &bytes)?;
                    let s = StoredFile {
                        name,
                        crc32: storage::crc32(&bytes),
                    };
                    *seg.stored.write().expect("kilit") = Some(s.clone());
                    s
                }
            };
            refs.push(SegmentRef {
                file: stored.name,
                crc32: stored.crc32,
                records: seg.index.len() as u64,
                tombstones: seg
                    .tombstones
                    .read()
                    .expect("kilit")
                    .iter()
                    .map(|id| id.0)
                    .collect(),
            });
        }

        let entries: Vec<(VectorId, Metadata)> = self
            .metadata
            .read()
            .expect("kilit")
            .iter()
            .map(|(id, m)| (*id, m.clone()))
            .collect();
        let (metadata_file, metadata_crc) = if entries.is_empty() {
            (None, 0)
        } else {
            let bytes = storage::encode_metadata(&entries)?;
            let name = Manifest::metadata_file_name(generation);
            storage::write_file_durable(&dir.join(&name), &bytes)?;
            (Some(name), storage::crc32(&bytes))
        };

        // WAL rotasyonu: buffer mühürlendiği için eski WAL'ın TÜM kayıtları
        // artık segmentlerde. Yeni (boş) dosyayı manifest yazılmadan ÖNCE
        // açarız; manifest yeni adı işaret eder, GC eskisini siler. Kesinti
        // olursa eski manifest hâlâ eski WAL'ı işaret eder — tutarlı.
        let wal_file = {
            let mut guard = self.wal.write().expect("kilit");
            match guard.as_ref() {
                Some(old) => {
                    let policy = old.policy();
                    let name = Manifest::wal_file_name(generation);
                    *guard = Some(Wal::open_append(dir.join(&name), policy)?);
                    Some(name)
                }
                None => None,
            }
        };

        let now = storage::now_unix_secs();
        let manifest = Manifest {
            generation,
            dim: self.dim as u64,
            metric: self.metric,
            hnsw_params: self.hnsw_params.clone(),
            seal_threshold: self.seal_threshold as u64,
            max_segments: self.max_segments as u64,
            segments: refs,
            metadata_file,
            metadata_crc,
            wal_file,
            created_unix_secs: now,
        };
        manifest.write(&dir)?;
        storage::gc_unreferenced(&dir, &manifest)?;
        self.last_checkpoint.store(now, Ordering::Relaxed);
        Ok(generation)
    }

    /// WAL'ı zorla fsync'ler (grup penceresini kapatır). Graceful shutdown ve
    /// yazıcı task'inin batch sonu bunu çağırır.
    pub fn flush_wal(&self) -> Result<(), IndexError> {
        if let Some(w) = self.wal.write().expect("kilit").as_mut() {
            w.sync().map_err(|e| IndexError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    /// Grup penceresi dolduysa fsync'ler; batch sonunda ucuz çağrı.
    pub fn sync_wal_if_due(&self) -> Result<bool, IndexError> {
        if let Some(w) = self.wal.write().expect("kilit").as_mut() {
            return w
                .sync_if_due()
                .map_err(|e| IndexError::Storage(e.to_string()));
        }
        Ok(false)
    }

    /// Batch sonu commit: politikanın vaat ettiği dayanıklılığı sağlar
    /// (None → yalnız OS'e teslim, diğerleri → fsync). Yazıcı task'i HTTP
    /// yanıtlarını göndermeden ÖNCE çağırır; group commit'in
    /// "200 = fsync'lendi" sözleşmesi buna dayanır.
    pub fn commit_wal(&self) -> Result<(), IndexError> {
        if let Some(w) = self.wal.write().expect("kilit").as_mut() {
            w.commit().map_err(|e| IndexError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    /// Aktif WAL'ın bayt boyutu (0 = WAL yok).
    pub fn wal_len_bytes(&self) -> u64 {
        self.wal
            .read()
            .expect("kilit")
            .as_ref()
            .map(|w| w.len_bytes())
            .unwrap_or(0)
    }

    pub fn wal_policy_label(&self) -> Option<String> {
        self.wal
            .read()
            .expect("kilit")
            .as_ref()
            .map(|w| w.policy().label())
    }

    /// Açılışta yapılan WAL replay'inin raporu.
    pub fn replay_report(&self) -> ReplayReport {
        self.replay_report.read().expect("kilit").clone()
    }

    /// Dizinde manifest varsa oradan kurar, yoksa verilen parametrelerle boş
    /// indeks açar. Her iki durumda da dizin bağlanır.
    ///
    /// Manifest varsa dim/metric/params/eşikler ONDAN gelir: diskteki gerçek,
    /// çağıranın varsayımını ezer (yanlış dim ile açıp veriyi bozmak yerine).
    pub fn open_or_create(
        dir: PathBuf,
        dim: usize,
        metric: Metric,
        hnsw_params: HnswParams,
        seal_threshold: usize,
    ) -> Result<Self, StorageError> {
        Self::open_inner(dir, dim, metric, hnsw_params, seal_threshold, None)
    }

    /// WAL'lı açılış: manifest + segmentler yüklenir, ardından WAL replay
    /// edilir ve log append modunda bağlanır. Kurtarma her zaman WAL'la
    /// tamamlanır — manifest tek başına yeterli sayılmaz (DECISIONS #33'teki
    /// Windows dizin-fsync boşluğu).
    pub fn open_durable(
        dir: PathBuf,
        dim: usize,
        metric: Metric,
        hnsw_params: HnswParams,
        seal_threshold: usize,
        policy: SyncPolicy,
    ) -> Result<Self, StorageError> {
        Self::open_inner(dir, dim, metric, hnsw_params, seal_threshold, Some(policy))
    }

    fn open_inner(
        dir: PathBuf,
        dim: usize,
        metric: Metric,
        hnsw_params: HnswParams,
        seal_threshold: usize,
        wal_policy: Option<SyncPolicy>,
    ) -> Result<Self, StorageError> {
        std::fs::create_dir_all(&dir)?;
        let Some(manifest) = Manifest::read(&dir)? else {
            let idx = Self::new(dim, metric, hnsw_params, seal_threshold);
            idx.attach_storage(dir.clone());
            if let Some(policy) = wal_policy {
                // Manifest yok ama WAL olabilir: önceki koşu checkpoint'e
                // ulaşamadan çöktüyse tüm veri oradadır.
                let name = Manifest::wal_file_name(0);
                idx.recover_wal(&dir, &name, policy)?;
            }
            return Ok(idx);
        };
        let mut idx = Self::new(
            manifest.dim as usize,
            manifest.metric,
            manifest.hnsw_params.clone(),
            manifest.seal_threshold as usize,
        );
        idx.set_max_segments(manifest.max_segments as usize);

        let mut segments = Vec::with_capacity(manifest.segments.len());
        for sref in &manifest.segments {
            let bytes = storage::read_verified(&dir, &sref.file, sref.crc32)?;
            let index = HnswIndex::load_from_bytes(&bytes)?;
            let tombstones: HashSet<VectorId> =
                sref.tombstones.iter().map(|&i| VectorId(i)).collect();
            segments.push(Arc::new(Segment {
                index,
                tombstones: RwLock::new(tombstones),
                stored: RwLock::new(Some(StoredFile {
                    name: sref.file.clone(),
                    crc32: sref.crc32,
                })),
            }));
        }
        *idx.segments.write().expect("kilit") = segments;

        if let Some(file) = &manifest.metadata_file {
            let bytes = storage::read_verified(&dir, file, manifest.metadata_crc)?;
            // Türetilmiş yapılar (posting-list'ler, sayısal indeksler) burada
            // yeniden kurulur — diske yazılmadılar, tek kaynak metadata.
            for (id, meta) in storage::decode_metadata(&bytes, &dir.join(file))? {
                idx.index_metadata(id, meta);
            }
        }
        idx.generation.store(manifest.generation, Ordering::Relaxed);
        idx.last_checkpoint
            .store(manifest.created_unix_secs, Ordering::Relaxed);
        idx.attach_storage(dir.clone());
        if let Some(policy) = wal_policy {
            let name = manifest
                .wal_file
                .clone()
                .unwrap_or_else(|| Manifest::wal_file_name(manifest.generation));
            idx.recover_wal(&dir, &name, policy)?;
        }
        Ok(idx)
    }

    /// WAL replay + logu append modunda bağlama.
    ///
    /// Replay sırasında `self.wal` HENÜZ None olduğu için kayıtlar yeniden
    /// yazılmaz; bu, "replay ettiğimi tekrar loglamak" hatasını yapısal
    /// olarak imkânsız kılar.
    fn recover_wal(
        &self,
        dir: &std::path::Path,
        name: &str,
        policy: SyncPolicy,
    ) -> Result<(), StorageError> {
        let path = dir.join(name);
        let (records, report) = wal::replay(&path)?;
        for rec in records {
            match rec {
                WalRecord::Insert { id, vector, meta } => {
                    self.apply_insert(VectorId(id), &vector, wal::record_meta(meta))?;
                }
                WalRecord::Delete { id } => {
                    // Replay'de "yok" durumu rotasyon sınırında meşru olabilir:
                    // sessizce geç, hayalet op üretme.
                    let _ = self.apply_delete(VectorId(id));
                }
            }
        }
        *self.replay_report.write().expect("kilit") = report;
        *self.wal.write().expect("kilit") = Some(Wal::open_append(path, policy)?);
        Ok(())
    }

    /// Toplam indeks belleği (vektör + graf, tüm segmentler; byte).
    pub fn memory_bytes(&self) -> usize {
        self.segments
            .read()
            .expect("kilit")
            .iter()
            .map(|s| {
                let (v, l) = s.index.memory_bytes();
                v + l
            })
            .sum::<usize>()
            + self.buffer.read().expect("kilit").memory_bytes()
    }

    /// Gözlem: (segment sayısı, buffer doluluğu).
    pub fn shape(&self) -> (usize, usize) {
        (
            self.segments.read().expect("kilit").len(),
            self.buffer.read().expect("kilit").len(),
        )
    }
}

/// Trait uyumu: tek thread'li kullanım için &mut imzalar paylaşımlı
/// implementasyona delege eder.
impl VectorIndex for SegmentedIndex {
    fn insert(&mut self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        self.insert_shared(id, vector)
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.search_shared(query, k)
    }

    fn delete(&mut self, id: VectorId) -> Result<(), IndexError> {
        self.delete_shared(id)
    }

    fn len(&self) -> usize {
        self.len_shared()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::random_vectors;
    use crate::eval::exact_top_k;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn build(vecs: &[Vec<f32>], seal_threshold: usize) -> SegmentedIndex {
        let idx = SegmentedIndex::new(
            vecs[0].len(),
            Metric::L2,
            HnswParams::default(),
            seal_threshold,
        );
        for (i, v) in vecs.iter().enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        idx
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 100);
        assert!(idx.is_empty());
        assert!(idx.search(&[0.0; 4], 5).is_empty());
    }

    #[test]
    fn single_element_and_k_larger_than_len() {
        let idx = build(&[vec![1.0, 2.0]], 100);
        let res = idx.search(&[0.0, 0.0], 10);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId(0));
    }

    #[test]
    fn spans_multiple_segments_and_buffer() {
        let vecs = random_vectors(1_050, 8, 42);
        let idx = build(&vecs, 400); // 2 segment (400+400) + 250 buffer
        let (n_seg, n_buf) = idx.shape();
        assert_eq!(n_seg, 2);
        assert_eq!(n_buf, 250);
        assert_eq!(idx.len(), 1_050);
        // doğruluk: exact referansla yüksek örtüşme
        let queries = random_vectors(20, 8, 43);
        let mut hits = 0;
        for q in &queries {
            let truth: Vec<_> = exact_top_k(&vecs, q, 10, Metric::L2)
                .iter()
                .map(|r| r.id)
                .collect();
            hits += idx
                .search(q, 10)
                .iter()
                .filter(|r| truth.contains(&r.id))
                .count();
        }
        let recall = hits as f64 / (20 * 10) as f64;
        assert!(recall >= 0.95, "segmentli recall {recall}");
    }

    #[test]
    fn duplicate_id_rejected_across_sources() {
        let vecs = random_vectors(500, 4, 42);
        let idx = build(&vecs, 300); // id 0..299 segmentte, 300.. buffer'da
                                     // segmentteki id
        assert!(matches!(
            idx.insert_shared(VectorId(5), &[0.0; 4]),
            Err(IndexError::DuplicateId(_))
        ));
        // buffer'daki id
        assert!(matches!(
            idx.insert_shared(VectorId(450), &[0.0; 4]),
            Err(IndexError::DuplicateId(_))
        ));
    }

    #[test]
    fn delete_from_buffer_and_segment() {
        let vecs = random_vectors(500, 4, 42);
        let idx = build(&vecs, 300);
        // buffer'dan gerçek silme
        idx.delete_shared(VectorId(400)).unwrap();
        // segmentten tombstone silme
        idx.delete_shared(VectorId(5)).unwrap();
        assert_eq!(idx.len(), 498);
        for q in random_vectors(10, 4, 43) {
            let ids: Vec<_> = idx.search(&q, 498).iter().map(|r| r.id).collect();
            assert!(!ids.contains(&VectorId(400)) && !ids.contains(&VectorId(5)));
        }
        assert!(matches!(
            idx.delete_shared(VectorId(5)),
            Err(IndexError::NotFound(_))
        ));
    }

    #[test]
    fn reinsert_after_segment_delete_returns_new_vector() {
        let vecs = random_vectors(300, 4, 42);
        let idx = build(&vecs, 200); // id 5 segmentte
        idx.delete_shared(VectorId(5)).unwrap();
        idx.insert_shared(VectorId(5), &[9.0; 4]).unwrap();
        assert_eq!(idx.len(), 300);
        // yeni vektör bulunur, eski kopya gölgede kalır
        let res = idx.search(&[9.0; 4], 1);
        assert_eq!(res[0].id, VectorId(5));
        let old = idx.search(&vecs[5].clone(), 3);
        // eski konumda id 5 dönerse bu hortlamış eski kopyadır — dönmemeli
        // (yeni vektör [9;4] eski konumdan uzak)
        assert!(old.iter().all(|r| r.id != VectorId(5)));
    }

    #[test]
    fn zero_vector_and_duplicate_vectors() {
        let idx = SegmentedIndex::new(3, Metric::Cosine, HnswParams::default(), 10);
        idx.insert_shared(VectorId(0), &[0.0; 3]).unwrap();
        idx.insert_shared(VectorId(1), &[1.0, 0.0, 0.0]).unwrap();
        idx.insert_shared(VectorId(2), &[1.0, 0.0, 0.0]).unwrap();
        let res = idx.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(res.len(), 3);
        assert!(res.iter().all(|r| !r.distance.is_nan()));
        assert!(res[..2].iter().all(|r| r.id != VectorId(0)));
    }

    // ---- Metadata filtreleme testleri ----

    use crate::meta::{MetaValue, Predicate};

    /// Çift id'li kayıtlar iki kategoriye bölünür; filtreli arama yalnız
    /// istenen kategoriyi döndürmeli ve brute-force filtreli referansla örtüşmeli.
    #[test]
    fn filtered_search_matches_reference() {
        let vecs = random_vectors(1_000, 8, 42);
        let idx = SegmentedIndex::new(8, Metric::L2, HnswParams::default(), 400);
        let mut bf = BruteForceIndex::new(8, Metric::L2);
        for (i, v) in vecs.iter().enumerate() {
            let meta: Metadata = [(
                "grup".to_string(),
                MetaValue::Str(if i % 2 == 0 { "çift" } else { "tek" }.into()),
            )]
            .into();
            idx.insert_with_meta(VectorId(i as u64), v, meta).unwrap();
            bf.insert(VectorId(i as u64), v).unwrap();
        }
        let filter = Filter {
            must: vec![Predicate::Eq {
                key: "grup".into(),
                value: MetaValue::Str("çift".into()),
            }],
        };
        let allow = |id: VectorId| id.0.is_multiple_of(2);
        let queries = random_vectors(20, 8, 43);
        let mut hits = 0;
        for q in &queries {
            let res = idx.search_filtered(q, 10, &filter);
            assert_eq!(res.len(), 10);
            assert!(
                res.iter().all(|r| r.id.0.is_multiple_of(2)),
                "filtre kaçağı"
            );
            let truth: Vec<_> = bf
                .search_filtered(q, 10, &allow)
                .iter()
                .map(|r| r.id)
                .collect();
            hits += res.iter().filter(|r| truth.contains(&r.id)).count();
        }
        assert!(
            hits as f64 / 200.0 >= 0.95,
            "filtreli recall düşük: {hits}/200"
        );
    }

    /// Aşırı seçici filtre (tek eşleşme): fallback doğrusal tarama devreye
    /// girmeli ve o tek kaydı bulmalı.
    #[test]
    fn highly_selective_filter_falls_back_to_exact() {
        let vecs = random_vectors(2_000, 8, 42);
        let idx = SegmentedIndex::new(8, Metric::L2, HnswParams::default(), 500);
        for (i, v) in vecs.iter().enumerate() {
            let meta: Metadata = [("nadir".to_string(), MetaValue::Bool(i == 1_234))].into();
            idx.insert_with_meta(VectorId(i as u64), v, meta).unwrap();
        }
        let filter = Filter {
            must: vec![Predicate::Eq {
                key: "nadir".into(),
                value: MetaValue::Bool(true),
            }],
        };
        let res = idx.search_filtered(&random_vectors(1, 8, 43)[0], 5, &filter);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId(1_234));
        // hiç eşleşme yoksa boş dönmeli
        let none = Filter {
            must: vec![Predicate::Eq {
                key: "yok".into(),
                value: MetaValue::Bool(true),
            }],
        };
        assert!(idx.search_filtered(&vecs[0].clone(), 5, &none).is_empty());
    }

    /// Silinen kaydın metadata'sı düşer; aynı id yeni metadata ile dönebilir.
    #[test]
    fn delete_drops_metadata_reinsert_gets_fresh() {
        let vecs = random_vectors(100, 4, 42);
        let idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 1_000);
        for (i, v) in vecs.iter().enumerate() {
            let meta: Metadata = [("v".to_string(), MetaValue::Int(1))].into();
            idx.insert_with_meta(VectorId(i as u64), v, meta).unwrap();
        }
        idx.delete_shared(VectorId(7)).unwrap();
        idx.insert_with_meta(
            VectorId(7),
            &vecs[7],
            [("v".to_string(), MetaValue::Int(2))].into(),
        )
        .unwrap();
        let f_v2 = Filter {
            must: vec![Predicate::Eq {
                key: "v".into(),
                value: MetaValue::Int(2),
            }],
        };
        let res = idx.search_filtered(&vecs[7].clone(), 1, &f_v2);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId(7));
    }

    // ---- Kalıcılık: soğuk yol testleri (Aşama 7a) ----

    fn meta_of(i: usize) -> Metadata {
        [
            ("grup".to_string(), MetaValue::Int((i % 4) as i64)),
            ("v".to_string(), MetaValue::Int(i as i64)),
        ]
        .into()
    }

    #[test]
    fn checkpoint_reopen_gives_identical_results() {
        let dir = crate::storage::temp_dir("ckpt-roundtrip");
        let vecs = random_vectors(800, 8, 42);
        let queries = random_vectors(15, 8, 43);
        let gen = {
            let idx = SegmentedIndex::open_or_create(
                dir.clone(),
                8,
                Metric::L2,
                HnswParams::default(),
                200,
            )
            .unwrap();
            for (i, v) in vecs.iter().enumerate() {
                idx.insert_with_meta(VectorId(i as u64), v, meta_of(i))
                    .unwrap();
            }
            let before: Vec<Vec<SearchResult>> =
                queries.iter().map(|q| idx.search_shared(q, 10)).collect();
            let gen = idx.checkpoint().unwrap();
            // checkpoint mühürlemesi sonuçları değiştirmemeli
            for (q, b) in queries.iter().zip(&before) {
                assert_eq!(&idx.search_shared(q, 10), b, "checkpoint sonuçları bozdu");
            }
            gen
        };
        // yeniden aç
        let idx = SegmentedIndex::open_or_create(
            dir.clone(),
            999, // yanlış dim: manifest'teki kazanmalı
            Metric::Dot,
            HnswParams::default(),
            1,
        )
        .unwrap();
        assert_eq!(idx.generation(), gen);
        assert_eq!(idx.len(), 800);
        assert_eq!(idx.shape().1, 0, "checkpoint sonrası buffer boş olmalı");
        // Aynı sorgular birebir aynı sonuçları vermeli
        let fresh = SegmentedIndex::new(8, Metric::L2, HnswParams::default(), 200);
        for (i, v) in vecs.iter().enumerate() {
            fresh
                .insert_with_meta(VectorId(i as u64), v, meta_of(i))
                .unwrap();
        }
        fresh.seal();
        for q in &queries {
            assert_eq!(idx.search_shared(q, 10), fresh.search_shared(q, 10));
        }
        // türetilmiş indeksler yeniden kurulmuş olmalı: Eq + Range filtreleri
        let f_eq = Filter {
            must: vec![Predicate::Eq {
                key: "grup".into(),
                value: MetaValue::Int(2),
            }],
        };
        let res = idx.search_filtered(&queries[0], 10, &f_eq);
        assert_eq!(res.len(), 10);
        assert!(
            res.iter().all(|r| r.id.0 % 4 == 2),
            "Eq posting yeniden kurulmadı"
        );
        let f_range = Filter {
            must: vec![Predicate::Range {
                key: "v".into(),
                min: 0.0,
                max: 49.0,
            }],
        };
        let res = idx.search_filtered(&queries[0], 10, &f_range);
        assert_eq!(
            idx.debug_plan_arm(&f_range, 10),
            "scan",
            "numeric indeks yok"
        );
        assert!(res.iter().all(|r| r.id.0 < 50));
    }

    #[test]
    fn segments_written_once_across_checkpoints() {
        let dir = crate::storage::temp_dir("ckpt-once");
        let vecs = random_vectors(500, 4, 42);
        let idx =
            SegmentedIndex::open_or_create(dir.clone(), 4, Metric::L2, HnswParams::default(), 200)
                .unwrap();
        for (i, v) in vecs.iter().take(400).enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        let g1 = idx.checkpoint().unwrap();
        let first: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let n = e.unwrap().file_name().to_string_lossy().to_string();
                n.starts_with("segment-").then_some(n)
            })
            .collect();
        // ikinci tur: yeni veri → yeni segment, eskiler AYNI dosyada kalmalı
        for (i, v) in vecs.iter().enumerate().skip(400) {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        let g2 = idx.checkpoint().unwrap();
        assert!(g2 > g1);
        let second: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let n = e.unwrap().file_name().to_string_lossy().to_string();
                n.starts_with("segment-").then_some(n)
            })
            .collect();
        for f in &first {
            assert!(
                second.contains(f),
                "eski segment yeniden yazıldı/silindi: {f} (değişmezlik ihlali)"
            );
        }
        assert!(second.len() > first.len(), "yeni segment yazılmadı");
    }

    #[test]
    fn reopen_restores_tombstones() {
        let dir = crate::storage::temp_dir("ckpt-tombstone");
        let vecs = random_vectors(400, 4, 42);
        {
            let idx = SegmentedIndex::open_or_create(
                dir.clone(),
                4,
                Metric::L2,
                HnswParams::default(),
                150,
            )
            .unwrap();
            for (i, v) in vecs.iter().enumerate() {
                idx.insert_with_meta(VectorId(i as u64), v, meta_of(i))
                    .unwrap();
            }
            idx.checkpoint().unwrap(); // hepsi segmentlerde
            idx.delete_shared(VectorId(7)).unwrap();
            idx.delete_shared(VectorId(200)).unwrap();
            idx.checkpoint().unwrap(); // tombstone'lar manifest'e
        }
        let idx =
            SegmentedIndex::open_or_create(dir, 4, Metric::L2, HnswParams::default(), 150).unwrap();
        assert_eq!(idx.len(), 398);
        let all: Vec<_> = idx
            .search_shared(&vecs[7].clone(), 400)
            .iter()
            .map(|r| r.id)
            .collect();
        assert!(!all.contains(&VectorId(7)), "tombstone kurtarılmadı");
        assert!(!all.contains(&VectorId(200)));
        // silinen metadata da geri gelmemeli
        let f = Filter {
            must: vec![Predicate::Eq {
                key: "v".into(),
                value: MetaValue::Int(7),
            }],
        };
        assert!(idx.search_filtered(&vecs[7].clone(), 5, &f).is_empty());
    }

    #[test]
    fn corrupt_segment_file_is_error_not_panic() {
        let dir = crate::storage::temp_dir("ckpt-corrupt");
        let vecs = random_vectors(300, 4, 42);
        {
            let idx = SegmentedIndex::open_or_create(
                dir.clone(),
                4,
                Metric::L2,
                HnswParams::default(),
                100,
            )
            .unwrap();
            for (i, v) in vecs.iter().enumerate() {
                idx.insert_shared(VectorId(i as u64), v).unwrap();
            }
            idx.checkpoint().unwrap();
        }
        let seg_file = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("segment-"))
            })
            .expect("segment dosyası");
        let mut bytes = std::fs::read(&seg_file).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xff;
        std::fs::write(&seg_file, &bytes).unwrap();
        let err =
            SegmentedIndex::open_or_create(dir.clone(), 4, Metric::L2, HnswParams::default(), 100);
        assert!(err.is_err(), "bozuk segment yakalanmalıydı");
        // kesik dosya da panic üretmemeli
        std::fs::write(&seg_file, &bytes[..bytes.len() / 3]).unwrap();
        assert!(
            SegmentedIndex::open_or_create(dir, 4, Metric::L2, HnswParams::default(), 100).is_err()
        );
    }

    #[test]
    fn merge_output_persists_and_gc_cleans_sources() {
        let dir = crate::storage::temp_dir("ckpt-merge");
        let vecs = random_vectors(1_200, 4, 42);
        let (before_len, gen) = {
            let mut idx = SegmentedIndex::open_or_create(
                dir.clone(),
                4,
                Metric::L2,
                HnswParams::default(),
                100,
            )
            .unwrap();
            idx.set_max_segments(4);
            for (i, v) in vecs.iter().enumerate() {
                idx.insert_shared(VectorId(i as u64), v).unwrap();
            }
            idx.checkpoint().unwrap();
            // merge'ler tavanı korumuş olmalı
            assert!(idx.shape().0 <= 4);
            let g = idx.checkpoint().unwrap();
            (idx.len(), g)
        };
        // GC: manifest'te olmayan segment dosyası kalmamalı
        let manifest = Manifest::read(&dir).unwrap().unwrap();
        assert_eq!(manifest.generation, gen);
        let on_disk: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let n = e.unwrap().file_name().to_string_lossy().to_string();
                n.starts_with("segment-").then_some(n)
            })
            .collect();
        assert_eq!(
            on_disk.len(),
            manifest.segments.len(),
            "yetim segment dosyası kaldı: {on_disk:?}"
        );
        let idx =
            SegmentedIndex::open_or_create(dir, 4, Metric::L2, HnswParams::default(), 100).unwrap();
        assert_eq!(idx.len(), before_len);
    }

    #[test]
    fn checkpoint_without_storage_dir_is_error() {
        let idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 100);
        assert!(idx.checkpoint().is_err());
    }

    // ---- Segment tavanı / merge testleri ----

    #[test]
    fn merge_guard_enforces_ceiling() {
        let vecs = random_vectors(1_200, 8, 42);
        let mut idx = SegmentedIndex::new(8, Metric::L2, HnswParams::default(), 100);
        idx.set_max_segments(4);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        let (n_seg, _) = idx.shape();
        assert!(n_seg <= 4, "tavan aşıldı: {n_seg}");
        assert_eq!(idx.len(), 1_200, "merge kayıt kaybetti");
        // doğruluk: exact referansla örtüşme
        let queries = random_vectors(20, 8, 43);
        let mut hits = 0;
        for q in &queries {
            let truth: Vec<_> = exact_top_k(&vecs, q, 10, Metric::L2)
                .iter()
                .map(|r| r.id)
                .collect();
            hits += idx
                .search_shared(q, 10)
                .iter()
                .filter(|r| truth.contains(&r.id))
                .count();
        }
        assert!(
            hits as f64 / 200.0 >= 0.95,
            "merge sonrası recall: {hits}/200"
        );
    }

    #[test]
    fn merge_drops_tombstones_and_preserves_reinserts() {
        let vecs = random_vectors(600, 4, 42);
        let mut idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 100);
        idx.set_max_segments(3);
        for (i, v) in vecs.iter().take(300).enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        // segmentlere düşmüş kayıtlardan sil + birini yeni vektörle geri ekle
        idx.delete_shared(VectorId(5)).unwrap();
        idx.delete_shared(VectorId(50)).unwrap();
        idx.insert_shared(VectorId(5), &[9.0; 4]).unwrap();
        // tavanı zorlayacak kadar ekle → merge'ler tetiklenir
        for (i, v) in vecs.iter().enumerate().skip(300) {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        let (n_seg, _) = idx.shape();
        assert!(n_seg <= 3);
        assert_eq!(idx.len(), 599); // 600 - 1 kalıcı silme
                                    // silinen id dönmüyor, yeniden eklenen yeni vektörüyle dönüyor
        let all: Vec<_> = idx
            .search_shared(&[9.0; 4], 599)
            .iter()
            .map(|r| r.id)
            .collect();
        assert!(!all.contains(&VectorId(50)));
        assert_eq!(idx.search_shared(&[9.0; 4], 1)[0].id, VectorId(5));
    }

    /// Kabul kriteri (DECISIONS #31): Range'li sorgularda seçilen kol, gerçek
    /// kardinaliteyle seçilecek kolla örtüşmeli. Sınırlı sayım tasarımı bunu
    /// scan sınırında yapısal olarak garanti eder; test yine de belgeler.
    #[test]
    fn range_planner_arm_matches_oracle() {
        let vecs = random_vectors(2_000, 4, 42);
        let idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 500);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert_with_meta(
                VectorId(i as u64),
                v,
                [("v".to_string(), MetaValue::Int(i as i64))].into(),
            )
            .unwrap();
        }
        let k = 10;
        let n = 2_000usize;
        let scan_limit =
            (idx.planner.scan_factor * k).max((idx.planner.scan_fraction * n as f64) as usize);
        for m in [1usize, 50, 160, 161, 300, 800, 1500] {
            let f = Filter {
                must: vec![Predicate::Range {
                    key: "v".into(),
                    min: 0.0,
                    max: (m - 1) as f64,
                }],
            };
            let oracle = if m <= scan_limit { "scan" } else { "post" };
            assert_eq!(idx.debug_plan_arm(&f, k), oracle, "m={m}");
            // ve sonuçlar exact referansla doğru
            let allow = |id: VectorId| (id.0 as usize) < m;
            let q = &vecs[m / 2];
            let truth: Vec<_> = {
                let mut bf = BruteForceIndex::new(4, Metric::L2);
                for (i, v) in vecs.iter().enumerate() {
                    bf.insert(VectorId(i as u64), v).unwrap();
                }
                bf.search_filtered(q, k, &allow)
                    .iter()
                    .map(|r| r.id)
                    .collect()
            };
            let got: Vec<_> = idx.search_filtered(q, k, &f).iter().map(|r| r.id).collect();
            let hit = got.iter().filter(|id| truth.contains(id)).count();
            assert!(hit * 10 >= truth.len() * 9, "m={m}: {hit}/{}", truth.len());
        }
        // Range'de sıfır eşleşme → boş
        let f_empty = Filter {
            must: vec![Predicate::Range {
                key: "v".into(),
                min: 1e9,
                max: 2e9,
            }],
        };
        assert!(idx
            .search_filtered(&vecs[0].clone(), k, &f_empty)
            .is_empty());
    }

    /// Posting-list'ler her mutasyon dizisinden sonra metadata deposuyla
    /// birebir tutarlı olmalı (planlayıcının tahmini buna dayanıyor).
    #[test]
    fn postings_consistent_after_insert_delete_reinsert() {
        let vecs = random_vectors(200, 4, 42);
        let idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 80);
        for (i, v) in vecs.iter().enumerate() {
            let meta: Metadata = [("g".to_string(), MetaValue::Int((i % 5) as i64))].into();
            idx.insert_with_meta(VectorId(i as u64), v, meta).unwrap();
        }
        for i in (0..200).step_by(3) {
            idx.delete_shared(VectorId(i)).unwrap();
        }
        // birkaç yeniden ekleme, farklı grupla
        for i in (0..30).step_by(3) {
            idx.insert_with_meta(
                VectorId(i),
                &vecs[i as usize],
                [("g".to_string(), MetaValue::Int(99))].into(),
            )
            .unwrap();
        }
        // yeniden say ve karşılaştır
        let meta_store = idx.metadata.read().unwrap();
        let postings = idx.postings.read().unwrap();
        for ((key, mk), set) in postings.iter() {
            let recount: HashSet<VectorId> = meta_store
                .iter()
                .filter(|(_, m)| m.get(key).is_some_and(|v| v.key() == *mk))
                .map(|(&id, _)| id)
                .collect();
            assert_eq!(*set, recount, "posting tutarsız: {key}/{mk:?}");
        }
        // tahmin, gerçek eşleşme sayısına eşit (tek Eq'de kesin)
        let f = Filter {
            must: vec![Predicate::Eq {
                key: "g".into(),
                value: MetaValue::Int(99),
            }],
        };
        let (est, cands) = idx.estimate(&f).unwrap();
        assert_eq!(est, 10);
        assert_eq!(cands.len(), 10);
    }

    /// Stres testi: çok okuyucu + tek yazıcı. Yazıcı insert+delete yaparken
    /// okuyucular sürekli arar; hiçbir panic olmamalı ve sonuçlar temel
    /// tutarlılık kurallarına uymalı (dup id yok, NaN yok, k aşımı yok).
    #[test]
    fn stress_concurrent_readers_single_writer() {
        let dim = 16;
        let idx = Arc::new(SegmentedIndex::new(
            dim,
            Metric::L2,
            HnswParams {
                ef_construction: 40, // stres testinde inşa hızı > graf kalitesi
                ..Default::default()
            },
            500,
        ));
        let vecs = random_vectors(4_000, dim, 42);
        // başlangıç yükü
        for (i, v) in vecs.iter().take(1_000).enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        let stop = AtomicBool::new(false);
        let queries = random_vectors(50, dim, 43);

        std::thread::scope(|s| {
            // 4 okuyucu
            for t in 0..4 {
                let idx = &idx;
                let stop = &stop;
                let queries = &queries;
                s.spawn(move || {
                    let mut iters = 0usize;
                    while !stop.load(Ordering::Relaxed) {
                        let q = &queries[(iters + t) % queries.len()];
                        let res = idx.search_shared(q, 10);
                        assert!(res.len() <= 10);
                        let mut seen = HashSet::new();
                        for r in &res {
                            assert!(!r.distance.is_nan());
                            assert!(seen.insert(r.id), "duplicate id sonuçta");
                        }
                        // sonuçlar artan mesafeli olmalı
                        for w in res.windows(2) {
                            assert!(w[0].distance <= w[1].distance);
                        }
                        iters += 1;
                    }
                    assert!(iters > 0);
                });
            }
            // tek yazıcı: 3.000 insert (5+ mühürleme tetikler) + aralıklı silme
            for (i, v) in vecs.iter().enumerate().skip(1_000) {
                idx.insert_shared(VectorId(i as u64), v).unwrap();
                if i % 7 == 0 {
                    // daha önce eklenmiş bir id'yi sil
                    let victim = VectorId((i / 2) as u64);
                    let _ = idx.delete_shared(victim); // zaten silinmişse NotFound: sorun değil
                }
            }
            stop.store(true, Ordering::Relaxed);
        });

        let (n_seg, _) = idx.shape();
        assert!(n_seg >= 5, "mühürleme hiç tetiklenmemiş: {n_seg}");
        // yazıcı bittikten sonra arama deterministik ve sağlıklı
        let res = idx.search_shared(&queries[0], 10);
        assert_eq!(res.len(), 10);
    }
}
