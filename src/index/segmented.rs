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
use crate::index::{IndexError, VectorIndex};
use crate::meta::{Filter, Metadata};
use crate::types::{SearchResult, VectorId};
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

/// Mühürlü, immutable HNSW + kendi tombstone kümesi.
///
/// Tombstone'lar segment-YEREL: bir id silinip yeniden eklendiğinde eski
/// kopya kendi segmentinde sonsuza dek gölgede kalır, yeni kopya buffer'da
/// (sonra başka segmentte) yaşar — global bir silinmiş-küme olsaydı yeniden
/// ekleme eski kopyayı hortlatırdı.
struct Segment {
    index: HnswIndex,
    tombstones: RwLock<HashSet<VectorId>>,
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
        }
    }

    /// Metadata'lı insert. Metadata'sız `insert_shared` boş geçer.
    pub fn insert_with_meta(
        &self,
        id: VectorId,
        vector: &[f32],
        meta: Metadata,
    ) -> Result<(), IndexError> {
        self.insert_shared(id, vector)?;
        if !meta.is_empty() {
            self.metadata.write().expect("kilit").insert(id, meta);
        }
        Ok(())
    }

    /// Filtreli arama: her segmentte gezinti-içi filtre + buffer'da filtreli
    /// tarama, sonuçlar id bazında birleştirilir.
    pub fn search_filtered(&self, query: &[f32], k: usize, filter: &Filter) -> Vec<SearchResult> {
        if filter.must.is_empty() {
            return self.search_shared(query, k);
        }
        if k == 0 {
            return Vec::new();
        }
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

    /// Paylaşımlı (&self) insert — tek yazıcı thread'inden çağrılmalı.
    /// Birden çok yazıcı data race üretmez (her şey kilitli) ama duplicate id
    /// kontrolü iki yazıcı arasında yarışabilir; sözleşme tek yazıcıdır.
    pub fn insert_shared(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        if vector.len() != self.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        // Duplicate kontrolü: buffer'da ya da herhangi bir segmentte (tombstone'suz) yaşıyor mu?
        {
            let buffer = self.buffer.read().expect("kilit zehirlenmedi");
            if buffer.contains(id) {
                return Err(IndexError::DuplicateId(id));
            }
        }
        {
            let segments = self.segments.read().expect("kilit");
            for seg in segments.iter() {
                if seg.index.contains(id) && !seg.tombstones.read().expect("kilit").contains(&id) {
                    return Err(IndexError::DuplicateId(id));
                }
            }
        }
        let should_seal = {
            let mut buffer = self.buffer.write().expect("kilit");
            buffer.insert(id, vector)?;
            buffer.len() >= self.seal_threshold
        }; // write kilidi burada düşer; mühürleme kilitsiz çalışacak
        if should_seal {
            self.seal();
        }
        Ok(())
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
        });
        // 3. Önce segmenti yayınla, SONRA buffer'ı boşalt. Aradaki pencerede
        // kopya görünür (id bazlı dedupe emer); ters sıra veri kaybettirirdi.
        self.segments.write().expect("kilit").push(segment);
        let mut buffer = self.buffer.write().expect("kilit");
        // Tek yazıcı sözleşmesi: seal ile bu satır arasında insert olamaz,
        // buffer içeriği hâlâ `entries` ile birebir aynı — komple sıfırla.
        *buffer = BruteForceIndex::new(self.dim, self.metric);
    }

    /// Paylaşımlı (&self) silme — tek yazıcı thread'inden.
    /// Metadata da düşürülür (yeniden eklemede eski metadata sızmasın).
    pub fn delete_shared(&self, id: VectorId) -> Result<(), IndexError> {
        let res = self.delete_vector_only(id);
        if res.is_ok() {
            self.metadata.write().expect("kilit").remove(&id);
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
