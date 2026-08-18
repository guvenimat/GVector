//! Kaza kurtarma matrisi (Aşama 7c).
//!
//! Süreç öldürmek yerine **deterministik kesme**: gerçek bir indekse op
//! dizisi uygulanır, WAL dosyası kayıt sınırında VE kayıt ortasında budanır,
//! indeks yeniden açılır. Taşınabilir (Windows dahil), tekrarlanabilir ve
//! kesme noktası tam olarak kontrol edilebilir.
//!
//! Doğruluk ölçütü: kurtarılan durum, WAL'ın **sağlam önekinin** durumuna
//! eşit olmalı — ne eksik (kayıp op) ne fazla (hayalet op).

use proptest::prelude::*;
use std::collections::HashSet;
use vector_gvector::dataset::random_vectors;
use vector_gvector::distance::Metric;
use vector_gvector::index::hnsw::HnswParams;
use vector_gvector::index::segmented::SegmentedIndex;
use vector_gvector::index::VectorIndex;
use vector_gvector::meta::{MetaValue, Metadata};
use vector_gvector::storage::wal::{replay_bytes, SyncPolicy, WalRecord};
use vector_gvector::storage::{temp_dir, Manifest};
use vector_gvector::types::VectorId;

const DIM: usize = 8;

fn meta_for(i: u64) -> Metadata {
    [("v".to_string(), MetaValue::Int(i as i64))].into()
}

/// Sağlam önekten beklenen canlı id kümesi. WAL'da yalnız BAŞARILI
/// mutasyonlar bulunur (validasyon append'ten önce), bu yüzden düz uygulama
/// doğru referansı verir.
fn expected_live(records: &[WalRecord], base: HashSet<u64>) -> HashSet<u64> {
    let mut s = base;
    for r in records {
        match r {
            WalRecord::Insert { id, .. } => {
                s.insert(*id);
            }
            WalRecord::Delete { id } => {
                s.remove(id);
            }
        }
    }
    s
}

fn live_ids(idx: &SegmentedIndex, probe: &[f32], n: usize) -> HashSet<u64> {
    idx.search_shared(probe, n).iter().map(|r| r.id.0).collect()
}

/// Op dizisini uygular ve WAL dosya yolunu döndürür.
fn build_wal(dir: &std::path::Path, ops: &[(bool, u64)], vecs: &[Vec<f32>]) -> std::path::PathBuf {
    let idx = SegmentedIndex::open_durable(
        dir.to_path_buf(),
        DIM,
        Metric::L2,
        HnswParams::default(),
        10_000, // mühürleme olmasın: tüm veri WAL + buffer'da kalsın
        SyncPolicy::PerOp,
    )
    .expect("open");
    for (is_insert, id) in ops {
        if *is_insert {
            let _ = idx.insert_with_meta(
                VectorId(*id),
                &vecs[(*id as usize) % vecs.len()],
                meta_for(*id),
            );
        } else {
            let _ = idx.delete_shared(VectorId(*id));
        }
    }
    idx.flush_wal().expect("flush");
    drop(idx);
    let manifest = Manifest::read(dir).expect("manifest okuma");
    let name = manifest
        .and_then(|m| m.wal_file)
        .unwrap_or_else(|| Manifest::wal_file_name(0));
    dir.join(name)
}

/// Kesilmiş WAL ile yeniden açıp durumu sağlam önekle karşılaştırır.
fn assert_recovers_to_prefix(dir: &std::path::Path, wal_path: &std::path::Path, cut: usize) {
    let full = std::fs::read(wal_path).expect("wal oku");
    let cut = cut.min(full.len());
    std::fs::write(wal_path, &full[..cut]).expect("kes");
    let (prefix_records, _) = replay_bytes(&full[..cut]);
    let expected = expected_live(&prefix_records, HashSet::new());

    let idx = SegmentedIndex::open_durable(
        dir.to_path_buf(),
        DIM,
        Metric::L2,
        HnswParams::default(),
        10_000,
        SyncPolicy::PerOp,
    )
    .expect("kesik WAL ile açılış hata vermemeli");
    let probe = vec![0.0f32; DIM];
    let got = live_ids(&idx, &probe, 10_000);
    assert_eq!(
        got, expected,
        "kesme={cut}: kurtarılan durum sağlam önekten farklı"
    );
    assert_eq!(idx.len(), expected.len());
}

#[test]
fn crash_matrix_record_boundaries_and_midpoints() {
    let vecs = random_vectors(64, DIM, 42);
    // insert/delete karışık, yeniden ekleme zincirleri dahil
    let ops: Vec<(bool, u64)> = vec![
        (true, 1),
        (true, 2),
        (true, 3),
        (false, 2),
        (true, 4),
        (true, 2), // silineni yeniden ekle
        (false, 1),
        (true, 5),
    ];
    let dir = temp_dir("crash-matrix");
    let wal_path = build_wal(&dir, &ops, &vecs);
    let full = std::fs::read(&wal_path).expect("wal");

    // Kayıt sınırlarını çıkar
    let mut boundaries = vec![0usize];
    let mut off = 0usize;
    while off + 8 <= full.len() {
        let len = u32::from_le_bytes(full[off..off + 4].try_into().unwrap()) as usize;
        off += 8 + len;
        if off <= full.len() {
            boundaries.push(off);
        }
    }
    assert!(boundaries.len() >= 5, "yeterli kayıt yok");

    // Her sınırda ve her sınırın ortasında kes
    let mut cuts: Vec<usize> = boundaries.clone();
    for w in boundaries.windows(2) {
        cuts.push((w[0] + w[1]) / 2); // kayıt ortası
        cuts.push(w[1] - 1); // son bayt eksik
        cuts.push(w[0] + 3); // başlık ortası
    }
    for cut in cuts {
        // her denemede temiz dizinle başla
        let d = temp_dir("crash-cut");
        let wp = build_wal(&d, &ops, &vecs);
        assert_recovers_to_prefix(&d, &wp, cut);
    }
}

#[test]
fn crash_after_checkpoint_keeps_segments_and_wal_prefix() {
    let vecs = random_vectors(64, DIM, 7);
    let dir = temp_dir("crash-after-ckpt");
    // checkpoint'e kadar olan veri segmentlerde, sonrası WAL'da
    let (wal_path, base_ids) = {
        let idx = SegmentedIndex::open_durable(
            dir.clone(),
            DIM,
            Metric::L2,
            HnswParams::default(),
            10_000,
            SyncPolicy::PerOp,
        )
        .unwrap();
        for id in 0..20u64 {
            idx.insert_with_meta(VectorId(id), &vecs[id as usize % 64], meta_for(id))
                .unwrap();
        }
        let gen = idx.checkpoint().unwrap();
        // checkpoint SONRASI ops → yeni WAL dosyasında
        for id in 20..30u64 {
            idx.insert_with_meta(VectorId(id), &vecs[id as usize % 64], meta_for(id))
                .unwrap();
        }
        idx.delete_shared(VectorId(3)).unwrap(); // segmentteki kaydı sil
        idx.flush_wal().unwrap();
        let base: HashSet<u64> = (0..20).collect();
        (dir.join(Manifest::wal_file_name(gen)), base)
    };
    let full = std::fs::read(&wal_path).unwrap();
    for cut in [
        0,
        full.len() / 3,
        full.len() / 2,
        full.len() - 2,
        full.len(),
    ] {
        std::fs::write(&wal_path, &full[..cut]).unwrap();
        let (recs, _) = replay_bytes(&full[..cut]);
        let expected = expected_live(&recs, base_ids.clone());
        let idx = SegmentedIndex::open_durable(
            dir.clone(),
            DIM,
            Metric::L2,
            HnswParams::default(),
            10_000,
            SyncPolicy::PerOp,
        )
        .expect("açılış");
        let got = live_ids(&idx, &[0.0; DIM], 1000);
        assert_eq!(got, expected, "checkpoint+kesme={cut}");
    }
}

#[test]
fn corrupt_wal_body_recovers_prefix_and_truncates() {
    let vecs = random_vectors(32, DIM, 9);
    let ops: Vec<(bool, u64)> = (1..=6).map(|i| (true, i)).collect();
    let dir = temp_dir("crash-corrupt");
    let wal_path = build_wal(&dir, &ops, &vecs);
    let full = std::fs::read(&wal_path).unwrap();
    // 3. kaydın gövdesini boz
    let mut off = 0usize;
    for _ in 0..2 {
        let len = u32::from_le_bytes(full[off..off + 4].try_into().unwrap()) as usize;
        off += 8 + len;
    }
    let mut bad = full.clone();
    bad[off + 10] ^= 0xff;
    std::fs::write(&wal_path, &bad).unwrap();

    let idx = SegmentedIndex::open_durable(
        dir.clone(),
        DIM,
        Metric::L2,
        HnswParams::default(),
        10_000,
        SyncPolicy::PerOp,
    )
    .expect("bozuk WAL panic değil hata-toleranslı açılmalı");
    let rep = idx.replay_report();
    assert_eq!(rep.applied, 2, "bozuk kayıttan sonrası uygulanmamalı");
    assert!(rep.truncated_at.is_some());
    assert_eq!(idx.len(), 2);
    // Dosya kesildiği için yeni yazmalar temiz devam etmeli
    idx.insert_with_meta(VectorId(99), &vecs[0], meta_for(99))
        .unwrap();
    idx.flush_wal().unwrap();
    drop(idx);
    let idx2 = SegmentedIndex::open_durable(
        dir,
        DIM,
        Metric::L2,
        HnswParams::default(),
        10_000,
        SyncPolicy::PerOp,
    )
    .unwrap();
    assert_eq!(idx2.len(), 3, "kesme sonrası append kurtarılmalı");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// Rastgele op dizisi × rastgele kesme noktası: panic yok ve kurtarılan
    /// durum her zaman sağlam önekin durumuna eşit.
    #[test]
    fn prop_random_ops_random_cut(
        ops in proptest::collection::vec((any::<bool>(), 0u64..12), 1..25),
        cut_pct in 0u32..=100,
    ) {
        let vecs = random_vectors(16, DIM, 42);
        let dir = temp_dir("crash-prop");
        let wal_path = build_wal(&dir, &ops, &vecs);
        let full = std::fs::read(&wal_path).unwrap();
        let cut = (full.len() * cut_pct as usize) / 100;
        std::fs::write(&wal_path, &full[..cut]).unwrap();

        let (recs, _) = replay_bytes(&full[..cut]);
        let expected = expected_live(&recs, HashSet::new());
        let idx = SegmentedIndex::open_durable(
            dir,
            DIM,
            Metric::L2,
            HnswParams::default(),
            10_000,
            SyncPolicy::PerOp,
        ).expect("kesik WAL açılışı hata vermemeli");
        let got = live_ids(&idx, &[0.0; DIM], 1000);
        prop_assert_eq!(got, expected);
    }

    /// Tamamen rastgele baytlar WAL olarak sunulduğunda da panic olmamalı.
    #[test]
    fn prop_garbage_wal_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..512)) {
        let dir = temp_dir("crash-garbage");
        std::fs::write(dir.join(Manifest::wal_file_name(0)), &bytes).unwrap();
        let idx = SegmentedIndex::open_durable(
            dir,
            DIM,
            Metric::L2,
            HnswParams::default(),
            10_000,
            SyncPolicy::PerOp,
        );
        // Ok ya da Err — ikisi de kabul; panic kabul değil.
        prop_assert!(idx.is_ok() || idx.is_err());
    }
}
