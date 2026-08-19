//! The crash-recovery matrix (phase 7c).
//!
//! Instead of killing a process, **deterministic truncation**: a sequence of
//! operations is applied to a real index, the WAL file is cut both at record
//! boundaries AND mid-record, and the index is reopened. This is portable
//! (Windows included), reproducible, and the cut point is under exact control.
//!
//! The correctness criterion: the recovered state must equal the state of the
//! WAL's **intact prefix** — nothing missing (a lost op) and nothing extra (a
//! phantom op).

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

/// The set of live ids expected from the intact prefix. The WAL contains only
/// SUCCESSFUL mutations (validation happens before the append), so applying
/// them straightforwardly yields the correct reference.
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

/// Applies the operation sequence and returns the WAL file path.
fn build_wal(dir: &std::path::Path, ops: &[(bool, u64)], vecs: &[Vec<f32>]) -> std::path::PathBuf {
    let idx = SegmentedIndex::open_durable(
        dir.to_path_buf(),
        DIM,
        Metric::L2,
        HnswParams::default(),
        10_000, // no sealing: keep all data in the WAL + buffer
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

/// Reopens with the truncated WAL and compares the state to the intact prefix.
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
    .expect("opening with a truncated WAL must not error");
    let probe = vec![0.0f32; DIM];
    let got = live_ids(&idx, &probe, 10_000);
    assert_eq!(
        got, expected,
        "cut={cut}: the recovered state differs from the intact prefix"
    );
    assert_eq!(idx.len(), expected.len());
}

#[test]
fn crash_matrix_record_boundaries_and_midpoints() {
    let vecs = random_vectors(64, DIM, 42);
    // mixed insert/delete, including re-insertion chains
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

    // Derive the record boundaries
    let mut boundaries = vec![0usize];
    let mut off = 0usize;
    while off + 8 <= full.len() {
        let len = u32::from_le_bytes(full[off..off + 4].try_into().unwrap()) as usize;
        off += 8 + len;
        if off <= full.len() {
            boundaries.push(off);
        }
    }
    assert!(boundaries.len() >= 5, "not enough records");

    // Cut at every boundary and in the middle of every record
    let mut cuts: Vec<usize> = boundaries.clone();
    for w in boundaries.windows(2) {
        cuts.push((w[0] + w[1]) / 2); // mid-record
        cuts.push(w[1] - 1); // son bayt eksik
        cuts.push(w[0] + 3); // mid-header
    }
    for cut in cuts {
        // start each attempt with a clean directory
        let d = temp_dir("crash-cut");
        let wp = build_wal(&d, &ops, &vecs);
        assert_recovers_to_prefix(&d, &wp, cut);
    }
}

#[test]
fn crash_after_checkpoint_keeps_segments_and_wal_prefix() {
    let vecs = random_vectors(64, DIM, 7);
    let dir = temp_dir("crash-after-ckpt");
    // data up to the checkpoint lives in segments, the rest in the WAL
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
        // ops AFTER the checkpoint → in the new WAL file
        for id in 20..30u64 {
            idx.insert_with_meta(VectorId(id), &vecs[id as usize % 64], meta_for(id))
                .unwrap();
        }
        idx.delete_shared(VectorId(3)).unwrap(); // delete a record living in a segment
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
        .expect("open");
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
    // Corrupt the body of the third record
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
    .expect("a corrupt WAL must open fault-tolerantly, not panic");
    let rep = idx.replay_report();
    assert_eq!(
        rep.applied, 2,
        "nothing after a corrupt record may be applied"
    );
    assert!(rep.truncated_at.is_some());
    assert_eq!(idx.len(), 2);
    // Since the file was truncated, new writes must continue cleanly
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
    assert_eq!(
        idx2.len(),
        3,
        "an append after truncation must be recovered"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]

    /// A random operation sequence × a random cut point: no panic, and the
    /// recovered state always equals the state of the intact prefix.
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
        ).expect("opening a truncated WAL must not error");
        let got = live_ids(&idx, &[0.0; DIM], 1000);
        prop_assert_eq!(got, expected);
    }

    /// Feeding entirely random bytes as a WAL must not panic either.
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
        // Ok or Err — both acceptable; a panic is not.
        prop_assert!(idx.is_ok() || idx.is_err());
    }
}

/// 9a-1: if the process dies while a merge is running IN THE BACKGROUND,
/// recovery must be consistent with the old (source) segments. Since the merge
/// output has not been written to any checkpoint yet, the manifest points at
/// the sources; because the sources hold the same data nothing is lost — a
/// merge is only a reorganization.
#[test]
fn crash_during_background_merge_recovers_from_source_segments() {
    let vecs = random_vectors(1_200, DIM, 11);
    let dir = temp_dir("crash-during-merge");
    let (expected_live, merge_was_running) = {
        let mut idx = SegmentedIndex::open_durable(
            dir.clone(),
            DIM,
            Metric::L2,
            HnswParams::default(),
            200,
            SyncPolicy::PerOp,
        )
        .unwrap();
        idx.set_max_segments(3);
        // First half + checkpoint: this data lives in segment files.
        for (i, v) in vecs.iter().take(600).enumerate() {
            idx.insert_with_meta(VectorId(i as u64), v, meta_for(i as u64))
                .unwrap();
        }
        idx.wait_for_merge();
        idx.checkpoint().unwrap();
        // Second half: exceeds the ceiling → a merge starts in the background.
        // There is NO checkpoint, so these records live only in the WAL.
        for (i, v) in vecs.iter().enumerate().skip(600) {
            idx.insert_with_meta(VectorId(i as u64), v, meta_for(i as u64))
                .unwrap();
        }
        idx.delete_shared(VectorId(5)).unwrap(); // a record in a segment
        idx.delete_shared(VectorId(900)).unwrap(); // a record in the WAL
        idx.flush_wal().unwrap();
        let running = idx.merge_in_flight();
        let live = idx.len();
        // "Crash": leave without checkpointing. The background merge is cut short.
        drop(idx);
        (live, running)
    };
    assert_eq!(expected_live, 1_198);

    let idx = SegmentedIndex::open_durable(
        dir,
        DIM,
        Metric::L2,
        HnswParams::default(),
        200,
        SyncPolicy::PerOp,
    )
    .expect("a directory cut mid-merge must open");
    assert_eq!(
        idx.len(),
        1_198,
        "cutting mid-merge lost records (merge_was_running={merge_was_running})"
    );
    let all: HashSet<u64> = idx
        .search_shared(&[0.0; DIM], 2_000)
        .iter()
        .map(|r| r.id.0)
        .collect();
    assert!(!all.contains(&5), "a deleted segment record came back");
    assert!(!all.contains(&900), "a deleted WAL record came back");
    // The survivors must be findable
    for probe in [0u64, 300, 700, 1_199] {
        assert!(all.contains(&probe), "record {probe} disappeared");
    }
}
