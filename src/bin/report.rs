//! End-to-end report: recall@10, p50/p99 latency, memory and build time.
//!
//! Usage:
//!   cargo run --release --bin report -- random [n] [dim] [n_query]
//!   cargo run --release --bin report -- sift <n> [n_query]   (reads from data/sift)
//!
//! Note: SIFT's bundled ground truth is for the FULL 1M base; when using a
//! subset we generate the GT ourselves with an exact scan (otherwise recall
//! comes out wrong).

use std::time::Instant;
use vector_gvector::dataset::{random_vectors, read_fvecs_subset, read_ivecs, DEFAULT_SEED};
use vector_gvector::distance::Metric;
use vector_gvector::eval::{exact_top_k, ground_truth, measure_latency, recall_at_k};
use vector_gvector::index::bruteforce::BruteForceIndex;
use vector_gvector::index::hnsw::{HnswIndex, HnswParams};
use vector_gvector::index::VectorIndex;
use vector_gvector::types::VectorId;

/// HNSW parameter sweep: prints a recall/latency table for combinations of
/// M × ef_search and reports the speedup against the brute-force reference.
fn hnsw_sweep(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize, metric: Metric) {
    let truth = ground_truth(base, queries, k, metric);
    let dim = base[0].len();

    // Brute-force reference latency (for the speedup ratio).
    let mut bf = BruteForceIndex::new(dim, metric);
    for (i, v) in base.iter().enumerate() {
        bf.insert(VectorId(i as u64), v).expect("bf insert");
    }
    let bf_stats = measure_latency(queries, |q| {
        std::hint::black_box(bf.search(q, k));
    });
    println!(
        "brute-force referans: p50={:?} p99={:?}",
        bf_stats.p50, bf_stats.p99
    );
    println!();
    println!(
        "| M | ef_c | ef_search | recall@{k} | p50 | p99 | speedup(p50) | build | graph B/vector |"
    );
    println!(
        "|---|------|-----------|-----------|-----|-----|---------------|------|----------------|"
    );

    for (m, ef_c) in [(8, 100), (16, 200), (32, 400)] {
        let params = HnswParams {
            m,
            m_max0: 2 * m,
            ef_construction: ef_c,
            ..Default::default()
        };
        let t = Instant::now();
        let mut hnsw = HnswIndex::new(dim, metric, params);
        for (i, v) in base.iter().enumerate() {
            hnsw.insert(VectorId(i as u64), v).expect("hnsw insert");
        }
        let build = t.elapsed();
        let (_, link_bytes) = hnsw.memory_bytes();
        for ef in [10, 25, 50, 100, 200] {
            let results: Vec<Vec<VectorId>> = queries
                .iter()
                .map(|q| hnsw.search_with_ef(q, k, ef).iter().map(|r| r.id).collect())
                .collect();
            let recall = recall_at_k(&results, &truth, k);
            let stats = measure_latency(queries, |q| {
                std::hint::black_box(hnsw.search_with_ef(q, k, ef));
            });
            let speedup = bf_stats.p50.as_secs_f64() / stats.p50.as_secs_f64();
            println!(
                "| {m} | {ef_c} | {ef} | {recall:.4} | {:?} | {:?} | {speedup:.1}x | {build:.1?} | {:.0} |",
                stats.p50,
                stats.p99,
                link_bytes as f64 / base.len() as f64
            );
        }
    }
}

/// Filter selectivity sweep (plan: the fallback measurement).
///
/// Three match distributions:
/// - uniform: spread evenly across id space (the baseline)
/// - clustered: clustered in VECTOR space — the nearest s·n neighbours of a
///   centre. Queries are grouped near/mid/far by their distance to that centre:
///   the fragility is expected when the query is FAR from the match region.
/// - contig: the first s·n records, contiguous in id space (kept separate for
///   the interaction with segment boundaries)
fn filter_sweep(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize, metric: Metric) {
    use std::collections::HashSet;
    use vector_gvector::meta::{Filter, MetaValue, Metadata, Predicate};

    let n = base.len();
    let dim = base[0].len();
    let ef = 50usize;
    let ef_cap = 4096usize; // cap for the scaled-ef arm (from user feedback)

    let mut hnsw = HnswIndex::new(dim, metric, HnswParams::default());
    let mut bf = BruteForceIndex::new(dim, metric);
    for (i, v) in base.iter().enumerate() {
        hnsw.insert(VectorId(i as u64), v).expect("insert");
        bf.insert(VectorId(i as u64), v).expect("insert");
    }

    // Unfiltered reference (the fixed-cost comparison for the s=1.0 row).
    let unfiltered = measure_latency(queries, |q| {
        std::hint::black_box(hnsw.search_with_ef(q, k, ef));
    });
    println!(
        "unfiltered search p50 = {:?} (comparison for s=1.0)",
        unfiltered.p50
    );

    // Planning cost: how expensive would an O(n) metadata scan be for a query
    // planner? Build a realistic metadata map and time a full count.
    let mut meta_store = vector_gvector::meta::MetaStore::new();
    for i in 0..n {
        meta_store.insert(
            VectorId(i as u64),
            [("b".to_string(), MetaValue::Int(i as i64))].into(),
        );
    }
    let probe = Filter {
        must: vec![Predicate::Range {
            key: "b".into(),
            min: 0.0,
            max: (n / 2) as f64,
        }],
    };
    let t = std::time::Instant::now();
    let cnt = (0..n)
        .filter(|&i| probe.matches_id(&meta_store, VectorId(i as u64)))
        .count();
    println!(
        "planning cost (O(n) full count, n={n}): {:?} (count={cnt})",
        t.elapsed()
    );
    println!();
    println!("| varyant | s | grup | recall | p50 | fallback | ziyaret | kabul/ziyaret | ef' recall | ef' p50 | tarama p50 |");
    println!("|---------|---|------|--------|-----|----------|---------|----------------|------------|---------|------------|");

    // The centre of the clustered variant: a vector from the base set.
    let center = &base[0];
    // Sort queries by distance to the centre and split into thirds
    // (near/mid/far).
    let mut q_order: Vec<usize> = (0..queries.len()).collect();
    q_order.sort_by(|&a, &b| {
        metric
            .distance(&queries[a], center)
            .total_cmp(&metric.distance(&queries[b], center))
    });
    let third = queries.len() / 3;

    let s_levels = [0.001f64, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0];
    let make_allow_set = |variant: &str, s: f64| -> HashSet<u64> {
        let m = ((s * n as f64) as usize).max(1);
        match variant {
            "uniform" => {
                let step = (n / m).max(1);
                (0..n).step_by(step).take(m).map(|i| i as u64).collect()
            }
            "clustered" => exact_top_k(base, center, m, metric)
                .iter()
                .map(|r| r.id.0)
                .collect(),
            _ => (0..m as u64).collect(),
        }
    };

    for variant in ["uniform", "clustered", "contig"] {
        for s in s_levels {
            let allow_set = make_allow_set(variant, s);
            let allow = |id: VectorId| allow_set.contains(&id.0);
            let s_real = allow_set.len() as f64 / n as f64;
            let ef_scaled = ((k as f64 / s_real).ceil() as usize).clamp(ef, ef_cap);

            // For the clustered variant queries are split into distance groups;
            // the others use a single group.
            let groups: Vec<(&str, Vec<usize>)> = if variant == "clustered" {
                vec![
                    ("near", q_order[..third].to_vec()),
                    ("orta", q_order[third..2 * third].to_vec()),
                    ("uzak", q_order[2 * third..].to_vec()),
                ]
            } else {
                vec![("-", (0..queries.len()).collect())]
            };

            for (gname, qidx) in groups {
                let qs: Vec<Vec<f32>> = qidx.iter().map(|&i| queries[i].clone()).collect();
                let mut hits = 0usize;
                let mut total = 0usize;
                let mut fallbacks = 0usize;
                let mut visited_sum = 0usize;
                let mut admitted_sum = 0usize;
                let mut hits_scaled = 0usize;
                for q in &qs {
                    let truth: Vec<VectorId> = bf
                        .search_filtered(q, k, &allow)
                        .iter()
                        .map(|r| r.id)
                        .collect();
                    let (res, st) = hnsw.search_filtered_stats(q, k, ef, &allow, None);
                    hits += res.iter().filter(|r| truth.contains(&r.id)).count();
                    total += truth.len();
                    fallbacks += st.fallback_used as usize;
                    visited_sum += st.visited;
                    admitted_sum += st.admitted;
                    let res2 = hnsw.search_filtered_with_ef(q, k, ef_scaled, &allow);
                    hits_scaled += res2.iter().filter(|r| truth.contains(&r.id)).count();
                }
                let nq = qs.len().max(1);
                let recall = hits as f64 / total.max(1) as f64;
                let recall_scaled = hits_scaled as f64 / total.max(1) as f64;
                let lat = measure_latency(&qs, |q| {
                    std::hint::black_box(hnsw.search_filtered_with_ef(q, k, ef, &allow));
                });
                let lat_scaled = measure_latency(&qs, |q| {
                    std::hint::black_box(hnsw.search_filtered_with_ef(q, k, ef_scaled, &allow));
                });
                // The planner's alternative arm: scanning only the matches.
                let lat_scan = measure_latency(&qs, |q| {
                    std::hint::black_box(bf.search_filtered(q, k, &allow));
                });
                println!(
                    "| {variant} | {s} | {gname} | {recall:.3} | {:?} | {}/{nq} | {} | {:.3} | {recall_scaled:.3} | {:?} | {:?} |",
                    lat.p50,
                    fallbacks,
                    visited_sum / nq,
                    admitted_sum as f64 / visited_sum.max(1) as f64,
                    lat_scaled.p50,
                    lat_scan.p50,
                );
            }
        }
    }

    // ---- The "after" table: SegmentedIndex with the planner, end to end ----
    // Her s seviyesi bir Bool etiketi olur ("s0".."s6"); filtre Eq ile
    // uses the posting-list path. One build per variant.
    use vector_gvector::index::segmented::SegmentedIndex;
    println!();
    println!("== SegmentedIndex with planner (posting list + scan / unfiltered over-fetch) ==");
    println!("| varyant | s | grup | recall | p50 |");
    println!("|---------|---|------|--------|-----|");
    for variant in ["uniform", "clustered", "contig"] {
        let sets: Vec<HashSet<u64>> = s_levels
            .iter()
            .map(|&s| make_allow_set(variant, s))
            .collect();
        let idx = SegmentedIndex::new(dim, metric, HnswParams::default(), (n / 4).max(1000));
        for (i, v) in base.iter().enumerate() {
            let mut m: Metadata = Metadata::new();
            for (si, set) in sets.iter().enumerate() {
                if set.contains(&(i as u64)) {
                    m.insert(format!("s{si}"), MetaValue::Bool(true));
                }
            }
            idx.insert_with_meta(VectorId(i as u64), v, m)
                .expect("insert");
        }
        for (si, s) in s_levels.iter().enumerate() {
            let filter = Filter {
                must: vec![Predicate::Eq {
                    key: format!("s{si}"),
                    value: MetaValue::Bool(true),
                }],
            };
            let allow_set = &sets[si];
            let allow = |id: VectorId| allow_set.contains(&id.0);
            let groups: Vec<(&str, Vec<usize>)> = if variant == "clustered" {
                vec![
                    ("near", q_order[..third].to_vec()),
                    ("orta", q_order[third..2 * third].to_vec()),
                    ("uzak", q_order[2 * third..].to_vec()),
                ]
            } else {
                vec![("-", (0..queries.len()).collect())]
            };
            for (gname, qidx) in groups {
                let qs: Vec<Vec<f32>> = qidx.iter().map(|&i| queries[i].clone()).collect();
                let mut hits = 0usize;
                let mut total = 0usize;
                for q in &qs {
                    let truth: Vec<VectorId> = bf
                        .search_filtered(q, k, &allow)
                        .iter()
                        .map(|r| r.id)
                        .collect();
                    let res = idx.search_filtered(q, k, &filter);
                    hits += res.iter().filter(|r| truth.contains(&r.id)).count();
                    total += truth.len();
                }
                let recall = hits as f64 / total.max(1) as f64;
                let lat = measure_latency(&qs, |q| {
                    std::hint::black_box(idx.search_filtered(q, k, &filter));
                });
                println!(
                    "| {variant} | {s} | {gname} | {recall:.3} | {:?} |",
                    lat.p50
                );
            }
        }
    }
}

/// Process RSS (bytes). Without adding a dependency, `WorkingSet64` is read via
/// PowerShell. It is called a handful of times between measurement steps, so the
/// ~200 ms process startup cost does not affect the measurement.
fn rss_bytes() -> u64 {
    let pid = std::process::id();
    std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            &format!("(Get-Process -Id {pid}).WorkingSet64"),
        ])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

/// Phase 8: the 1M end-to-end reality check (the full system: segmented index,
/// planner, filters and WAL). Pre-registered thresholds: DECISIONS #40/#41.
fn full_scale(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize, metric: Metric) {
    use std::collections::HashSet;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use vector_gvector::index::segmented::SegmentedIndex;
    use vector_gvector::meta::{Filter, MetaValue, Metadata, Predicate};
    use vector_gvector::storage::wal::SyncPolicy;

    let n = base.len();
    let dim = base[0].len();
    let seal = n / 8; // tavan=8 ile tam 8 segment
    let dir = std::path::PathBuf::from("data/fullscale");
    let _ = std::fs::remove_dir_all(&dir);
    // The clustered match sets for the critical filter cells are computed UP
    // FRONT and marked as metadata fields — that way the measurement goes through
    // the real planner path (arm agreement can only be measured this way).
    let center = base[0].clone();
    let s_levels = [0.001f64, 0.05, 0.3];
    let cluster_names = ["c001", "c05", "c3"];
    println!("preparing clustered filter sets (exact top-k × 3)...");
    let t_prep = Instant::now();
    let clusters: Vec<HashSet<u64>> = s_levels
        .iter()
        .map(|&s| {
            let m = ((s * n as f64) as usize).max(1);
            exact_top_k(base, &center, m, metric)
                .iter()
                .map(|r| r.id.0)
                .collect()
        })
        .collect();
    println!("  ready ({:?})", t_prep.elapsed());
    let meta_of = |i: usize| -> Metadata {
        let mut m: Metadata = [
            ("grup".to_string(), MetaValue::Int((i % 8) as i64)),
            ("v".to_string(), MetaValue::Int(i as i64)),
            ("f".to_string(), MetaValue::Float(i as f64 * 0.25)),
        ]
        .into();
        for (name, set) in cluster_names.iter().zip(&clusters) {
            if set.contains(&(i as u64)) {
                m.insert((*name).to_string(), MetaValue::Bool(true));
            }
        }
        m
    };

    println!("### 1. Build (n={n}, seal={seal}, ceiling=8, 3 metadata fields, WAL=group:20)");
    let idx = SegmentedIndex::open_durable(
        dir.clone(),
        dim,
        metric,
        HnswParams::default(),
        seal,
        SyncPolicy::Group { window_ms: 20 },
    )
    .expect("open");
    let t = Instant::now();
    for (i, v) in base.iter().enumerate() {
        idx.insert_with_meta(VectorId(i as u64), v, meta_of(i))
            .expect("insert");
        if (i + 1) % 250_000 == 0 {
            println!("  {} / {n} ({:?})", i + 1, t.elapsed());
        }
    }
    idx.commit_wal().expect("commit");
    let build = t.elapsed();
    let (n_seg, n_buf) = idx.shape();
    println!("build: {build:.1?} → {n_seg} segments + {n_buf} buffer");

    println!();
    println!("### 2. Memory (computed; RSS sampled externally)");
    let index_mem = idx.memory_bytes();
    let (m_meta, m_post, m_num) = idx.metadata_memory_bytes();
    let meta_total = m_meta + m_post + m_num;
    let mb = |b: usize| b as f64 / 1048576.0;
    println!("vectors+graph (f32): {:.0} MB", mb(index_mem));
    println!(
        "metadata total: {:.0} MB (map {:.0} + postings {:.0} + numeric {:.0})",
        mb(meta_total),
        mb(m_meta),
        mb(m_post),
        mb(m_num)
    );
    let meta_share = meta_total as f64 / (index_mem + meta_total) as f64;
    println!(
        "metadata share: {:.1}% — the 9c threshold is 25% → {}",
        meta_share * 100.0,
        if meta_share > 0.25 { "GO" } else { "NO-GO" }
    );

    println!();
    println!("### 3. Checkpoint + disk");
    let t = Instant::now();
    let gen = idx.checkpoint().expect("checkpoint");
    let ck = t.elapsed();
    let disk: u64 = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok()?.metadata().ok().map(|m| m.len()))
        .sum();
    println!(
        "checkpoint (gen={gen}): {ck:.2?}, disk {:.0} MB ({:.0} B/vector)",
        disk as f64 / 1048576.0,
        disk as f64 / n as f64
    );

    println!();
    println!("### 4. Recall baseline (ef=100)");
    // The official SIFT ground truth is valid ONLY for the full 1M base; on a
    // subset the ids point at different vectors (this was caught as recall
    // 0.0000 in a smoke test). On a subset we generate it with an exact scan.
    let truth: Vec<Vec<VectorId>> = if n == 1_000_000 {
        println!("  (official SIFT ground truth — full set, valid)");
        read_ivecs(std::path::Path::new("data/sift/sift_groundtruth.ivecs"))
            .expect("ground truth")
            .iter()
            .take(queries.len())
            .map(|row| row.iter().take(k).map(|&i| VectorId(i as u64)).collect())
            .collect()
    } else {
        println!("  (subset: the official GT is invalid → generating via exact scan)");
        ground_truth(base, queries, k, metric)
    };
    let results: Vec<Vec<VectorId>> = queries
        .iter()
        .map(|q| idx.search_shared(q, k).iter().map(|r| r.id).collect())
        .collect();
    let recall = recall_at_k(&results, &truth, k);
    let lat = measure_latency(queries, |q| {
        std::hint::black_box(idx.search_shared(q, k));
    });
    println!(
        "recall@{k} = {recall:.4} (threshold ≥0.99 → {}), p50={:?} p99={:?}",
        if recall >= 0.99 { "TUTTU" } else { "TUTMADI" },
        lat.p50,
        lat.p99
    );

    println!();
    println!("### 5. Critical filter cells (clustered × distant query, real planner path)");
    let mut q_order: Vec<usize> = (0..queries.len()).collect();
    q_order.sort_by(|&a, &b| {
        metric
            .distance(&queries[a], &center)
            .total_cmp(&metric.distance(&queries[b], &center))
    });
    let far: Vec<Vec<f32>> = q_order[q_order.len() * 2 / 3..]
        .iter()
        .map(|&i| queries[i].clone())
        .collect();
    let scan_limit = (16 * k).max(n / 20);
    println!("scan_limit = {scan_limit}");
    println!("| s | matches | arm (oracle) | recall | p50 |");
    println!("|---|---------|--------------|--------|-----|");
    let mut arm_agree = 0usize;
    for (si, s) in s_levels.iter().enumerate() {
        let allow_set = &clusters[si];
        let filter = Filter {
            must: vec![Predicate::Eq {
                key: cluster_names[si].to_string(),
                value: MetaValue::Bool(true),
            }],
        };
        let arm = idx.debug_plan_arm(&filter, k);
        let oracle = if allow_set.len() <= scan_limit {
            "scan"
        } else {
            "post"
        };
        arm_agree += (arm == oracle) as usize;
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &far {
            // reference: exact top-k over the matches only
            let mut cand: Vec<(f32, u64)> = allow_set
                .iter()
                .map(|&id| (metric.distance(q, &base[id as usize]), id))
                .collect();
            cand.sort_by(|a, b| a.0.total_cmp(&b.0));
            let tr: Vec<u64> = cand.iter().take(k).map(|x| x.1).collect();
            let res = idx.search_filtered(q, k, &filter);
            hits += res.iter().filter(|r| tr.contains(&r.id.0)).count();
            total += tr.len();
        }
        let latf = measure_latency(&far, |q| {
            std::hint::black_box(idx.search_filtered(q, k, &filter));
        });
        println!(
            "| {s} | {} | {arm} ({oracle}) | {:.3} | {:?} |",
            allow_set.len(),
            hits as f64 / total.max(1) as f64,
            latf.p50
        );
    }
    println!(
        "arm agreement: {arm_agree}/{} ({:.0}%)",
        s_levels.len(),
        arm_agree as f64 / s_levels.len() as f64 * 100.0
    );

    println!();
    println!("### 6. Merge window (the rationale and baseline for 9a)");
    let extra = seal + 5_000; // seals a 9th segment → triggers a merge
    println!("  {extra} extra writes (to sample before and after the window)");
    let mut lats: Vec<std::time::Duration> = Vec::with_capacity(extra);
    let t_all = Instant::now();
    for i in 0..extra {
        let id = (n + i) as u64;
        let v = &base[i % n];
        let t = Instant::now();
        idx.insert_with_meta(VectorId(id), v, Metadata::new())
            .expect("insert");
        lats.push(t.elapsed());
        if (i + 1) % 50_000 == 0 {
            println!("    {} / {extra} ({:?})", i + 1, t_all.elapsed());
        }
    }
    idx.commit_wal().expect("commit");
    let mut sorted_lat = lats.clone();
    sorted_lat.sort();
    let pct =
        |p: f64| sorted_lat[((sorted_lat.len() as f64 * p) as usize).min(sorted_lat.len() - 1)];
    // The windows: the longest write = the call that does seal+merge
    let max_lat = *sorted_lat.last().unwrap();
    // Baseline: p99 excluding the 3 longest writes (the seal/merge calls)
    let cutoff = sorted_lat.len().saturating_sub(3);
    let base_p99 = sorted_lat[(cutoff as f64 * 0.99) as usize];
    println!(
        "baseline p50={:?} p99={:?} (excluding seal/merge calls)",
        pct(0.5),
        base_p99
    );
    println!(
        "EN UZUN yazma: {:?} — seal {:?} + merge {:?} (toplam {} merge)",
        max_lat,
        std::time::Duration::from_micros(idx.last_seal_us()),
        std::time::Duration::from_micros(idx.last_merge_us()),
        idx.merge_count()
    );
    let ratio = max_lat.as_secs_f64() / base_p99.as_secs_f64().max(1e-9);
    println!(
        "ratio: {ratio:.0}x (the 9a acceptance threshold is 50x — to be measured \
         AFTER 9a; the current value is the rationale for 9a)"
    );
    let (n_seg2, _) = idx.shape();
    println!("segment count: {n_seg2} (ceiling 8 holds)");

    println!();
    println!("### 7. Cold start (the 9b baseline)");
    idx.checkpoint().expect("checkpoint2");
    drop(idx);
    let mut cold_times = Vec::new();
    for round in 0..3 {
        let t = Instant::now();
        let re = SegmentedIndex::open_durable(
            dir.clone(),
            dim,
            metric,
            HnswParams::default(),
            seal,
            SyncPolicy::Group { window_ms: 20 },
        )
        .expect("reopen");
        let el = t.elapsed();
        cold_times.push(el);
        if round == 0 {
            println!("  empty WAL: {el:.2?} ({} records)", re.len_shared());
        }
    }
    cold_times.sort();
    println!(
        "cold start median (3 rounds, empty WAL): {:.2?}",
        cold_times[1]
    );

    // Cold start with a 10K WAL
    let idx = SegmentedIndex::open_durable(
        dir.clone(),
        dim,
        metric,
        HnswParams::default(),
        seal,
        SyncPolicy::Group { window_ms: 20 },
    )
    .expect("open");
    for i in 0..10_000usize {
        let id = (n + extra + i) as u64;
        idx.insert_with_meta(VectorId(id), &base[i % n], Metadata::new())
            .expect("insert");
    }
    idx.flush_wal().expect("flush");
    let wal_mb = idx.wal_len_bytes() as f64 / 1048576.0;
    drop(idx);
    let t = Instant::now();
    let re = SegmentedIndex::open_durable(
        dir.clone(),
        dim,
        metric,
        HnswParams::default(),
        seal,
        SyncPolicy::Group { window_ms: 20 },
    )
    .expect("reopen");
    println!(
        "cold start + 10K WAL ({wal_mb:.1} MB): {:.2?} (replayed {} records)",
        t.elapsed(),
        re.replay_report().applied
    );

    println!();
    println!("### 8. Mixed load: 8 readers + 1 writer × 3 fsync policies");
    // The writer runs at a REALISTIC rate (throttled). Unbounded writing broke
    // two things at once: (a) the buffer swelled and condemned readers to a
    // brute-force scan, and (b) sealing was triggered so the table measured "an
    // HNSW build cutting in" rather than "the fsync policy". Our question is "do
    // readers slow down while a writer is active", not "how fast is the writer"
    // (that was phase 7).
    const WRITE_RATE: u64 = 200; // target op/s (even per_op must keep up)
    let idx = Arc::new(re);
    let bench_reads = |idx: &Arc<SegmentedIndex>,
                       with_writer: bool|
     -> (f64, f64, std::time::Duration) {
        let stop = AtomicBool::new(false);
        let reads = AtomicUsize::new(0);
        let writes = AtomicUsize::new(0);
        let read_ns = AtomicUsize::new(0);
        let secs = 3u64;
        std::thread::scope(|sc| {
            for t in 0..8 {
                let (idx, stop, reads, read_ns) = (idx, &stop, &reads, &read_ns);
                sc.spawn(move || {
                    let mut i = 0usize;
                    while !stop.load(Ordering::Relaxed) {
                        let q = &queries[(i + t) % queries.len()];
                        let t0 = Instant::now();
                        std::hint::black_box(idx.search_shared(q, k));
                        read_ns.fetch_add(t0.elapsed().as_nanos() as usize, Ordering::Relaxed);
                        i += 1;
                        reads.fetch_add(1, Ordering::Relaxed);
                    }
                });
            }
            if with_writer {
                let (idx, stop, writes) = (idx, &stop, &writes);
                sc.spawn(move || {
                    let interval = std::time::Duration::from_micros(1_000_000 / WRITE_RATE);
                    let mut next = Instant::now();
                    let mut i = 0usize;
                    while !stop.load(Ordering::Relaxed) {
                        next += interval;
                        let id = VectorId(9_000_000 + i as u64);
                        let _ = idx.insert_with_meta(id, &base[i % 1000], Metadata::new());
                        let _ = idx.commit_wal();
                        i += 1;
                        writes.fetch_add(1, Ordering::Relaxed);
                        let now = Instant::now();
                        if next > now {
                            std::thread::sleep(next - now);
                        }
                    }
                });
            }
            std::thread::sleep(std::time::Duration::from_secs(secs));
            stop.store(true, Ordering::Relaxed);
        });
        let r = reads.load(Ordering::Relaxed);
        (
            r as f64 / secs as f64,
            writes.load(Ordering::Relaxed) as f64 / secs as f64,
            std::time::Duration::from_nanos((read_ns.load(Ordering::Relaxed) / r.max(1)) as u64),
        )
    };
    let (base_qps, _, base_p50) = bench_reads(&idx, false);
    println!("writer-free baseline: {base_qps:.0} QPS, read p50 {base_p50:?} (8 readers)");
    println!("| politika | okuma QPS | tabana oran | yazma op/s | okuma p50 |");
    println!("|----------|-----------|-------------|------------|-----------|");
    for policy in [
        SyncPolicy::None,
        SyncPolicy::Group { window_ms: 20 },
        SyncPolicy::PerOp,
    ] {
        idx.set_wal_policy(policy).expect("policy");
        let (qps, wps, p50) = bench_reads(&idx, true);
        println!(
            "| {} | {qps:.0} | {:.2} | {wps:.0} | {p50:?} |",
            policy.label(),
            qps / base_qps.max(1.0)
        );
    }

    println!();
    println!("### 9. 1M kaza testi (dolu WAL + kesme)");
    let idx = Arc::try_unwrap(idx).ok().expect("tek referans");
    // Deliberately push the WAL ABOVE the sealing threshold, to see how replay
    // affects recovery time (once the replayed inserts fill the buffer an HNSW
    // build is triggered — so recovery time is NOT linear in WAL size).
    let wal_fill = idx.seal_threshold().min(150_000) + 20_000;
    println!(
        "  writing {wal_fill} records to the WAL (sealing threshold {})",
        idx.seal_threshold()
    );
    for i in 0..wal_fill {
        let id = VectorId(20_000_000 + i as u64);
        idx.insert_with_meta(id, &base[i % n], Metadata::new())
            .expect("insert");
    }
    idx.flush_wal().expect("flush");
    let wal_path = dir.join(vector_gvector::storage::Manifest::wal_file_name(
        idx.generation(),
    ));
    let live_before = idx.len_shared();
    drop(idx);
    let full = std::fs::read(&wal_path).unwrap_or_default();
    if full.is_empty() {
        println!("the WAL is empty — the truncation scenario was skipped");
    } else {
        let cut = full.len() * 2 / 3;
        std::fs::write(&wal_path, &full[..cut]).expect("kes");
        let (prefix, _) = vector_gvector::storage::wal::replay_bytes(&full[..cut]);
        let t = Instant::now();
        let re = SegmentedIndex::open_durable(
            dir.clone(),
            dim,
            metric,
            HnswParams::default(),
            seal,
            SyncPolicy::Group { window_ms: 20 },
        )
        .expect("opening with a truncated WAL");
        println!(
            "truncated WAL ({:.1} MB → {:.1} MB) open: {:.2?}, replayed {} records (intact prefix {} records)",
            full.len() as f64 / 1048576.0,
            cut as f64 / 1048576.0,
            t.elapsed(),
            re.replay_report().applied,
            prefix.len()
        );
        println!(
            "  record count: {live_before} before the cut → {} recovered (the difference is the severed tail)",
            re.len_shared()
        );
    }
    println!();
    println!("=== fullscale complete ===");
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("random");
    let k = 10;
    let metric = Metric::L2; // SIFT is evaluated with L2 in the literature

    let (base, queries, label) = match mode {
        "sift" | "sweep" | "persist" | "delete" | "concurrent" | "quant" | "sift1m" | "filter"
        | "segcurve" | "mergecost" | "rangefilter" | "durability" | "wal" | "fullscale"
        | "int8scale" | "coldprofile" | "mergewindow" | "accumulation" | "memverify"
        | "postingcost" => {
            let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(10_000);
            let n_query: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(100);
            // If SIFT is absent we fall back to random data INSTEAD OF PANICKING:
            // someone new to the project should be able to run a measurement
            // without downloading a 1 GB dataset. The fallback is NOT SILENT —
            // recall numbers are not comparable to real data, so a warning is
            // printed and the label
            // "random" olur.
            match std::fs::File::open("data/sift/sift_base.fvecs") {
                Ok(bf) => {
                    let mut f = std::io::BufReader::new(bf);
                    let base = read_fvecs_subset(&mut f, n).expect("could not read base");
                    let mut fq = std::io::BufReader::new(
                        std::fs::File::open("data/sift/sift_query.fvecs").expect("query file"),
                    );
                    let queries =
                        read_fvecs_subset(&mut fq, n_query).expect("could not read queries");
                    (base, queries, format!("SIFT subset n={n}"))
                }
                Err(_) => {
                    eprintln!("WARNING: data/sift not found → running on random vectors.");
                    eprintln!("  Recall numbers on random data are not comparable to");
                    eprintln!("  SIFT results: random high-dimensional data is the worst");
                    eprintln!("  case for ANN. For real measurements, extract SIFT-1M");
                    eprintln!("  into data/sift/.");
                    let dim = 128;
                    (
                        random_vectors(n, dim, DEFAULT_SEED),
                        random_vectors(n_query, dim, DEFAULT_SEED + 1),
                        format!("random n={n} (SIFT yok)"),
                    )
                }
            }
        }
        _ => {
            let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(10_000);
            let dim: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(128);
            let n_query: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(100);
            (
                random_vectors(n, dim, DEFAULT_SEED),
                random_vectors(n_query, dim, DEFAULT_SEED + 1),
                format!("random n={n}"),
            )
        }
    };
    let dim = base[0].len();
    println!(
        "== rapor: {label} dim={dim} queries={} k={k} metric={metric:?} seed={DEFAULT_SEED} ==",
        queries.len()
    );

    if mode == "sweep" {
        hnsw_sweep(&base, &queries, k, metric);
        return;
    }

    if mode == "filter" {
        filter_sweep(&base, &queries, k, metric);
        return;
    }

    if mode == "postingcost" {
        // The 9c risk: inserting into a sorted Vec costs an O(n) shift. If ids
        // arrive in ASCENDING order the insert always lands at the end (O(1)); in
        // RANDOM order it degrades to O(n²). Our measurements always generated ids
        // in ascending order, so this difference was invisible — here it is broken
        // on purpose.
        use rand::seq::SliceRandom;
        use rand::SeedableRng;
        use vector_gvector::index::segmented::SegmentedIndex;
        use vector_gvector::meta::{MetaValue, Metadata};

        let n = base.len().min(200_000);
        println!("{n} records into a single posting list (all sharing one Eq value)");
        println!();
        println!("| id order | time | records/s |");
        println!("|-----------|------|---------|");
        for label in ["artan", "rastgele"] {
            let mut ids: Vec<u64> = (0..n as u64).collect();
            if label == "rastgele" {
                let mut rng = rand::rngs::StdRng::seed_from_u64(DEFAULT_SEED);
                ids.shuffle(&mut rng);
            }
            // Sealing off: what is measured is posting maintenance, not HNSW
            // construction.
            let idx = SegmentedIndex::new(dim, metric, HnswParams::default(), usize::MAX);
            let t = Instant::now();
            for (i, id) in ids.iter().enumerate() {
                let meta: Metadata = [("sabit".to_string(), MetaValue::Int(1))].into();
                idx.insert_with_meta(VectorId(*id), &base[i % base.len()], meta)
                    .expect("insert");
            }
            let el = t.elapsed();
            println!("| {label} | {el:?} | {:.0} |", n as f64 / el.as_secs_f64());
        }
        return;
    }

    if mode == "memverify" {
        // 9c-0: validate the metadata memory ESTIMATE against real RSS.
        //
        // The structures are dropped one at a time and RSS is measured at each
        // step, comparing the estimate (capacity × a fixed factor) with the real
        // delta. 9c's GO decision (a 51.5% share) rests on that estimate; if it is
        // far off, 9c's scope has to be redrawn.
        use vector_gvector::index::segmented::SegmentedIndex;
        use vector_gvector::storage::wal::SyncPolicy;

        let dir = std::path::PathBuf::from("data/fullscale");
        if !dir.join("MANIFEST").exists() {
            eprintln!("data/fullscale is missing — run `report -- fullscale 1000000 99` first");
            return;
        }
        let idx = SegmentedIndex::open_durable(
            dir,
            dim,
            metric,
            HnswParams::default(),
            125_000,
            SyncPolicy::None,
        )
        .expect("open");
        // Warmup: touch the structures (so lazy page loading completes).
        let _ = idx.search_shared(&queries[0], k);
        let (m_est, p_est, n_est) = idx.metadata_memory_bytes();
        let mb = |b: usize| b as f64 / 1e6;
        let rss_mb = |b: u64| b as f64 / 1e6;
        println!("records: {}", idx.len_shared());
        println!();
        println!("| step | RSS | drop | estimate | estimate/real |");
        println!("|------|-----|-------|--------|----------------|");
        let r0 = rss_bytes();
        println!("| start | {:.0} MB | — | — | — |", rss_mb(r0));
        let mut prev = r0;
        for (what, est) in [("numeric", n_est), ("postings", p_est), ("metadata", m_est)] {
            idx.clear_for_measurement(what);
            let now = rss_bytes();
            let drop_b = prev.saturating_sub(now);
            println!(
                "| −{what} | {:.0} MB | **{:.0} MB** | {:.0} MB | {:.2}x |",
                rss_mb(now),
                rss_mb(drop_b),
                mb(est),
                est as f64 / (drop_b as f64).max(1.0)
            );
            prev = now;
        }
        let total_est = m_est + p_est + n_est;
        let total_real = r0.saturating_sub(prev);
        println!();
        println!(
            "TOTAL: estimate {:.0} MB, real drop {:.0} MB → estimate/real {:.2}x",
            mb(total_est),
            rss_mb(total_real),
            total_est as f64 / (total_real as f64).max(1.0)
        );
        println!(
            "metadata share (real): {:.1}% (of {:.0} MB RSS)",
            total_real as f64 / r0 as f64 * 100.0,
            rss_mb(r0)
        );
        println!();
        println!("NOTE: freed memory may not return to the OS immediately;");
        println!(
            "in that case the real drop looks SMALLER than it is (a bias favouring the estimate)."
        );
        return;
    }

    if mode == "accumulation" {
        // 9a-2 CRITERION 2 (pre-registration #49): under sustained FULL-SPEED
        // writing, does the segment count stabilize or grow monotonically?
        //
        // The threshold (fixed before the measurement): the average of the last
        // third may exceed that of the first third by at most 20% AND no sample
        // may exceed ceiling+4 (12) → accepted without backpressure. Otherwise
        // backpressure is done in the same arc.
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::Arc;
        use vector_gvector::index::segmented::SegmentedIndex;
        use vector_gvector::meta::Metadata;
        use vector_gvector::storage::wal::SyncPolicy;

        // WAL OFF: the accumulation measurement targets the INTERNAL dynamics of
        // the write path; with the WAL on, the first attempt wrote a 4.3 GB log in
        // 120 s and drowned the measurement in disk/memory pressure. Duration per
        // pre-registration #59: 10 minutes. A 2-minute window was not enough for
        // the segment count to REACH the merge ceiling, so the "approaching the
        // ceiling" curve was misread as "monotone growth".
        let secs: u64 = 600;
        let seal = 125_000usize;
        let dir = std::path::PathBuf::from("data/accum");
        let _ = std::fs::remove_dir_all(&dir);
        let mut idx = SegmentedIndex::open_durable(
            dir.clone(),
            dim,
            metric,
            HnswParams::default(),
            seal,
            SyncPolicy::None,
        )
        .expect("open");
        idx.set_max_segments(8);
        let idx = Arc::new(idx);
        println!("FULL-SPEED writing for {secs} s; seal={seal}, ceiling=8, sampled every 5 s");
        println!();
        println!("| t (s) | segments | sealing | buffer | total records | write op/s |");
        println!("|-------|---------|------------|--------|--------------|------------|");

        let stop = AtomicBool::new(false);
        let written = AtomicUsize::new(0);
        let mut samples: Vec<usize> = Vec::new();
        // #59: the primary criterion is the QUEUE; segments are tracked as a
        // separate item.
        let mut queue_samples: Vec<usize> = Vec::new();
        let mut seg_samples: Vec<usize> = Vec::new();
        std::thread::scope(|sc| {
            let w = &written;
            let st = &stop;
            let ix = &idx;
            sc.spawn(move || {
                let mut i = 0usize;
                while !st.load(Ordering::Relaxed) {
                    let id = VectorId(i as u64);
                    if ix
                        .insert_with_meta(id, &base[i % base.len()], Metadata::new())
                        .is_ok()
                    {
                        w.fetch_add(1, Ordering::Relaxed);
                    }
                    i += 1;
                }
            });
            let mut last = 0usize;
            for t in 1..=(secs / 5) {
                std::thread::sleep(std::time::Duration::from_secs(5));
                let (segs, buf) = idx.shape();
                let sealing = idx.sealing_count();
                let total = written.load(Ordering::Relaxed);
                samples.push(segs + sealing); // pre-registration #49's (flawed) metric
                queue_samples.push(sealing); // #59 birincil
                seg_samples.push(segs); // #59 secondary (a test of the merge ceiling)
                println!(
                    "| {} | {segs} | {sealing} | {buf} | {} | {:.0} |",
                    t * 5,
                    idx.len_shared(),
                    (total - last) as f64 / 5.0
                );
                use std::io::Write;
                let _ = std::io::stdout().flush();
                last = total;
            }
            stop.store(true, Ordering::Relaxed);
        });

        let third = samples.len() / 3;
        let avg = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len().max(1) as f64;
        let first_avg = avg(&samples[..third]);
        let last_avg = avg(&samples[samples.len() - third..]);
        let peak = *samples.iter().max().unwrap_or(&0);
        let growth = if first_avg > 0.0 {
            (last_avg / first_avg - 1.0) * 100.0
        } else {
            0.0
        };
        println!();
        println!("| criterion | value | threshold | result |");
        println!("|--------|-------|------|-------|");
        println!(
            "| ilk 1/3 ort. → son 1/3 ort. | {first_avg:.1} → {last_avg:.1} ({growth:+.0}%) | ≤ +%20 | {} |",
            if growth <= 20.0 { "OK" } else { "EXCEEDED" }
        );
        println!(
            "| peak (segments+sealing) | {peak} | ≤ 12 (ceiling+4) | {} |",
            if peak <= 12 { "OK" } else { "EXCEEDED" }
        );
        let verdict = growth <= 20.0 && peak <= 12;
        // #59 PRIMARY: does the queue settle at a fixed upper bound?
        let q_first = avg(&queue_samples[..third]);
        let q_last = avg(&queue_samples[queue_samples.len() - third..]);
        let q_peak = *queue_samples.iter().max().unwrap_or(&0);
        let q_growth = if q_first > 0.0 {
            (q_last / q_first - 1.0) * 100.0
        } else {
            0.0
        };
        let seg_peak = *seg_samples.iter().max().unwrap_or(&0);
        println!();
        println!("| #59 item | value | threshold | result |");
        println!("|-----------|-------|------|-------|");
        println!(
            "| PRIMARY queue: first 1/3 → last 1/3 | {q_first:.1} → {q_last:.1} ({q_growth:+.0}%) | settles (no monotone growth) | {} |",
            if q_growth <= 20.0 { "OK" } else { "EXCEEDED" }
        );
        println!("| PRIMARY queue peak | {q_peak} | a fixed upper bound | — |");
        println!(
            "| SECONDARY segments (a test of the merge ceiling) | peak {seg_peak} | ≤ 12 | {} |",
            if seg_peak <= 12 { "OK" } else { "EXCEEDED" }
        );
        println!();
        println!(
            "#59 PRIMARY CRITERION: {}",
            if q_growth <= 20.0 {
                "THE QUEUE SETTLES → 9a-2 passes on this item"
            } else {
                "THE QUEUE GROWS → not met"
            }
        );
        println!();
        println!(
            "(reference) pre-registration #49's flawed metric: {}",
            if verdict {
                "SETTLES → 9a-2 could be accepted without backpressure"
            } else {
                "ACCUMULATES → backpressure is part of 9a-2 (pre-registration #49)"
            }
        );
        // NOTE: `wait_for_background()` is deliberately ABSENT. Draining an
        // accumulated queue takes many times longer than the measurement itself
        // (on the first attempt the process hung with half an hour of queue) —
        // and the question asked is "does it accumulate", whose answer is the size
        // of the queue.
        let (stalls, stall_us) = idx.stall_stats();
        println!(
            "backpressure (#53): {stalls} insert bekletildi, toplam {:.1} s",
            stall_us as f64 / 1e6
        );
        println!(
            "end of measurement: {} segments + {} sealing (queue not drained)",
            idx.shape().0,
            idx.sealing_count()
        );
        return;
    }

    if mode == "mergewindow" {
        // The 9a-1 measurement: after moving merging to the background, how long
        // is the window in which the writer is blocked? The pre-registered
        // threshold (DECISIONS #40): the p99 of writes coinciding with the merge
        // window must not exceed 50x the baseline p99. THE MEASUREMENT CONDITION
        // is kept identical to phase 8 (WAL group:20, NO commit inside the loop) —
        // an fsync-inclusive measurement would blow the threshold no matter how
        // good 9a was.
        use vector_gvector::index::segmented::SegmentedIndex;
        use vector_gvector::meta::Metadata;
        use vector_gvector::storage::wal::SyncPolicy;

        let dir = std::path::PathBuf::from("data/fullscale");
        if !dir.join("MANIFEST").exists() {
            eprintln!("data/fullscale is missing — run `report -- fullscale 1000000 99` first");
            return;
        }
        let idx = SegmentedIndex::open_durable(
            dir.clone(),
            dim,
            metric,
            HnswParams::default(),
            125_000,
            // The pre-registered condition: group:20 (same as phase 8).
            // GVDB_DIAG_NOWAL=1 is for DIAGNOSIS only: is the source of the
            // 6-10 ms spikes fsync, or the write path itself? The acceptance
            // decision is ALWAYS made from the pre-registered condition.
            if std::env::var("GVDB_DIAG_NOWAL").is_ok() {
                SyncPolicy::None
            } else {
                SyncPolicy::Group { window_ms: 20 }
            },
        )
        .expect("open");
        if std::env::var("GVDB_DIAG_NOWAL").is_ok() {
            println!("** DIAGNOSTIC RUN: WAL sync off — NOT the pre-registered condition **");
        }
        let (seg0, buf0) = idx.shape();
        println!(
            "start: {seg0} segments + {buf0} buffer, {} records, ceiling 8",
            idx.len_shared()
        );

        // Warmup: a few thousand writes (so the baseline p99 is measured warm).
        //
        // Id base: `data/fullscale` is a PERSISTENT directory that every run
        // writes into. If the base were derived from the record count (as it once
        // was), the ranges would overlap as runs accumulated and the measurement
        // would blow up with DuplicateId — which it duly did. A clock-based base
        // gives each run a non-colliding region. Determinism is not harmed: ids
        // are mere labels, the measured quantities (latency, windows) do not
        // depend on their values, and the vector data and seed stay fixed.
        let base_id = 1_000_000_000u64
            + std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64 % 500_000_000)
                .unwrap_or(0);
        for i in 0..5_000usize {
            idx.insert_with_meta(
                VectorId(base_id + i as u64),
                &base[i % base.len()],
                Metadata::new(),
            )
            .expect("warmup insert");
        }

        // Measurement: write enough to cross the sealing threshold → seal +
        // (background) merge.
        let extra = idx.seal_threshold() + 5_000;
        println!("measuring {extra} writes (including samples before and after the window)...");
        let mut lats: Vec<std::time::Duration> = Vec::with_capacity(extra);
        // #61: backpressure-induced waits are a SEPARATE item. The stall counter
        // is read before and after every write (two relaxed atomic loads, on the
        // order of nanoseconds) so that "a defect" and "a deliberate restriction"
        // are not summed into one number (the metric flaw recorded in #60).
        let mut stalled: Vec<bool> = Vec::with_capacity(extra);
        let t_all = Instant::now();
        for i in 0..extra {
            let id = VectorId(base_id + 1_000_000 + i as u64);
            let (c0, _) = idx.stall_stats();
            let t = Instant::now();
            idx.insert_with_meta(id, &base[i % base.len()], Metadata::new())
                .expect("insert");
            lats.push(t.elapsed());
            let (c1, _) = idx.stall_stats();
            stalled.push(c1 > c0);
        }
        let wall = t_all.elapsed();
        // To find the SOURCE of the 6 ms spike: the index of the 5 slowest writes
        // is compared with the index at which sealing occurs. (The threshold does
        // not change; this is diagnosis only.) Sealing happens when the buffer
        // reaches its threshold, so that index can be computed in advance.
        // #61 PRIMARY: the longest write EXCLUDING backpressure.
        let clean: Vec<std::time::Duration> = lats
            .iter()
            .zip(stalled.iter())
            .filter(|(_, st)| !**st)
            .map(|(d, _)| *d)
            .collect();
        let bp: Vec<std::time::Duration> = lats
            .iter()
            .zip(stalled.iter())
            .filter(|(_, st)| **st)
            .map(|(d, _)| *d)
            .collect();
        let max_clean = clean.iter().max().copied().unwrap_or_default();
        let mut sorted_clean = clean.clone();
        sorted_clean.sort();
        let p99_clean = sorted_clean
            .get(sorted_clean.len() * 99 / 100)
            .copied()
            .unwrap_or_default();
        println!();
        println!("| #61 item | value |");
        println!("|-----------|-------|");
        println!("| PRIMARY: longest write excluding backpressure | {max_clean:?} |");
        println!("| p99 excluding backpressure | {p99_clean:?} |");
        println!(
            "| PRIMARY ratio (longest / p99) | {:.0}x |",
            max_clean.as_secs_f64() / p99_clean.as_secs_f64().max(1e-12)
        );
        println!(
            "| SECONDARY (NO threshold): number of stalled writes | {} |",
            bp.len()
        );
        println!(
            "| SECONDARY: longest stall | {:?} |",
            bp.iter().max().copied().unwrap_or_default()
        );
        println!(
            "| SECONDARY: total stall | {:?} |",
            bp.iter().sum::<std::time::Duration>()
        );

        let mut slowest: Vec<(usize, std::time::Duration)> =
            lats.iter().copied().enumerate().collect();
        slowest.sort_by_key(|b| std::cmp::Reverse(b.1));
        let seal_at = idx.seal_threshold().saturating_sub(buf0);
        println!();
        println!("diagnosis: sealing at write ~{seal_at}; the 5 slowest writes:");
        for (i, d) in slowest.iter().take(5) {
            println!("  #{i} → {d:?}");
        }
        idx.commit_wal().expect("commit");

        let mut sorted = lats.clone();
        sorted.sort();
        let pct = |p: f64| sorted[((sorted.len() as f64 * p) as usize).min(sorted.len() - 1)];
        let max_lat = *sorted.last().unwrap();
        // Baseline: p99 excluding the 3 longest calls (those triggering
        // seal/merge)
        let cut = sorted.len().saturating_sub(3);
        let base_p99 = sorted[(cut as f64 * 0.99) as usize];
        let ratio = max_lat.as_secs_f64() / base_p99.as_secs_f64().max(1e-9);

        println!();
        println!("| measurement | value |");
        println!("|-------|-------|");
        println!("| total time (writer thread) | {wall:.2?} |");
        println!("| taban p50 | {:?} |", pct(0.5));
        println!("| baseline p99 (excluding seal/merge calls) | {base_p99:?} |");
        println!("| **en uzun tek yazma** | **{max_lat:.3?}** |");
        println!(
            "| last sealing (the part blocking the writer) | {:?} |",
            std::time::Duration::from_micros(idx.last_seal_us())
        );
        println!(
            "| pre-registered 50x threshold | **{}** |",
            if ratio <= 50.0 { "MET" } else { "NOT MET" }
        );
        // The merge statistics are only meaningful once the worker has finished:
        // at measurement time a merge may still be running in the background (it
        // showed up as 0 on the first run).
        let merge_running_at_end = idx.merge_in_flight();
        idx.wait_for_merge();
        println!(
            "| last merge (BACKGROUND, does NOT block the writer) | {:?} |",
            std::time::Duration::from_micros(idx.last_merge_us())
        );
        println!("| merge count | {} |", idx.merge_count());
        println!(
            "| was a merge running as the measurement ended | {} |",
            if merge_running_at_end {
                "YES (overlap confirmed)"
            } else {
                "no"
            }
        );
        let (seg1, buf1) = idx.shape();
        println!();
        println!(
            "after the merge: {seg1} segments + {buf1} buffer, {} records",
            idx.len_shared()
        );
        return;
    }

    if mode == "int8scale" {
        // Phase 8a: int8 multi-reader scaling. Hypothesis (corrected): int8
        // shrinks the working set by ~2x (the graph is NOT quantized), not 3-4x.
        // The pre-registered acceptance threshold: 8 threads / 1 thread ≥ 2.0 →
        // GO.
        //
        // A fair comparison: the f32 side was measured with 8 segments, so the
        // int8 side is measured at the SAME segment count (comparing against a
        // single-graph int8
        // would credit int8 with the 1→8 segment difference from segcurve).
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use vector_gvector::index::quant::QuantizedHnsw;
        use vector_gvector::index::segmented::SegmentedIndex;
        use vector_gvector::types::SearchResult;

        let dir = std::path::PathBuf::from("data/fullscale");
        if !dir.join("MANIFEST").exists() {
            eprintln!("data/fullscale is missing — run `report -- fullscale 1000000 99` first");
            return;
        }
        println!(
            "machine: {} logical cores",
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(0)
        );
        let t = Instant::now();
        let idx = SegmentedIndex::open_or_create(
            dir.clone(),
            dim,
            metric,
            HnswParams::default(),
            125_000,
        )
        .expect("open");
        let (n_seg, n_buf) = idx.shape();
        println!(
            "f32 index: {n_seg} segments + {n_buf} buffer, {} records ({:.1?})",
            idx.len_shared(),
            t.elapsed()
        );
        let f32_mem = idx.memory_bytes();

        // Ground truth: the official one for the full set, otherwise exact.
        let truth: Vec<Vec<VectorId>> = if base.len() == 1_000_000 {
            read_ivecs(std::path::Path::new("data/sift/sift_groundtruth.ivecs"))
                .expect("gt")
                .iter()
                .take(queries.len())
                .map(|row| row.iter().take(k).map(|&i| VectorId(i as u64)).collect())
                .collect()
        } else {
            ground_truth(&base, &queries, k, metric)
        };

        // The multi-reader measurement core.
        //
        // METHODOLOGY NOTE: the first version of this measurement was NOT
        // REPRODUCIBLE — same code, same data, two runs: f32 scaling of 5.08x and
        // 1.14x. The cause: opening a second large index in the same process
        // (memory pressure + cache pollution). Phase 8's finding that "reads do
        // not scale at 1M" had likewise been measured in a process that had run
        // for five minutes with RSS at 3.1 GB. The fix: warmup plus the MEDIAN of
        // three repeats.
        let bench =
            |threads: usize, search: &(dyn Fn(&[f32]) -> Vec<SearchResult> + Sync)| -> f64 {
                let run = |secs: u64| -> f64 {
                    let stop = AtomicBool::new(false);
                    let count = AtomicUsize::new(0);
                    std::thread::scope(|sc| {
                        for t in 0..threads {
                            let (stop, count, queries) = (&stop, &count, &queries);
                            sc.spawn(move || {
                                let mut i = 0usize;
                                while !stop.load(Ordering::Relaxed) {
                                    std::hint::black_box(search(&queries[(i + t) % queries.len()]));
                                    i += 1;
                                    count.fetch_add(1, Ordering::Relaxed);
                                }
                            });
                        }
                        std::thread::sleep(std::time::Duration::from_secs(secs));
                        stop.store(true, Ordering::Relaxed);
                    });
                    count.load(Ordering::Relaxed) as f64 / secs as f64
                };
                run(1); // warmup: let the cache warm, excluded from the measurement
                let mut s = [run(3), run(3), run(3)];
                s.sort_by(f64::total_cmp);
                s[1] // the median — against the noise of a single run
            };

        let merge_hits = |mut all: Vec<SearchResult>, k: usize| -> Vec<SearchResult> {
            all.sort();
            let mut seen = std::collections::HashSet::with_capacity(all.len());
            all.retain(|r| seen.insert(r.id));
            all.truncate(k);
            all
        };

        println!();
        println!("| index | ef | 1 thread | 2 | 4 | 8 | scaling (8/1) | recall@10 |");
        println!("|--------|----|----------|---|---|---|------------------|-----------|");

        // --- the f32 measurement (before converting to int8) ---
        let mut f32_rows = Vec::new();
        for ef in [50usize, 100] {
            let search = |q: &[f32]| idx.search_shared_with_ef(q, k, ef);
            let results: Vec<Vec<VectorId>> = queries
                .iter()
                .map(|q| search(q).iter().map(|r| r.id).collect())
                .collect();
            let recall = recall_at_k(&results, &truth, k);
            let mut qps = Vec::new();
            for th in [1usize, 2, 4, 8] {
                qps.push(bench(th, &search));
            }
            println!(
                "| f32 | {ef} | {:.0} | {:.0} | {:.0} | {:.0} | {:.2}x | {recall:.4} |",
                qps[0],
                qps[1],
                qps[2],
                qps[3],
                qps[3] / qps[0].max(1.0)
            );
            f32_rows.push((ef, qps, recall));
        }

        // --- convert to int8 and DROP the f32 (so it does not pollute the cache) ---
        let t = Instant::now();
        let quantized: Vec<QuantizedHnsw> = idx.quantize_segments();
        let qt = t.elapsed();
        let int8_mem: usize = quantized
            .iter()
            .map(|q| {
                let (c, l) = q.memory_bytes();
                c + l
            })
            .sum();
        drop(idx);
        println!();
        println!(
            "quantize: {qt:.1?} — working set f32 {:.0} MB → int8 {:.0} MB ({:.2}x)",
            f32_mem as f64 / 1048576.0,
            int8_mem as f64 / 1048576.0,
            f32_mem as f64 / int8_mem.max(1) as f64
        );
        println!();

        let mut int8_rows = Vec::new();
        for ef in [50usize, 100] {
            let search = |q: &[f32]| {
                let mut all = Vec::new();
                for seg in &quantized {
                    all.extend(seg.search_with_ef(q, k, ef));
                }
                merge_hits(all, k)
            };
            let results: Vec<Vec<VectorId>> = queries
                .iter()
                .map(|q| search(q).iter().map(|r| r.id).collect())
                .collect();
            let recall = recall_at_k(&results, &truth, k);
            let mut qps = Vec::new();
            for th in [1usize, 2, 4, 8] {
                qps.push(bench(th, &search));
            }
            println!(
                "| int8 | {ef} | {:.0} | {:.0} | {:.0} | {:.0} | {:.2}x | {recall:.4} |",
                qps[0],
                qps[1],
                qps[2],
                qps[3],
                qps[3] / qps[0].max(1.0)
            );
            int8_rows.push((ef, qps, recall));
        }

        println!();
        for ((ef, fq, fr), (_, iq, ir)) in f32_rows.iter().zip(&int8_rows) {
            let scale = iq[3] / iq[0].max(1.0);
            println!(
                "ef={ef}: int8 scaling {scale:.2}x (threshold ≥2.0 → {}), \
                 8-thread QPS {:.0} → {:.0} ({:.2}x), recall {fr:.4} → {ir:.4} (loss {:.4})",
                if scale >= 2.0 { "GO" } else { "NO-GO" },
                fq[3],
                iq[3],
                iq[3] / fq[3].max(1.0),
                fr - ir
            );
        }
        return;
    }

    if mode == "coldprofile" {
        // For the 9b decision: the components of cold start. mmap can only remove
        // the "file read + vector copy" part; graph parsing and metadata rebuild
        // remain. This is the UPPER BOUND on the gain.
        use vector_gvector::storage::{read_verified, Manifest};
        let dir = std::path::PathBuf::from("data/fullscale");
        let manifest = Manifest::read(&dir)
            .expect("manifest")
            .expect("manifest yok");
        println!(
            "generation={}, {} segment",
            manifest.generation,
            manifest.segments.len()
        );

        let t = Instant::now();
        let mut blobs = Vec::new();
        let mut total = 0usize;
        for s in &manifest.segments {
            let b = read_verified(&dir, &s.file, s.crc32).expect("segment oku");
            total += b.len();
            blobs.push(b);
        }
        let t_read = t.elapsed();
        println!(
            "(a) reading segment files + CRC verification: {t_read:.2?} ({:.0} MB)",
            total as f64 / 1048576.0
        );

        let t = Instant::now();
        let mut n_rec = 0usize;
        for b in &blobs {
            let h = HnswIndex::load_from_bytes(b).expect("parse");
            n_rec += h.len();
        }
        let t_parse = t.elapsed();
        println!("(b) segment parse (graph + vector copy): {t_parse:.2?} ({n_rec} records)");
        drop(blobs);

        let t = Instant::now();
        let mfile = manifest.metadata_file.clone().expect("metadata file");
        let mbytes = read_verified(&dir, &mfile, manifest.metadata_crc).expect("meta oku");
        let entries =
            vector_gvector::storage::decode_metadata(&mbytes, &dir.join(&mfile)).expect("decode");
        let t_meta_read = t.elapsed();
        println!(
            "(c) metadata read + decode: {t_meta_read:.2?} ({} records, {:.0} MB)",
            entries.len(),
            mbytes.len() as f64 / 1048576.0
        );

        let t = Instant::now();
        let full = vector_gvector::index::segmented::SegmentedIndex::open_or_create(
            dir.clone(),
            dim,
            metric,
            HnswParams::default(),
            125_000,
        )
        .expect("open");
        let t_total = t.elapsed();
        println!("(d) full open (a+b+c+derived indexes): {t_total:.2?}");
        let derived = t_total.saturating_sub(t_read + t_parse + t_meta_read);
        println!("    → building the derived indexes ≈ {derived:.2?}");
        println!();
        println!(
            "the UPPER BOUND mmap could remove ≈ (a) + the vector share of (b) = {:.2?} + partial",
            t_read
        );
        println!(
            "the 9b threshold: gain ≥ 40% AND ≥ 2 s. Total {t_total:.2?} → 40% = {:.2?}",
            t_total.mul_f64(0.4)
        );
        println!("    (verification: {} records opened)", full.len_shared());
        return;
    }

    if mode == "fullscale" {
        full_scale(&base, &queries, k, metric);
        return;
    }

    if mode == "wal" {
        // The phase 7c measurement: fsync policy × write throughput + replay.
        use vector_gvector::index::segmented::SegmentedIndex;
        use vector_gvector::meta::{MetaValue, Metadata};
        use vector_gvector::storage::wal::SyncPolicy;

        let n = base.len().min(20_000);
        // The server's writer task batches commands and performs a SINGLE commit
        // at the end of a batch; the measurement must model that exactly, or group
        // commit effectively degenerates into per_op and the table shows no
        // difference between the policies.
        const BATCH: usize = 64;
        println!(
            "WAL measurement: {n} inserts, batch={BATCH} (like the server writer task), sealing off"
        );
        println!();
        println!(
            "| policy | time | throughput | fsync/op | WAL size | replay time | replayed records |"
        );
        println!("|----------|------|------------|----------|------------|---------------|--------------|");
        for policy in [
            SyncPolicy::None,
            SyncPolicy::Group { window_ms: 20 },
            SyncPolicy::PerOp,
        ] {
            let dir =
                std::path::PathBuf::from(format!("data/wal-{}", policy.label().replace(':', "-")));
            let _ = std::fs::remove_dir_all(&dir);
            let idx = SegmentedIndex::open_durable(
                dir.clone(),
                dim,
                metric,
                HnswParams::default(),
                usize::MAX, // no sealing: keep HNSW construction out of the measurement
                policy,
            )
            .expect("open");
            let t = Instant::now();
            for (i, v) in base.iter().take(n).enumerate() {
                let meta: Metadata = [("v".to_string(), MetaValue::Int(i as i64))].into();
                idx.insert_with_meta(VectorId(i as u64), v, meta)
                    .expect("insert");
                // End-of-batch commit: responses are only sent after this.
                if (i + 1) % BATCH == 0 {
                    idx.commit_wal().expect("commit");
                }
            }
            idx.commit_wal().expect("son commit");
            let elapsed = t.elapsed();
            let fsync_per_op = match policy {
                SyncPolicy::PerOp => 1.0,
                SyncPolicy::Group { .. } => 1.0 / BATCH as f64,
                SyncPolicy::None => 0.0,
            };
            let wal_bytes = idx.wal_len_bytes();
            drop(idx);

            let t = Instant::now();
            let reopened = SegmentedIndex::open_durable(
                dir,
                dim,
                metric,
                HnswParams::default(),
                usize::MAX,
                policy,
            )
            .expect("reopen");
            let replay = t.elapsed();
            println!(
                "| {} | {:.2?} | {:.0} op/s | {fsync_per_op:.3} | {:.1} MB | {:.2?} | {} |",
                policy.label(),
                elapsed,
                n as f64 / elapsed.as_secs_f64(),
                wal_bytes as f64 / 1048576.0,
                replay,
                reopened.replay_report().applied
            );
        }

        // A large WAL replay: the full base, written without fsync (replay time
        // is independent of the policy — the read path is the same).
        if base.len() > n {
            let dir = std::path::PathBuf::from("data/wal-bigreplay");
            let _ = std::fs::remove_dir_all(&dir);
            let idx = SegmentedIndex::open_durable(
                dir.clone(),
                dim,
                metric,
                HnswParams::default(),
                usize::MAX,
                SyncPolicy::None,
            )
            .expect("open");
            for (i, v) in base.iter().enumerate() {
                let meta: Metadata = [("v".to_string(), MetaValue::Int(i as i64))].into();
                idx.insert_with_meta(VectorId(i as u64), v, meta)
                    .expect("insert");
            }
            idx.flush_wal().expect("flush");
            let wal_mb = idx.wal_len_bytes() as f64 / 1048576.0;
            drop(idx);
            let t = Instant::now();
            let re = SegmentedIndex::open_durable(
                dir,
                dim,
                metric,
                HnswParams::default(),
                usize::MAX,
                SyncPolicy::None,
            )
            .expect("reopen");
            println!();
            println!(
                "full WAL replay: {} records / {wal_mb:.1} MB → {:?} ({:.0} records/s)",
                re.replay_report().applied,
                t.elapsed(),
                base.len() as f64 / t.elapsed().as_secs_f64()
            );
        }
        return;
    }

    if mode == "durability" {
        // The phase 7a measurement: checkpoint + cold-start cost.
        use vector_gvector::index::segmented::SegmentedIndex;
        use vector_gvector::meta::{MetaValue, Metadata};
        let dir = std::path::PathBuf::from("data/durability");
        let _ = std::fs::remove_dir_all(&dir);
        let idx = SegmentedIndex::open_or_create(
            dir.clone(),
            dim,
            metric,
            HnswParams::default(),
            base.len() / 8,
        )
        .expect("open");
        let t = Instant::now();
        for (i, v) in base.iter().enumerate() {
            let meta: Metadata = [
                ("grup".to_string(), MetaValue::Int((i % 8) as i64)),
                ("v".to_string(), MetaValue::Int(i as i64)),
                ("f".to_string(), MetaValue::Float(i as f64 * 0.5)),
            ]
            .into();
            idx.insert_with_meta(VectorId(i as u64), v, meta)
                .expect("insert");
        }
        println!("build (with 3 metadata fields): {:?}", t.elapsed());

        let t = Instant::now();
        let g1 = idx.checkpoint().expect("checkpoint");
        let ck1 = t.elapsed();
        let disk: u64 = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok()?.metadata().ok().map(|m| m.len()))
            .sum();
        println!(
            "first checkpoint (gen={g1}): {ck1:?}, disk {:.1} MB ({:.0} B/vector)",
            disk as f64 / 1048576.0,
            disk as f64 / base.len() as f64
        );

        // Second checkpoint: no new segments at all → metadata+manifest only
        let t = Instant::now();
        let g2 = idx.checkpoint().expect("checkpoint2");
        println!(
            "ikinci checkpoint (gen={g2}, yeni segment yok): {:?}",
            t.elapsed()
        );
        let (n_seg, _) = idx.shape();
        drop(idx);

        let t = Instant::now();
        let reopened =
            SegmentedIndex::open_or_create(dir.clone(), dim, metric, HnswParams::default(), 1)
                .expect("reopen");
        let cold = t.elapsed();
        println!(
            "cold start: {cold:?} ({n_seg} segments, {} records, derived indexes rebuilt)",
            reopened.len_shared()
        );
        // Correctness: does the reopened index give the same results?
        let truth = ground_truth(&base, &queries, k, metric);
        let results: Vec<Vec<VectorId>> = queries
            .iter()
            .map(|q| reopened.search_shared(q, k).iter().map(|r| r.id).collect())
            .collect();
        println!(
            "recall@{k} after reopening = {:.4}",
            recall_at_k(&results, &truth, k)
        );
        return;
    }

    if mode == "rangefilter" {
        // The Range histogram measurement (acceptance criteria of DECISIONS #31):
        // uniform + skewed (log-normal) distributions, estimated interval vs
        // truth, a correlated Eq+Range cell, the arm agreement rate, and the
        // maintenance cost.
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};
        use std::collections::HashSet;
        use vector_gvector::index::segmented::SegmentedIndex;
        use vector_gvector::meta::{Filter, MetaValue, Metadata, Predicate};

        let n = base.len();
        // log-normal alan: exp(N(0,1)), deterministik
        let mut rng = StdRng::seed_from_u64(DEFAULT_SEED + 2);
        let lognormal: Vec<f64> = (0..n)
            .map(|_| {
                // Box-Muller ile N(0,1)
                let u1: f64 = rng.gen_range(1e-12..1.0);
                let u2: f64 = rng.gen_range(0.0..1.0);
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z.exp()
            })
            .collect();

        // Maintenance cost: a build without metadata vs one with two numeric
        // fields.
        let t = Instant::now();
        let plain = SegmentedIndex::new(dim, metric, HnswParams::default(), n / 4);
        for (i, v) in base.iter().enumerate() {
            plain.insert_shared(VectorId(i as u64), v).expect("insert");
        }
        let t_plain = t.elapsed();
        drop(plain);
        let t = Instant::now();
        let idx = SegmentedIndex::new(dim, metric, HnswParams::default(), n / 4);
        for (i, v) in base.iter().enumerate() {
            let meta: Metadata = [
                ("v".to_string(), MetaValue::Int(i as i64)),
                ("lv".to_string(), MetaValue::Float(lognormal[i])),
                (
                    "par".to_string(),
                    MetaValue::Int(if i < n / 2 { 0 } else { 1 }),
                ),
            ]
            .into();
            idx.insert_with_meta(VectorId(i as u64), v, meta)
                .expect("insert");
        }
        let t_meta = t.elapsed();
        println!(
            "maintenance cost: build without metadata {t_plain:.1?}, with 3 fields (2 numeric) {t_meta:.1?} (+{:.0}%)",
            (t_meta.as_secs_f64() / t_plain.as_secs_f64() - 1.0) * 100.0
        );

        let mut bf = BruteForceIndex::new(dim, metric);
        for (i, v) in base.iter().enumerate() {
            bf.insert(VectorId(i as u64), v).expect("insert");
        }
        let scan_limit = (16 * k).max(n / 20);
        println!("scan_limit = {scan_limit}");
        println!();
        println!(
            "| field | s | truth | estimate [lower,upper] | upper/truth | arm (oracle) | recall | p50 |"
        );
        println!(
            "|------|---|--------|------------------|-----------|--------------|--------|-----|"
        );

        let mut agree = 0usize;
        let mut rows = 0usize;
        let mut sorted_lv = lognormal.clone();
        sorted_lv.sort_by(f64::total_cmp);

        for s in [0.001f64, 0.01, 0.05, 0.1, 0.3, 0.5] {
            let m = ((s * n as f64) as usize).max(1);
            // (label, filter, the true matching id set, the estimation query)
            type RangeCase<'a> = (&'a str, Filter, HashSet<u64>, (&'a str, f64, f64));
            let cases: Vec<RangeCase> = vec![
                (
                    "v(uniform)",
                    Filter {
                        must: vec![Predicate::Range {
                            key: "v".into(),
                            min: 0.0,
                            max: (m - 1) as f64,
                        }],
                    },
                    (0..m as u64).collect(),
                    ("v", 0.0, (m - 1) as f64),
                ),
                (
                    "lv(skewed)",
                    {
                        let cutoff = sorted_lv[m - 1];
                        Filter {
                            must: vec![Predicate::Range {
                                key: "lv".into(),
                                min: f64::NEG_INFINITY,
                                max: cutoff,
                            }],
                        }
                    },
                    {
                        let cutoff = sorted_lv[m - 1];
                        (0..n as u64)
                            .filter(|&i| lognormal[i as usize] <= cutoff)
                            .collect()
                    },
                    ("lv", f64::NEG_INFINITY, sorted_lv[m - 1]),
                ),
            ];
            for (label, filter, truth_set, (ekey, elo, ehi)) in cases {
                let truth_n = truth_set.len();
                let (est_l, est_u) = idx.debug_range_estimate(ekey, elo, ehi);
                let arm = idx.debug_plan_arm(&filter, k);
                let oracle = if truth_n <= scan_limit {
                    "scan"
                } else {
                    "post"
                };
                agree += (arm == oracle) as usize;
                rows += 1;
                let allow = |id: VectorId| truth_set.contains(&id.0);
                let mut hits = 0usize;
                let mut total = 0usize;
                for q in &queries {
                    let tr: Vec<VectorId> = bf
                        .search_filtered(q, k, &allow)
                        .iter()
                        .map(|r| r.id)
                        .collect();
                    let res = idx.search_filtered(q, k, &filter);
                    hits += res.iter().filter(|r| tr.contains(&r.id)).count();
                    total += tr.len();
                }
                let stats = measure_latency(&queries, |q| {
                    std::hint::black_box(idx.search_filtered(q, k, &filter));
                });
                println!(
                    "| {label} | {s} | {truth_n} | [{est_l},{est_u}] | {:.2} | {arm} ({oracle}) | {:.3} | {:?} |",
                    est_u as f64 / truth_n.max(1) as f64,
                    hits as f64 / total.max(1) as f64,
                    stats.p50
                );
            }
        }

        // The correlated cell: Eq(par=0) [i < n/2] ∧ Range v∈[0.4n, 0.6n) — the
        // true intersection is 0.1n, while the independence/min-upper estimate is
        // ~0.2n → a ratio of ~2.
        let f_corr = Filter {
            must: vec![
                Predicate::Eq {
                    key: "par".into(),
                    value: MetaValue::Int(0),
                },
                Predicate::Range {
                    key: "v".into(),
                    min: (n as f64) * 0.4,
                    max: (n as f64) * 0.6 - 1.0,
                },
            ],
        };
        let truth_corr: HashSet<u64> = ((n as u64 * 2 / 5)..(n as u64 / 2)).collect();
        let (el, eu) = idx.debug_range_estimate("v", n as f64 * 0.4, n as f64 * 0.6 - 1.0);
        let arm = idx.debug_plan_arm(&f_corr, k);
        let oracle = if truth_corr.len() <= scan_limit {
            "scan"
        } else {
            "post"
        };
        agree += (arm == oracle) as usize;
        rows += 1;
        let allow = |id: VectorId| truth_corr.contains(&id.0);
        let mut hits = 0;
        let mut total = 0;
        for q in &queries {
            let tr: Vec<VectorId> = bf
                .search_filtered(q, k, &allow)
                .iter()
                .map(|r| r.id)
                .collect();
            let res = idx.search_filtered(q, k, &f_corr);
            hits += res.iter().filter(|r| tr.contains(&r.id)).count();
            total += tr.len();
        }
        let stats = measure_latency(&queries, |q| {
            std::hint::black_box(idx.search_filtered(q, k, &f_corr));
        });
        println!(
            "| Eq∧Range (correlated) | 0.1 | {} | range:[{el},{eu}] min-upper:{} | {:.2} | {arm} ({oracle}) | {:.3} | {:?} |",
            truth_corr.len(),
            eu.min(n / 2),
            eu.min(n / 2) as f64 / truth_corr.len().max(1) as f64,
            hits as f64 / total.max(1) as f64,
            stats.p50
        );
        println!();
        println!(
            "arm agreement: {agree}/{rows} ({:.0}%)",
            agree as f64 / rows as f64 * 100.0
        );
        return;
    }

    if mode == "mergecost" {
        // The cost of the ceiling guard: same data, with and without a ceiling.
        use vector_gvector::index::segmented::SegmentedIndex;
        let seal = base.len() / 10; // produces 10 segments without a ceiling
        for (label, ceiling) in [("no ceiling", 100usize), ("ceiling=8", 8)] {
            let mut idx = SegmentedIndex::new(dim, metric, HnswParams::default(), seal);
            idx.set_max_segments(ceiling);
            let t = Instant::now();
            for (i, v) in base.iter().enumerate() {
                idx.insert_shared(VectorId(i as u64), v).expect("insert");
            }
            let build = t.elapsed();
            let (n_seg, n_buf) = idx.shape();
            let stats = measure_latency(&queries, |q| {
                std::hint::black_box(idx.search_shared(q, k));
            });
            println!(
                "{label}: build {build:.1?}, {n_seg} segments (+{n_buf} buffer), \
                 arama p50 {:?}, bellek {:.0} MB",
                stats.p50,
                idx.memory_bytes() as f64 / 1048576.0
            );
        }
        // Peak memory (analytical): during a merge the two sources and the merged
        // segment coexist. Worst case with two equal segments: peak ≈ steady +
        // 2×seg.
        let seg_bytes = (seal * (dim * 4 + 404)) as f64 / 1048576.0;
        println!(
            "peak merge memory ≈ steady + 2×{seg_bytes:.0} MB (the two sources \
             live until the swap; the merged one is already part of steady state)"
        );
        return;
    }

    if mode == "segcurve" {
        // The segment count × latency/recall curve (an input to the merge
        // policy). The same data is split with different seal thresholds; searches
        // are unfiltered.
        use vector_gvector::index::segmented::SegmentedIndex;
        let truth = ground_truth(&base, &queries, k, metric);
        println!("| segments | p50 | p99 | recall@{k} | build |");
        println!("|---------|-----|-----|-----------|------|");
        for parts in [1usize, 2, 4, 5, 8, 10] {
            let seal = base.len().div_ceil(parts);
            let idx = SegmentedIndex::new(dim, metric, HnswParams::default(), seal);
            let t = Instant::now();
            for (i, v) in base.iter().enumerate() {
                idx.insert_shared(VectorId(i as u64), v).expect("insert");
            }
            let build = t.elapsed();
            let (n_seg, n_buf) = idx.shape();
            let results: Vec<Vec<VectorId>> = queries
                .iter()
                .map(|q| idx.search_shared(q, k).iter().map(|r| r.id).collect())
                .collect();
            let recall = recall_at_k(&results, &truth, k);
            let stats = measure_latency(&queries, |q| {
                std::hint::black_box(idx.search_shared(q, k));
            });
            println!(
                "| {n_seg} (+{n_buf} buffer) | {:?} | {:?} | {recall:.4} | {build:.1?} |",
                stats.p50, stats.p99
            );
        }
        return;
    }

    if mode == "delete" {
        // Phase 4 validation: recall after 20% deletion + the memory effect of
        // compaction.
        let mut hnsw = HnswIndex::new(
            dim,
            metric,
            HnswParams {
                tombstone_threshold: 2.0, // disable the automatic one to compact manually
                ..Default::default()
            },
        );
        let mut bf = BruteForceIndex::new(dim, metric);
        for (i, v) in base.iter().enumerate() {
            hnsw.insert(VectorId(i as u64), v).expect("insert");
            bf.insert(VectorId(i as u64), v).expect("insert");
        }
        let recall_of = |hnsw: &HnswIndex, bf: &BruteForceIndex| {
            let truth: Vec<Vec<VectorId>> = queries
                .iter()
                .map(|q| bf.search(q, k).iter().map(|r| r.id).collect())
                .collect();
            let results: Vec<Vec<VectorId>> = queries
                .iter()
                .map(|q| hnsw.search_with_ef(q, k, 50).iter().map(|r| r.id).collect())
                .collect();
            recall_at_k(&results, &truth, k)
        };
        println!(
            "recall@{k} before deletion (ef=50) = {:.4}",
            recall_of(&hnsw, &bf)
        );
        for i in (0..base.len()).step_by(5) {
            hnsw.delete(VectorId(i as u64)).expect("hnsw delete");
            bf.delete(VectorId(i as u64)).expect("bf delete");
        }
        println!(
            "recall@{k} after 20% deletion = {:.4} (tombstone ratio {:.2})",
            recall_of(&hnsw, &bf),
            hnsw.tombstone_ratio()
        );
        let (v0, l0) = hnsw.memory_bytes();
        let t = Instant::now();
        hnsw.compact();
        println!(
            "compaction: {:?}; bellek vec {:.1}→{:.1} MB, link {:.1}→{:.1} MB",
            t.elapsed(),
            v0 as f64 / 1048576.0,
            hnsw.memory_bytes().0 as f64 / 1048576.0,
            l0 as f64 / 1048576.0,
            hnsw.memory_bytes().1 as f64 / 1048576.0
        );
        println!("recall@{k} after compaction = {:.4}", recall_of(&hnsw, &bf));
        return;
    }

    if mode == "sift1m" {
        // Tam 1M stres testi: resmi ground truth (sift_groundtruth.ivecs)
        // is VALID here — unlike with subsets, we do not generate it ourselves.
        use vector_gvector::index::quant::QuantizedHnsw;
        let gt = read_ivecs(std::path::Path::new("data/sift/sift_groundtruth.ivecs"))
            .expect("could not read the ground truth");
        let truth: Vec<Vec<VectorId>> = gt
            .iter()
            .take(queries.len())
            .map(|row| row.iter().take(k).map(|&i| VectorId(i as u64)).collect())
            .collect();
        assert_eq!(truth.len(), queries.len(), "GT/query counts must match");

        let t = Instant::now();
        let mut hnsw = HnswIndex::new(dim, metric, HnswParams::default());
        for (i, v) in base.iter().enumerate() {
            hnsw.insert(VectorId(i as u64), v).expect("insert");
            if (i + 1) % 100_000 == 0 {
                println!("  insert {} / {} ({:?})", i + 1, base.len(), t.elapsed());
            }
        }
        println!("build: {:?} ({} vectors)", t.elapsed(), hnsw.len());
        let (vmem, lmem) = hnsw.memory_bytes();
        println!(
            "memory: vectors {:.0} MB + graph {:.0} MB (graph {:.0} B/vector)",
            vmem as f64 / 1048576.0,
            lmem as f64 / 1048576.0,
            lmem as f64 / base.len() as f64
        );
        for ef in [50, 100, 200] {
            let results: Vec<Vec<VectorId>> = queries
                .iter()
                .map(|q| hnsw.search_with_ef(q, k, ef).iter().map(|r| r.id).collect())
                .collect();
            let recall = recall_at_k(&results, &truth, k);
            let stats = measure_latency(&queries, |q| {
                std::hint::black_box(hnsw.search_with_ef(q, k, ef));
            });
            println!(
                "f32  ef={ef}: recall@{k}={recall:.4} p50={:?} p99={:?}",
                stats.p50, stats.p99
            );
        }
        let t = Instant::now();
        let quant = QuantizedHnsw::from_hnsw(&hnsw);
        drop(hnsw); // drop the f32 copy: only the codes + graph stay in memory
        println!("quantize: {:?}", t.elapsed());
        let (cmem, qlmem) = quant.memory_bytes();
        println!(
            "quantize bellek: kod {:.0} MB + graf {:.0} MB",
            cmem as f64 / 1048576.0,
            qlmem as f64 / 1048576.0
        );
        for ef in [50, 100, 200] {
            let results: Vec<Vec<VectorId>> = queries
                .iter()
                .map(|q| {
                    quant
                        .search_with_ef(q, k, ef)
                        .iter()
                        .map(|r| r.id)
                        .collect()
                })
                .collect();
            let recall = recall_at_k(&results, &truth, k);
            let stats = measure_latency(&queries, |q| {
                std::hint::black_box(quant.search_with_ef(q, k, ef));
            });
            println!(
                "int8 ef={ef}: recall@{k}={recall:.4} p50={:?} p99={:?}",
                stats.p50, stats.p99
            );
        }
        return;
    }

    if mode == "quant" {
        use vector_gvector::index::quant::QuantizedHnsw;
        let truth = ground_truth(&base, &queries, k, metric);
        let mut hnsw = HnswIndex::new(dim, metric, HnswParams::default());
        for (i, v) in base.iter().enumerate() {
            hnsw.insert(VectorId(i as u64), v).expect("insert");
        }
        let (f32_vec_mem, link_mem) = hnsw.memory_bytes();
        let t = Instant::now();
        let quant = QuantizedHnsw::from_hnsw(&hnsw);
        println!("quantize (kalibrasyon + kodlama): {:?}", t.elapsed());
        let (code_mem, qlink_mem) = quant.memory_bytes();

        println!();
        println!("| index | ef | recall@{k} | p50 | p99 | vectors MB | total MB |");
        println!("|--------|----|-----------|-----|-----|-----------|-----------|");
        for ef in [25, 50, 100] {
            for use_quant in [false, true] {
                let search = |q: &[f32]| -> Vec<VectorId> {
                    if use_quant {
                        quant
                            .search_with_ef(q, k, ef)
                            .iter()
                            .map(|r| r.id)
                            .collect()
                    } else {
                        hnsw.search_with_ef(q, k, ef).iter().map(|r| r.id).collect()
                    }
                };
                let results: Vec<Vec<VectorId>> = queries.iter().map(|q| search(q)).collect();
                let recall = recall_at_k(&results, &truth, k);
                let stats = measure_latency(&queries, |q| {
                    std::hint::black_box(search(q));
                });
                let (name, vmem, lmem) = if use_quant {
                    ("int8", code_mem, qlink_mem)
                } else {
                    ("f32", f32_vec_mem, link_mem)
                };
                println!(
                    "| {name} | {ef} | {recall:.4} | {:?} | {:?} | {:.1} | {:.1} |",
                    stats.p50,
                    stats.p99,
                    vmem as f64 / 1048576.0,
                    (vmem + lmem) as f64 / 1048576.0
                );
            }
        }
        return;
    }

    if mode == "concurrent" {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        use std::sync::Arc;
        use vector_gvector::index::segmented::SegmentedIndex;

        let idx = Arc::new(SegmentedIndex::new(
            dim,
            metric,
            HnswParams::default(),
            20_000, // 100K → 5 segment
        ));
        let t = Instant::now();
        for (i, v) in base.iter().enumerate() {
            idx.insert_shared(VectorId(i as u64), v).expect("insert");
        }
        let (n_seg, n_buf) = idx.shape();
        println!(
            "build: {:?} ({n_seg} segments + {n_buf} buffer)",
            t.elapsed()
        );

        // recall check (correctness of segment merging)
        let truth = ground_truth(&base, &queries, k, metric);
        let results: Vec<Vec<VectorId>> = queries
            .iter()
            .map(|q| idx.search_shared(q, k).iter().map(|r| r.id).collect())
            .collect();
        println!("recall@{k} = {:.4}", recall_at_k(&results, &truth, k));

        // throughput: each thread searches for 3 s and the total queries are counted
        let measure_qps = |threads: usize, with_writer: bool| -> f64 {
            let stop = AtomicBool::new(false);
            let total = AtomicUsize::new(0);
            std::thread::scope(|s| {
                for t in 0..threads {
                    let idx = &idx;
                    let stop = &stop;
                    let total = &total;
                    let queries = &queries;
                    s.spawn(move || {
                        let mut n = 0usize;
                        while !stop.load(Ordering::Relaxed) {
                            let q = &queries[(n + t) % queries.len()];
                            std::hint::black_box(idx.search_shared(q, k));
                            n += 1;
                        }
                        total.fetch_add(n, Ordering::Relaxed);
                    });
                }
                if with_writer {
                    let idx = &idx;
                    let stop = &stop;
                    let base = &base;
                    s.spawn(move || {
                        // writer: a delete + re-insert loop (net size stays constant)
                        let mut i = 0u64;
                        while !stop.load(Ordering::Relaxed) {
                            let id = VectorId(i % 10_000);
                            if idx.delete_shared(id).is_ok() {
                                idx.insert_shared(id, &base[(i % 10_000) as usize])
                                    .expect("yeniden ekleme");
                            }
                            i += 1;
                        }
                    });
                }
                std::thread::sleep(std::time::Duration::from_secs(3));
                stop.store(true, Ordering::Relaxed);
            });
            total.load(Ordering::Relaxed) as f64 / 3.0
        };

        for threads in [1, 4, 8] {
            println!(
                "read throughput ({threads} threads, no writer): {:.0} QPS",
                measure_qps(threads, false)
            );
        }
        println!(
            "read throughput (4 threads + an active writer): {:.0} QPS",
            measure_qps(4, true)
        );
        let (n_seg, n_buf) = idx.shape();
        println!(
            "son durum: {n_seg} segment + {n_buf} buffer, len={}",
            idx.len_shared()
        );
        return;
    }

    if mode == "persist" {
        // Persistence validation: save → load → are the results identical?
        let t = Instant::now();
        let mut hnsw = HnswIndex::new(dim, metric, HnswParams::default());
        for (i, v) in base.iter().enumerate() {
            hnsw.insert(VectorId(i as u64), v).expect("insert");
        }
        println!("build: {:?}", t.elapsed());
        let path = std::path::Path::new("data/index.gvdb");
        let t = Instant::now();
        hnsw.save(path).expect("save");
        let size = std::fs::metadata(path).expect("stat").len();
        println!(
            "save: {:?}, file {:.1} MB ({:.0} B/vector)",
            t.elapsed(),
            size as f64 / (1024.0 * 1024.0),
            size as f64 / base.len() as f64
        );
        let t = Instant::now();
        let loaded = HnswIndex::load(path, false).expect("load");
        println!("load: {:?}", t.elapsed());
        let mut identical = true;
        for q in &queries {
            if hnsw.search(q, k) != loaded.search(q, k) {
                identical = false;
                break;
            }
        }
        println!(
            "results identical across {} queries after reloading: {identical}",
            queries.len()
        );
        assert!(identical);
        return;
    }

    let t = Instant::now();
    let truth = ground_truth(&base, &queries, k, metric);
    println!("ground truth generation (exact, rayon): {:?}", t.elapsed());

    let t = Instant::now();
    let mut index = BruteForceIndex::new(dim, metric);
    for (i, v) in base.iter().enumerate() {
        index.insert(VectorId(i as u64), v).expect("insert");
    }
    let build = t.elapsed();
    println!("build time: {build:?} ({} vectors)", index.len());

    let results: Vec<Vec<VectorId>> = queries
        .iter()
        .map(|q| index.search(q, k).iter().map(|r| r.id).collect())
        .collect();
    let recall = recall_at_k(&results, &truth, k);
    println!("recall@{k} = {recall:.4}");

    let stats = measure_latency(&queries, |q| {
        std::hint::black_box(index.search(q, k));
    });
    println!(
        "search latency: p50={:?} p99={:?} mean={:?} ({} samples)",
        stats.p50, stats.p99, stats.mean, stats.samples
    );

    let mem = index.memory_bytes();
    println!(
        "index memory: {:.1} MB total, {:.1} bytes/vector (raw data {} bytes/vector)",
        mem as f64 / (1024.0 * 1024.0),
        mem as f64 / index.len() as f64,
        dim * 4
    );
}
