//! Uçtan uca rapor: recall@10, p50/p99 latency, bellek ve inşa süresi.
//!
//! Kullanım:
//!   cargo run --release --bin report -- random [n] [dim] [n_query]
//!   cargo run --release --bin report -- sift <n> [n_query]   (data/sift altından okur)
//!
//! Not: SIFT'in hazır ground truth'u 1M'lik TAM taban içindir; alt küme
//! kullanırken GT'yi exact taramayla kendimiz üretiriz (aksi yanlış recall verir).

use std::time::Instant;
use vector_gvector::dataset::{random_vectors, read_fvecs_subset, read_ivecs, DEFAULT_SEED};
use vector_gvector::distance::Metric;
use vector_gvector::eval::{exact_top_k, ground_truth, measure_latency, recall_at_k};
use vector_gvector::index::bruteforce::BruteForceIndex;
use vector_gvector::index::hnsw::{HnswIndex, HnswParams};
use vector_gvector::index::VectorIndex;
use vector_gvector::types::VectorId;

/// HNSW parametre süpürmesi: M × ef_search kombinasyonları için
/// recall/latency tablosu basar; brute-force referansıyla hızlanmayı raporlar.
fn hnsw_sweep(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize, metric: Metric) {
    let truth = ground_truth(base, queries, k, metric);
    let dim = base[0].len();

    // Brute-force referans latency'si (hızlanma oranı için).
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
        "| M | ef_c | ef_search | recall@{k} | p50 | p99 | hızlanma(p50) | inşa | graf B/vektör |"
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

/// Filtre seçicilik süpürmesi (plan: fallback ölçümü).
///
/// Üç eşleşme dağılımı:
/// - uniform: id-uzayında düzgün serpilmiş (taban çizgisi)
/// - clustered: VEKTÖR uzayında kümelenmiş — merkezin en yakın s·n komşusu.
///   Sorgular merkeze uzaklığa göre yakın/orta/uzak gruplanır: kırılganlık
///   sorgu eşleşme bölgesinden UZAKKEN bekleniyor.
/// - contig: id-bitişik ilk s·n kayıt (segment sınırı etkileşimi için ayrı)
fn filter_sweep(base: &[Vec<f32>], queries: &[Vec<f32>], k: usize, metric: Metric) {
    use std::collections::HashSet;
    use vector_gvector::meta::{Filter, MetaValue, Metadata, Predicate};

    let n = base.len();
    let dim = base[0].len();
    let ef = 50usize;
    let ef_cap = 4096usize; // ölçekli ef tavanı (kullanıcı geri bildirimi)

    let mut hnsw = HnswIndex::new(dim, metric, HnswParams::default());
    let mut bf = BruteForceIndex::new(dim, metric);
    for (i, v) in base.iter().enumerate() {
        hnsw.insert(VectorId(i as u64), v).expect("insert");
        bf.insert(VectorId(i as u64), v).expect("insert");
    }

    // Filtresiz referans (s=1.0 satırının sabit maliyet karşılaştırması).
    let unfiltered = measure_latency(queries, |q| {
        std::hint::black_box(hnsw.search_with_ef(q, k, ef));
    });
    println!(
        "filtresiz search p50 = {:?} (s=1.0 karşılaştırması)",
        unfiltered.p50
    );

    // Planlama maliyeti: O(n) metadata taraması bir sorgu planlayıcısına ne
    // kadar pahalı olurdu? Gerçekçi bir metadata haritası kurup tam sayım süresi.
    let meta_store: std::collections::HashMap<VectorId, Metadata> = (0..n)
        .map(|i| {
            (
                VectorId(i as u64),
                [("b".to_string(), MetaValue::Int(i as i64))].into(),
            )
        })
        .collect();
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
        "planlama maliyeti (O(n) tam sayım, n={n}): {:?} (sayım={cnt})",
        t.elapsed()
    );
    println!();
    println!("| varyant | s | grup | recall | p50 | fallback | ziyaret | kabul/ziyaret | ef' recall | ef' p50 | tarama p50 |");
    println!("|---------|---|------|--------|-----|----------|---------|----------------|------------|---------|------------|");

    // Kümelenmiş varyantın merkezi: taban kümeden bir vektör.
    let center = &base[0];
    // Sorguları merkeze uzaklığa göre sırala, üçe böl (yakın/orta/uzak).
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

            // Kümelenmişte sorgular uzaklık gruplarına ayrılır; diğerlerinde tek grup.
            let groups: Vec<(&str, Vec<usize>)> = if variant == "clustered" {
                vec![
                    ("yakın", q_order[..third].to_vec()),
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
                // Planlayıcının alternatif kolu: yalnız eşleşenlerde tarama.
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

    // ---- "Sonra" tablosu: planlayıcılı SegmentedIndex uçtan uca ----
    // Her s seviyesi bir Bool etiketi olur ("s0".."s6"); filtre Eq ile
    // posting-list yolunu kullanır. Varyant başına tek inşa.
    use vector_gvector::index::segmented::SegmentedIndex;
    println!();
    println!("== planlayıcılı SegmentedIndex (posting-list + tarama / filtresiz over-fetch) ==");
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
                    ("yakın", q_order[..third].to_vec()),
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("random");
    let k = 10;
    let metric = Metric::L2; // SIFT literatürde L2 ile değerlendirilir

    let (base, queries, label) = match mode {
        "sift" | "sweep" | "persist" | "delete" | "concurrent" | "quant" | "sift1m" | "filter"
        | "segcurve" | "mergecost" | "rangefilter" | "durability" | "wal" => {
            let n: usize = args.get(1).and_then(|a| a.parse().ok()).unwrap_or(10_000);
            let n_query: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(100);
            let mut f = std::io::BufReader::new(
                std::fs::File::open("data/sift/sift_base.fvecs").expect("data/sift yok mu?"),
            );
            let base = read_fvecs_subset(&mut f, n).expect("base okunamadı");
            let mut fq = std::io::BufReader::new(
                std::fs::File::open("data/sift/sift_query.fvecs").expect("query dosyası"),
            );
            let queries = read_fvecs_subset(&mut fq, n_query).expect("query okunamadı");
            (base, queries, format!("SIFT alt küme n={n}"))
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

    if mode == "wal" {
        // Aşama 7c ölçümü: fsync politikası × yazma throughput'u + replay.
        use vector_gvector::index::segmented::SegmentedIndex;
        use vector_gvector::meta::{MetaValue, Metadata};
        use vector_gvector::storage::wal::SyncPolicy;

        let n = base.len().min(20_000);
        // Sunucunun yazıcı task'i komutları batch'ler ve batch sonunda TEK
        // commit yapar; ölçüm bunu birebir modellemeli, yoksa group commit
        // pratikte per_op'a dönüşür ve tablo politikaların farkını göstermez.
        const BATCH: usize = 64;
        println!(
            "WAL ölçümü: {n} insert, batch={BATCH} (sunucu yazıcı task'i gibi), mühürleme kapalı"
        );
        println!();
        println!("| politika | süre | throughput | fsync/op | WAL boyutu | replay süresi | replay kayıt |");
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
                usize::MAX, // mühürleme yok: HNSW inşası ölçümü kirletmesin
                policy,
            )
            .expect("open");
            let t = Instant::now();
            for (i, v) in base.iter().take(n).enumerate() {
                let meta: Metadata = [("v".to_string(), MetaValue::Int(i as i64))].into();
                idx.insert_with_meta(VectorId(i as u64), v, meta)
                    .expect("insert");
                // Batch sonu commit: yanıtlar ancak bundan sonra gönderilir.
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

        // Büyük WAL replay'i: tam taban, fsync'siz yazım (replay süresi
        // politikadan bağımsız — okuma yolu aynı).
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
                "dolu WAL replay: {} kayıt / {wal_mb:.1} MB → {:?} ({:.0} kayıt/s)",
                re.replay_report().applied,
                t.elapsed(),
                base.len() as f64 / t.elapsed().as_secs_f64()
            );
        }
        return;
    }

    if mode == "durability" {
        // Aşama 7a ölçümü: checkpoint + soğuk başlangıç maliyeti.
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
        println!("inşa (3 metadata alanı ile): {:?}", t.elapsed());

        let t = Instant::now();
        let g1 = idx.checkpoint().expect("checkpoint");
        let ck1 = t.elapsed();
        let disk: u64 = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok()?.metadata().ok().map(|m| m.len()))
            .sum();
        println!(
            "ilk checkpoint (gen={g1}): {ck1:?}, disk {:.1} MB ({:.0} B/vektör)",
            disk as f64 / 1048576.0,
            disk as f64 / base.len() as f64
        );

        // İkinci checkpoint: hiç yeni segment yok → yalnız metadata+manifest
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
            "soğuk başlangıç: {cold:?} ({n_seg} segment, {} kayıt, türetilmiş indeksler yeniden kuruldu)",
            reopened.len_shared()
        );
        // Doğruluk: yeniden açılan indeks aynı sonuçları veriyor mu?
        let truth = ground_truth(&base, &queries, k, metric);
        let results: Vec<Vec<VectorId>> = queries
            .iter()
            .map(|q| reopened.search_shared(q, k).iter().map(|r| r.id).collect())
            .collect();
        println!(
            "yeniden açılış sonrası recall@{k} = {:.4}",
            recall_at_k(&results, &truth, k)
        );
        return;
    }

    if mode == "rangefilter" {
        // Range histogramı ölçümü (DECISIONS #31 kabul kriterleri):
        // düzgün + çarpık (log-normal) dağılım, tahmin aralığı vs gerçek,
        // korelasyonlu Eq+Range hücresi, kol-örtüşme oranı, bakım maliyeti.
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

        // Bakım maliyeti: metadata'sız vs iki sayısal alanlı inşa.
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
            "bakım maliyeti: metadata'sız inşa {t_plain:.1?}, 3 alanlı (2 sayısal) {t_meta:.1?} (+{:.0}%)",
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
            "| alan | s | gerçek | tahmin [alt,üst] | üst/gerçek | kol (oracle) | recall | p50 |"
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
            // (etiket, filtre, gerçek eşleşen id kümesi, tahmin sorgusu)
            type RangeCase<'a> = (&'a str, Filter, HashSet<u64>, (&'a str, f64, f64));
            let cases: Vec<RangeCase> = vec![
                (
                    "v(düzgün)",
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
                    "lv(çarpık)",
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

        // Korelasyonlu hücre: Eq(par=0) [i < n/2] ∧ Range v∈[0.4n, 0.6n) —
        // gerçek kesişim 0.1n; bağımsızlık/min-üst tahmini ~0.2n → oran ~2.
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
            "| Eq∧Range (korelasyonlu) | 0.1 | {} | range:[{el},{eu}] min-üst:{} | {:.2} | {arm} ({oracle}) | {:.3} | {:?} |",
            truth_corr.len(),
            eu.min(n / 2),
            eu.min(n / 2) as f64 / truth_corr.len().max(1) as f64,
            hits as f64 / total.max(1) as f64,
            stats.p50
        );
        println!();
        println!(
            "kol örtüşmesi: {agree}/{rows} ({:.0}%)",
            agree as f64 / rows as f64 * 100.0
        );
        return;
    }

    if mode == "mergecost" {
        // Tavan bekçisinin maliyeti: aynı veri, tavanlı vs tavansız.
        use vector_gvector::index::segmented::SegmentedIndex;
        let seal = base.len() / 10; // tavansızda 10 segment üretir
        for (label, ceiling) in [("tavansız", 100usize), ("tavan=8", 8)] {
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
                "{label}: inşa {build:.1?}, {n_seg} segment (+{n_buf} buffer), \
                 arama p50 {:?}, bellek {:.0} MB",
                stats.p50,
                idx.memory_bytes() as f64 / 1048576.0
            );
        }
        // Bellek tepe noktası (analitik): merge sırasında iki kaynak + birleşik
        // aynı anda yaşar. En kötü durum iki eşit segment: tepe ≈ kalıcı + 2×seg.
        let seg_bytes = (seal * (dim * 4 + 404)) as f64 / 1048576.0;
        println!(
            "merge tepe belleği ≈ kalıcı + 2×{seg_bytes:.0} MB (iki kaynak, \
             takas anına dek yaşar; birleşik zaten kalıcının parçası)"
        );
        return;
    }

    if mode == "segcurve" {
        // Segment sayısı × latency/recall eğrisi (merge politikası girdisi).
        // Aynı veri farklı seal eşikleriyle bölünür; arama filtresiz.
        use vector_gvector::index::segmented::SegmentedIndex;
        let truth = ground_truth(&base, &queries, k, metric);
        println!("| segment | p50 | p99 | recall@{k} | inşa |");
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
        // Aşama 4 doğrulaması: %20 silme sonrası recall + compaction bellek etkisi.
        let mut hnsw = HnswIndex::new(
            dim,
            metric,
            HnswParams {
                tombstone_threshold: 2.0, // manuel compaction için otomatiği kapat
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
            "silme öncesi recall@{k} (ef=50) = {:.4}",
            recall_of(&hnsw, &bf)
        );
        for i in (0..base.len()).step_by(5) {
            hnsw.delete(VectorId(i as u64)).expect("hnsw delete");
            bf.delete(VectorId(i as u64)).expect("bf delete");
        }
        println!(
            "%20 silme sonrası recall@{k} = {:.4} (tombstone oranı {:.2})",
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
        println!(
            "compaction sonrası recall@{k} = {:.4}",
            recall_of(&hnsw, &bf)
        );
        return;
    }

    if mode == "sift1m" {
        // Tam 1M stres testi: resmi ground truth (sift_groundtruth.ivecs)
        // burada GEÇERLİ — alt kümelerdeki gibi kendimiz üretmiyoruz.
        use vector_gvector::index::quant::QuantizedHnsw;
        let gt = read_ivecs(std::path::Path::new("data/sift/sift_groundtruth.ivecs"))
            .expect("ground truth okunamadı");
        let truth: Vec<Vec<VectorId>> = gt
            .iter()
            .take(queries.len())
            .map(|row| row.iter().take(k).map(|&i| VectorId(i as u64)).collect())
            .collect();
        assert_eq!(truth.len(), queries.len(), "GT/query sayısı uyuşmalı");

        let t = Instant::now();
        let mut hnsw = HnswIndex::new(dim, metric, HnswParams::default());
        for (i, v) in base.iter().enumerate() {
            hnsw.insert(VectorId(i as u64), v).expect("insert");
            if (i + 1) % 100_000 == 0 {
                println!("  insert {} / {} ({:?})", i + 1, base.len(), t.elapsed());
            }
        }
        println!("inşa: {:?} ({} vektör)", t.elapsed(), hnsw.len());
        let (vmem, lmem) = hnsw.memory_bytes();
        println!(
            "bellek: vektör {:.0} MB + graf {:.0} MB (graf {:.0} B/vektör)",
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
        drop(hnsw); // f32 kopyasını bırak: bellekte yalnız kodlar + graf kalır
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
        println!("| indeks | ef | recall@{k} | p50 | p99 | vektör MB | toplam MB |");
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
        println!("inşa: {:?} ({n_seg} segment + {n_buf} buffer)", t.elapsed());

        // recall kontrolü (segment birleştirme doğruluğu)
        let truth = ground_truth(&base, &queries, k, metric);
        let results: Vec<Vec<VectorId>> = queries
            .iter()
            .map(|q| idx.search_shared(q, k).iter().map(|r| r.id).collect())
            .collect();
        println!("recall@{k} = {:.4}", recall_at_k(&results, &truth, k));

        // throughput: her thread 3 sn boyunca arar, toplam sorgu sayılır
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
                        // yazıcı: sil + geri ekle döngüsü (net boyut sabit kalır)
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
                "okuma throughput ({threads} thread, yazıcı yok): {:.0} QPS",
                measure_qps(threads, false)
            );
        }
        println!(
            "okuma throughput (4 thread + aktif yazıcı): {:.0} QPS",
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
        // Kalıcılık doğrulaması: kaydet → yükle → sonuçlar birebir aynı mı?
        let t = Instant::now();
        let mut hnsw = HnswIndex::new(dim, metric, HnswParams::default());
        for (i, v) in base.iter().enumerate() {
            hnsw.insert(VectorId(i as u64), v).expect("insert");
        }
        println!("inşa: {:?}", t.elapsed());
        let path = std::path::Path::new("data/index.gvdb");
        let t = Instant::now();
        hnsw.save(path).expect("save");
        let size = std::fs::metadata(path).expect("stat").len();
        println!(
            "save: {:?}, dosya {:.1} MB ({:.0} B/vektör)",
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
            "yeniden yükleme sonrası {} sorguda sonuçlar birebir aynı: {identical}",
            queries.len()
        );
        assert!(identical);
        return;
    }

    let t = Instant::now();
    let truth = ground_truth(&base, &queries, k, metric);
    println!("ground truth üretimi (exact, rayon): {:?}", t.elapsed());

    let t = Instant::now();
    let mut index = BruteForceIndex::new(dim, metric);
    for (i, v) in base.iter().enumerate() {
        index.insert(VectorId(i as u64), v).expect("insert");
    }
    let build = t.elapsed();
    println!("inşa süresi: {build:?} ({} vektör)", index.len());

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
        "search latency: p50={:?} p99={:?} mean={:?} ({} örnek)",
        stats.p50, stats.p99, stats.mean, stats.samples
    );

    let mem = index.memory_bytes();
    println!(
        "indeks belleği: {:.1} MB toplam, {:.1} byte/vektör (ham veri {} byte/vektör)",
        mem as f64 / (1024.0 * 1024.0),
        mem as f64 / index.len() as f64,
        dim * 4
    );
}
