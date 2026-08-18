//! Uçtan uca rapor: recall@10, p50/p99 latency, bellek ve inşa süresi.
//!
//! Kullanım:
//!   cargo run --release --bin report -- random [n] [dim] [n_query]
//!   cargo run --release --bin report -- sift <n> [n_query]   (data/sift altından okur)
//!
//! Not: SIFT'in hazır ground truth'u 1M'lik TAM taban içindir; alt küme
//! kullanırken GT'yi exact taramayla kendimiz üretiriz (aksi yanlış recall verir).

use std::time::Instant;
use vector_gvector::dataset::{random_vectors, read_fvecs_subset, DEFAULT_SEED};
use vector_gvector::distance::Metric;
use vector_gvector::eval::{ground_truth, measure_latency, recall_at_k};
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("random");
    let k = 10;
    let metric = Metric::L2; // SIFT literatürde L2 ile değerlendirilir

    let (base, queries, label) = match mode {
        "sift" | "sweep" | "persist" | "delete" | "concurrent" | "quant" => {
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
