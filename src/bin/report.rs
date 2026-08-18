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
        "sift" | "sweep" | "persist" => {
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
