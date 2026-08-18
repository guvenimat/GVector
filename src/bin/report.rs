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
use vector_gvector::index::VectorIndex;
use vector_gvector::types::VectorId;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("random");
    let k = 10;
    let metric = Metric::L2; // SIFT literatürde L2 ile değerlendirilir

    let (base, queries, label) = match mode {
        "sift" => {
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
