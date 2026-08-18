//! Uçtan uca rapor: rastgele (veya ileride gerçek) veriyle recall@10 ve
//! latency (p50/p99) basar. Aşama 0'da indeks henüz yok; boru hattını
//! exact taramanın kendisiyle doğruluyoruz (recall tanım gereği 1.0 çıkmalı —
//! çıkmıyorsa ölçüm altyapısında bug var demektir).
//!
//! Kullanım: `cargo run --release --bin report [n] [dim] [n_query]`

use vector_gvector::dataset::{random_vectors, DEFAULT_SEED};
use vector_gvector::distance::Metric;
use vector_gvector::eval::{exact_top_k, ground_truth, measure_latency, recall_at_k};

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(10_000);
    let dim: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(128);
    let n_query: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(100);
    let k = 10;
    let metric = Metric::L2;

    println!("== rapor: n={n} dim={dim} queries={n_query} k={k} metric={metric:?} seed={DEFAULT_SEED} ==");

    let base = random_vectors(n, dim, DEFAULT_SEED);
    // query'ler taban kümeden ayrı seed'le üretilir ki arama trivyal olmasın
    let queries = random_vectors(n_query, dim, DEFAULT_SEED + 1);

    let t = std::time::Instant::now();
    let truth = ground_truth(&base, &queries, k, metric);
    println!("ground truth üretimi: {:?} (paralel)", t.elapsed());

    // Aşama 0 doğrulaması: exact taramanın recall'u kendi GT'sine karşı 1.0 olmalı.
    let results: Vec<_> = queries
        .iter()
        .map(|q| {
            exact_top_k(&base, q, k, metric)
                .iter()
                .map(|r| r.id)
                .collect::<Vec<_>>()
        })
        .collect();
    let recall = recall_at_k(&results, &truth, k);
    println!("recall@{k} = {recall:.4}");

    let stats = measure_latency(&queries, |q| {
        std::hint::black_box(exact_top_k(&base, q, k, metric));
    });
    println!(
        "latency (tek thread exact tarama): p50={:?} p99={:?} mean={:?} ({} örnek)",
        stats.p50, stats.p99, stats.mean, stats.samples
    );

    let bytes_per_vec = dim * std::mem::size_of::<f32>();
    println!(
        "ham vektör verisi: {} MB toplam, {} byte/vektör",
        n * bytes_per_vec / (1024 * 1024),
        bytes_per_vec
    );

    assert!(
        (recall - 1.0).abs() < f64::EPSILON,
        "ölçüm altyapısı hatalı!"
    );
    println!("OK: ölçüm boru hattı doğrulandı.");
}
