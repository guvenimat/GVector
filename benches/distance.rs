//! Mesafe fonksiyonları micro-benchmark'ı (criterion).
//! Sabit seed'le üretilen 128-boyutlu vektörler — SIFT ile aynı boyut.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use vector_gvector::dataset::random_vectors;
use vector_gvector::distance::{dot, l2_squared, normalized, Metric};

fn bench_distances(c: &mut Criterion) {
    let vecs = random_vectors(2, 128, 42);
    let (a, b) = (&vecs[0], &vecs[1]);
    let (an, bn) = (normalized(a), normalized(b));

    c.bench_function("dot_128", |bch| {
        bch.iter(|| dot(black_box(a), black_box(b)))
    });
    c.bench_function("l2_squared_128", |bch| {
        bch.iter(|| l2_squared(black_box(a), black_box(b)))
    });
    c.bench_function("cosine_prenormalized_128", |bch| {
        bch.iter(|| Metric::Cosine.distance(black_box(&an), black_box(&bn)))
    });

    // ADC (quantize arama yolu) — f32 yollarla karşılaştırmak için
    use vector_gvector::index::quant::ScalarQuantizer;
    let base = random_vectors(100, 128, 43);
    let quant = ScalarQuantizer::fit(base.iter().map(|v| v.as_slice()), 128);
    let mut code = Vec::new();
    quant.encode(b, &mut code);
    c.bench_function("adc_l2_128", |bch| {
        bch.iter(|| quant.dist(Metric::L2, black_box(a), black_box(&code)))
    });
}

criterion_group!(benches, bench_distances);
criterion_main!(benches);
