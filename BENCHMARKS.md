# Benchmark Kayıtları

## Aşama 1 — 2026-08-18 (brute-force indeks, SIFT1M alt kümeleri)

Veri: SIFT base'in ilk n vektörü, 100 gerçek SIFT query, k=10, L2.
Ground truth alt küme üzerinde exact taramayla üretildi (hazır GT 1M taban içindir).

| Ölçüm | SIFT 10K | SIFT 100K |
|---|---|---|
| recall@10 | **1.0000** | **1.0000** |
| search p50 | 611.7 µs (seri yol) | 672.7 µs (rayon paralel) |
| search p99 | 776.7 µs | 1.09 ms |
| inşa süresi | 2.2 ms | 21.4 ms |
| indeks belleği | 8.5 MB (886 B/vektör) | 67.6 MB (709 B/vektör) |
| ham veri | 512 B/vektör | 512 B/vektör |

Notlar:
- Vektör başına ek yük (886/709 vs 512 B) `Vec` capacity büyüme payı + id
  haritasından geliyor; brute-force için kabul edilebilir, HNSW'de ayrıca raporlanacak.
- 10x veri ≈ aynı p50: 20K eşiği üstünde tarama rayon'a dağılıyor (paralel
  parça başına yerel top-k heap + kilitsiz birleştirme).


Ortam: Windows 11, rustc 1.97.1, release profili. Seed = 42, tekrarlanabilir.

## Aşama 0 — 2026-08-18 (ölçüm altyapısı doğrulaması, rastgele veri)

Veri: 10.000 × 128d rastgele vektör (uniform [-1,1)), 100 query, k=10, metrik L2.
Henüz indeks yok; ölçülen, referans exact tarama (`eval::exact_top_k`).

| Ölçüm | Değer |
|---|---|
| recall@10 (exact vs exact GT) | 1.0000 (boru hattı doğrulaması) |
| latency p50 (tek thread exact) | 634.6 µs |
| latency p99 | 666.7 µs |
| ground truth üretimi (100 query, rayon) | 6.4 ms |
| bellek (ham f32 vektör) | 512 byte/vektör (128 × 4B), toplam 4 MB |
| inşa süresi | — (indeks yok) |

Criterion micro-bench (128d, tek çift vektör):

| Fonksiyon | Süre |
|---|---|
| dot | ~58.4 ns |
| l2_squared | ~61.3 ns |
| cosine (ön-normalize edilmiş, -dot) | ~57.8 ns |

Not: cosine'ın dot ile aynı maliyette olması normalizasyon politikamızın
(insert anında normalize) beklenen sonucudur; aramada norm hesaplanmıyor.
