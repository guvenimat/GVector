# Benchmark Kayıtları

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
