# Benchmark Kayıtları

## Aşama 2 — 2026-08-18 (HNSW, SIFT1M alt kümeleri, k=10, L2)

Referans brute-force (rayon, tüm çekirdekler): 10K p50=617µs; 100K p50=547µs.
HNSW araması tek thread. "hızlanma" = bf_p50 / hnsw_p50.

### SIFT 10K (kabul: recall ≥ 0.95 ✓)

| M | ef_c | ef_search | recall@10 | p50 | p99 | hızlanma | inşa | graf B/vektör |
|---|------|-----------|-----------|-----|-----|----------|------|----------------|
| 8 | 100 | 10 | 0.8830 | 15.8µs | 22µs | 39.1x | 1.0s | 233 |
| 8 | 100 | 25 | 0.9700 | 26.7µs | 40.4µs | 23.1x | 1.0s | 233 |
| 8 | 100 | 50 | 0.9890 | 44.3µs | 57µs | 13.9x | 1.0s | 233 |
| 16 | 200 | 10 | 0.9500 | 22.1µs | 32.3µs | 27.9x | 2.3s | 403 |
| 16 | 200 | 25 | 0.9940 | 40.3µs | 53µs | 15.3x | 2.3s | 403 |
| 16 | 200 | 50 | 0.9990 | 64.2µs | 82.3µs | 9.6x | 2.3s | 403 |
| 32 | 400 | 10 | 0.9860 | 32.3µs | 46.9µs | 19.1x | 5.6s | 735 |
| 32 | 400 | 25 | 0.9990 | 55.3µs | 78.9µs | 11.2x | 5.6s | 735 |

### SIFT 100K (kabul: ≥10x hızlı ✓, inşa dakikalar içinde ✓)

| M | ef_c | ef_search | recall@10 | p50 | p99 | hızlanma | inşa | graf B/vektör |
|---|------|-----------|-----------|-----|-----|----------|------|----------------|
| 8 | 100 | 25 | 0.9100 | 47.3µs | 64.9µs | 11.6x | 14.4s | 233 |
| 8 | 100 | 50 | 0.9740 | 73.9µs | 98.3µs | 7.4x | 14.4s | 233 |
| 16 | 200 | 10 | 0.8770 | 43.1µs | 69.5µs | 12.7x | 40.4s | 404 |
| 16 | 200 | 25 | 0.9660 | 81.7µs | 118.2µs | 6.7x | 40.4s | 404 |
| 16 | 200 | 50 | 0.9890 | 124.3µs | 168.3µs | 4.4x | 40.4s | 404 |
| 16 | 200 | 100 | 0.9980 | 218.2µs | 297.4µs | 2.5x | 40.4s | 404 |
| 32 | 400 | 25 | 0.9860 | 110.7µs | 145.7µs | 4.9x | 106.8s | 740 |
| 32 | 400 | 50 | 0.9960 | 188µs | 248.6µs | 2.9x | 106.8s | 740 |

Notlar:
- Graf bellek maliyeti (M=16): ~404 B/vektör — ham veri 512 B/vektör'ün üstüne
  ~%79 ek yük. M=8 seçilirse 233 B'ye düşer, recall için ef_search artırmak gerekir.
- Referans brute-force ÇOK çekirdekli; tek thread'e karşı hızlanma çok daha
  yüksek olurdu (~5.5ms/547µs ölçeğinde). 10x kabulü muhafazakar yorumla geçildi.
- Tatlı nokta: M=16, ef_c=200, ef_search 25–50 (recall 0.966–0.989, 4–7x).


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
