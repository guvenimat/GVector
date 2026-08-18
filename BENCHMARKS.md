# Benchmark Kayıtları

## Filtre seçicilik süpürmesi + planlayıcı — 2026-08-18 (SIFT, k=10, ef=50)

Üç eşleşme dağılımı: uniform (id-uzayı), clustered (vektör-uzayı: merkezin
en yakın s·n komşusu; sorgular merkeze uzaklıkla yakın/orta/uzak), contig
(id-bitişik). Referans: brute-force filtreli tarama.

### Ölçüm bulguları (gezinti-içi filtre, ham HNSW, 100K)

- Sessiz recall düşüşü ölçekle GERÇEK: clustered×uzak×s=0.3 → **0.948**
  (10K'da 0.952 en kötüydü), fallback sayacı her hücrede 0 — tek sinyal
  kabul/ziyaret oranının 0.167'den 0.002'ye çöküşü.
- Asıl hasar latency: clustered×uzak hücrelerinde gezinti grafın tamamına
  yayılıyor — s=0.001'de p50 **35.3ms** (filtresiz arama 65µs).
- Ölçekli ef kolu (ef'=k/s) reddedildi: recall'u zaten koruyan mekanizmaya
  sadece latency ekliyor (39ms'e kadar).
- O(n) planlama sayımı reddedildi: 100K'da 14.4ms.
- İlk planlayıcı denemesi (ziyaret bütçesi 24·ef/√ŝ + tarama fallback)
  10K'da çalıştı, 100K'da yanlış kesmelerle geriledi → üretim yolundan
  çıkarıldı (enstrümantasyon olarak duruyor).

### Nihai planlayıcı (posting-list + tarama / filtresiz over-fetch)

10K: 21 hücrenin HEPSİNDE recall 1.000; en kötü p50 1.03ms (tarama tabanı).
100K örnek hücreler (önce = gezinti-içi ham, sonra = planlayıcı):

| hücre | önce recall/p50 | sonra recall/p50 |
|---|---|---|
| uniform s=0.001 | 1.000 / 20.2ms | 1.000 / 12µs |
| clustered×uzak s=0.001 | 1.000 / 35.3ms | 1.000 / 12µs |
| clustered×uzak s=0.01 | 1.000 / 25.8ms | 1.000 / 179µs |
| clustered×uzak s=0.05 | 1.000 / 20.3ms | 1.000 / 1.03ms |
| clustered×orta s=0.1 | 0.997 / 866µs | 0.997 / 1.6ms |
| clustered×uzak s=0.3 (kritik hücre) | **0.948** / 13.6ms | **1.000** / 10.8ms |
| clustered×uzak s=0.5 | 0.985 / 10.1ms | 1.000 / 20.4ms |
| contig s=0.5 | 0.998 / 127µs | 1.000 / 621µs |
| s=1.0 satırları | 0.989 / ~90µs | 1.000 / ~550–610µs |

Not: 100K'da en düşük hücre 0.988 (clustered×orta s=0.3) — eski 0.948
tabanının üstünde.

### scan_candidates optimizasyonu + ŝ≈1 kısayolu (aynı gün, ikinci tur)

İlk tarama kolu brute-force'un ~4 katıydı (id başına metadata yeniden
kontrolü + kaynak başına hash yoklama + tam sıralama). Düzeltmeler: tek-Eq
filtrede posting listesi kesin küme sayılır (yeniden kontrol yok), kaynak-dışı
döngü (bulunan id tekrar denenmez), top-k heap. Ayrıca tek Eq + est=n ise
filtre davranışsal boş → filtresiz `search_shared` kısayolu.

| hücre (100K) | opt. öncesi | sonrası |
|---|---|---|
| tarama bandı (s≤0.05) | 12µs–1.03ms | 7.6µs–440µs (~2.4x) |
| clustered×uzak s=0.3 | 10.8ms | **3.9ms** (gezinti-içi ham: 13.6ms/0.948) |
| clustered×uzak s=0.5 | 20.4ms | **6.25ms** (ham: 10.1ms/0.985 — artık hem hızlı hem recall 1.000) |
| s=1.0 satırları | 550–610µs | 430–530µs = segmentli filtresiz taban |

s=1.0 açıklaması: ilk rapordaki "65µs'e karşı 6x" karşılaştırması yanıltıcıydı —
65µs TEK HnswIndex'in filtresiz p50'siydi; segmentli indeksin filtresiz tabanı
zaten ~470µs (5 segment × ef araması; eşzamanlılık ölçümündeki 1893 QPS ≈ 528µs
ile tutarlı). Kısayol sonrası s=1.0 bu tabana oturuyor; gerileme yapısal değildi,
karşılaştırma hatasıydı. Recall tabanı değişmedi: 0.988.


## SIMD — 2026-08-18 (wide f32x8 + target-cpu=native)

Micro (128d):

| fonksiyon | önce | sonra | hızlanma |
|---|---|---|---|
| dot | 60.4 ns | 6.4 ns | 9.4x |
| l2_squared | 65.2 ns | 7.4 ns | 8.8x |
| ADC (quantize L2) | ~130 ns (skaler) | 15.6 ns | ~8x |

Uçtan uca (SIFT 100K, M=16/ef_c=200):

| ölçüm | önce | sonra |
|---|---|---|
| HNSW inşa | 40.4 s | 17.1 s |
| HNSW p50 (ef=50) | 124.3 µs | 52.7 µs |
| int8 p50 (ef=50) | 142.9 µs | 61.2 µs |
| brute-force p50 (rayon) | 547 µs | 160 µs |
| recall'lar | — | birebir aynı (determinizm korundu) |

Not: `target-cpu=native` tek başına kazandırmadı — `map().sum()` float
toplama sırası sabit olduğundan LLVM reduction'ı vektörleştiremiyor.
Kazanç, açık f32x8 + çift akümülatörden geldi (sıra değişimi ~1 ulp fark
yaratır, mesafe karşılaştırmasında önemsiz). Brute-force artık o kadar
hızlı ki 100K'da HNSW'nin görünür "hızlanma çarpanı" düştü — iki taraf da
aynı çekirdeği kullandığı için bu beklenen bir yeniden dengelenme.


## SIFT1M tam set — 2026-08-18 (stres testi, resmi ivecs ground truth)

M=16, ef_c=200. İnşa: **802 s (13.4 dk)**, segment başına süre sabit
(~80 s/100K) — saatler sürme endişesi doğrulanmadı.

| indeks | bellek | ef=50 | ef=100 | ef=200 |
|---|---|---|---|---|
| f32 | 496 MB vektör + 383 MB graf | 0.9680 / 296µs | 0.9900 / 480µs | 0.9960 / 782µs |
| int8 | 122 MB kod + 252 MB graf | 0.9630 / 325µs | 0.9830 / 479µs | 0.9890 / 778µs |

(hücreler: recall@10 / p50). Quantize dönüşümü 1.1 s. Kayıp 1M'de de ≤ 0.011.
Not: int8 graf belleğinin küçük görünmesi kopya sırasında Vec capacity'lerinin
tam boyuta oturmasından (f32 tarafı büyüme payı taşıyor).


## Aşama 6 — 2026-08-18 (scalar quantization f32→int8, ADC, M=16/ef_c=200)

Saf quantization (rerank yok); graf f32 ile inşa edilip donduruldu.
Kalibrasyon + kodlama: 10K → 7 ms, 100K → 92 ms.

### SIFT 10K

| indeks | ef | recall@10 | p50 | p99 | vektör MB | toplam MB (graf dahil) |
|--------|----|-----------|-----|-----|-----------|------------------------|
| f32 | 50 | 0.9990 | 65.3µs | 90.6µs | 5.0 | 8.9 |
| int8 | 50 | 0.9890 | 79.5µs | 124.4µs | 1.2 | 3.7 |
| f32 | 100 | 1.0000 | 107.1µs | 131.8µs | 5.0 | 8.9 |
| int8 | 100 | 0.9900 | 124.8µs | 192.8µs | 1.2 | 3.7 |

### SIFT 100K

| indeks | ef | recall@10 | p50 | p99 | vektör MB | toplam MB (graf dahil) |
|--------|----|-----------|-----|-----|-----------|------------------------|
| f32 | 25 | 0.9660 | 85µs | 274µs | 49.8 | 88.3 |
| int8 | 25 | 0.9610 | 91.2µs | 318.7µs | 12.2 | 37.6 |
| f32 | 50 | 0.9890 | 129µs | 400.7µs | 49.8 | 88.3 |
| int8 | 50 | 0.9800 | 142.9µs | 366.5µs | 12.2 | 37.6 |
| f32 | 100 | 0.9980 | 222.3µs | 477µs | 49.8 | 88.3 |
| int8 | 100 | 0.9870 | 225.9µs | 556.5µs | 12.2 | 37.6 |

Kabul kontrolü:
- Vektör verisi belleği: 49.8 → 12.2 MB = **4.1x düşüş** ✓ (hedef 4x)
- Toplam indeks (graf komşulukları dahil): 88.3 → 37.6 MB = 2.35x —
  graf başına ~404 B/vektör sabit kaldığı için toplam oran vektör oranından düşük.
- recall@10 kaybı: 0.005–0.011 arası, hepsi **< 0.02** ✓
- Latency ~%5-10 daha yüksek: ADC'de eleman başına ekstra mul+add
  (dequantize) var; bant genişliği kazancı 128d'de bunu henüz telafi etmiyor.


## Aşama 5 — 2026-08-18 (segment modeli eşzamanlılık, SIFT 100K)

Yapı: 5 × 20K HNSW segment + brute-force yazma buffer'ı. recall@10 = 1.0000
(ef=50'de segment-birleşimli arama; küçük segmentlerde recall tek büyük
indeksten daha yüksek çıkar, arama 5 kez ef genişliğinde çalıştığı için).

| Senaryo | Throughput |
|---|---|
| 1 okuyucu thread | 1893 QPS |
| 4 okuyucu | 8303 QPS (4.4x — ölçekleniyor, okuyucular birbirini bloklamıyor ✓) |
| 8 okuyucu | 16460 QPS (8.7x) |
| 4 okuyucu + aktif yazıcı (sürekli sil+ekle, mühürlemeler dahil) | 3018 QPS |

Notlar:
- Yazıcılı senaryodaki düşüş kilit beklemesinden çok CPU paylaşımından
  geliyor: yazıcı thread'i sıcak döngüde mühürleme inşaları yapıyor (her
  10K insert'te bir ~2 s'lik HNSW inşası) ve çekirdek çalıyor. Aramalar
  hiçbir noktada mühürleme süresince DURMUYOR — tek RwLock yaklaşımında
  bu 2 s'lik inşa tüm aramaları donduracaktı; ölçülebilir fark bu.
- Stres testi (4 okuyucu + 1 yazıcı, 3K insert + aralıklı silme, 5+ mühürleme):
  panic yok, sonuç invariant'ları (dup yok, NaN yok, sıralı) her sorguda doğrulandı.


## Aşama 4 — 2026-08-18 (silme + compaction, SIFT 10K, M=16, ef=50)

| Ölçüm | Değer |
|---|---|
| silme öncesi recall@10 | 0.9990 |
| %20 silme sonrası recall@10 | 0.9990 (bozulma yok ✓) |
| compaction süresi | 1.9 s (8K yaşayan eleman yeniden inşa) |
| bellek (vektör) | 5.0 → 4.0 MB (−%20 ✓) |
| bellek (graf) | 3.8 → 3.1 MB ✓ |
| compaction sonrası recall@10 | 1.0000 |

Entry point silme senaryosu ayrı testte: yeni entry yaşayanların en yüksek
seviyelisi seçiliyor, arama kesintisiz (`delete_entry_point_picks_new_entry_and_search_works`).


## Aşama 3 — 2026-08-18 (kalıcılık, SIFT 100K, M=16/ef_c=200)

| Ölçüm | Değer |
|---|---|
| save | 121 ms |
| load (tam okuma; mmap izni beklemede) | 73 ms |
| dosya boyutu | 71.8 MB (753 B/vektör: 512 B veri + graf + id'ler) |
| yeniden yükleme sonrası sonuçlar | 100/100 sorguda birebir aynı ✓ |
| kesik/bozuk dosya | panic yok, Err (test + proptest mini-fuzz) ✓ |


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
