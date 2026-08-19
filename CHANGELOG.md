# CHANGELOG

## v0.1.0 — 2026-08-19

İlk sürüm. Sıfırdan yazılmış vektör arama motoru; hazır ANN kütüphanesi
kullanılmadı. Aşağıdaki liste aşamaları ve her aşamanın **başlık bulgusunu**
verir; gerekçeler `DECISIONS.md`'de, ölçümler `BENCHMARKS.md`'de.

### Aşamalar

| aşama | ne geldi | başlık bulgusu |
|---|---|---|
| 0 | İskelet + ölçüm altyapısı | Ölçüm altyapısı koddan önce kuruldu; her kabul kararı ondan çıktı. |
| 1 | Brute-force indeks (rayon) | Kalıcı doğruluk referansı: her ANN sonucu buna karşı doğrulanıyor. |
| 2 | HNSW (Malkov & Yashunin) | M=16, ef_c=200 ile recall 0.999 @ ef=50; tarama 9.6x hızlanma. |
| 3 | Kalıcılık (GVDB formatı) | magic + versiyon + CRC32, atomik yazım (tmp + fsync + rename). |
| 4 | Tombstone silme + compaction | %20 silmede recall bozulmuyor (0.9990 → 0.9990). |
| 5 | Segment modeli | Lucene/Qdrant modeli: değişmez segmentler + yazma buffer'ı; okuyucular hiç durmuyor. |
| 6 | Scalar quantization (f32→u8) | Bellek 2.35x düşüyor; ADC ile arama. |
| — | SIMD (`wide` f32x8) | Mesafede 9x mikro, uçtan uca 2.3x — `unsafe` açılmadan. |
| — | Metadata filtreleri | Gezinti-içi filtre + brute-force fallback (doğruluk garantisi). |
| — | HTTP API (axum) | insert / search / delete / stats / checkpoint. |
| 7 | Manifest + WAL + kaza kurtarma | Manifest EN SON yazılır, GC ondan sonra; replay ilk tutarsızlıkta durup dosyayı sağlam önekte keser. |
| — | Filtre planlayıcısı | **Küçük kol kararı asla tahminle verilmez.** Histogram çarpıkta 49x sapsa da kol seçimi doğru kalıyor; kol örtüşmesi %100. |
| — | Merge tavanı | Gerekçe latency değil (eşit-recall kazanç ~%20), sınırsız büyümeyi kesmek. |
| 8 | 1M uçtan uca gerçeklik ölçümü | Üç ön-kayıtlı kalemin ikisinde "problem başka yerdeymiş" sonucu. |
| 8-düzeltme | #44 | **"1M'de okuma ölçeklenmiyor" bulgusu yanlıştı** — kirli süreçte ölçülmüştü; f32 aslında 5.4–6.1x ölçekleniyor. |
| 8a | int8 ölçeklenme ölçümü | Eşik geçti ama **varsayım çürüdü**: int8 çok thread'de f32'den yavaş → reddedildi. |
| 9a-1 | Merge arka plana | Yazıcının bloke olduğu pencere 80.5 s → ~29 s. Tombstone diff-replay yarışı kilit disiplinine bağlandı. |
| 9a-2 | Mühürleme arka plana | Pencere **20.8 s → 0–2 µs**. Tek worker + kuyruk + backpressure (ilk tasarım 35 eşzamanlı thread doğuruyordu). |
| 9c | Metadata bellek sıkıştırması | Metadata 934 → 618 MB (−%34); harita 2.7x. Gerçek RSS 3088 → 2504 MB. |
| 10 | Kapanış | README, `NOT-DONE.md`, `METHODOLOGY.md`, v0.1.0. |

### Bu sürümde karşılanmamış eşikler

Ayrıntısı `NOT-DONE.md`'de. Özet: #40 kriter 1 (latency oranı), #49 ve #59
ikincil (segment birikmesi), #61 birincil group:20 politikasında. Her birinin
yanında kusur kaydı var; hiçbir eşik sonradan değiştirilmedi.

### Bilinen sınırlar

Segment birikmesi (sürekli yüksek yazma yükünde), `metadata_memory_bytes()`
sistematik olarak eksik gösteriyor, sıralı posting listeleri id sırasına
duyarlı, filtrede VEYA/negasyon yok, tek düğüm + tek yazar.

### Sözleşmeler

Tek yazar; okuyucular mühürleme/merge sırasında durmaz; HTTP 200 =
politikanın vaat ettiği dayanıklılık (varsayılan `group:20`); seed=42;
1M recall ef=100 ≥ 0.99; kol örtüşmesi %100; `#![deny(unsafe_code)]`.
