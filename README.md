# gvector

Rust ile **sıfırdan yazılmış** bir vektör arama motoru: HNSW grafı, segment
modeli, WAL tabanlı kalıcılık, metadata filtreleri ve ölçüme dayalı bir
sorgu planlayıcısı.

## Ne olduğu / ne olmadığı

**Bu bir öğrenme projesi ve üretim öncesi bir prototiptir.** Hazır ANN
kütüphanesi kullanılmadı (hnsw_rs, faiss, instant-distance yok); algoritmalar
makaleden okunup elle yazıldı. Amaç çalışan bir sistem kurmak ve **her
kararı ölçümle gerekçelendirmekti**.

**Yok** (ve bu bilinçli): kimlik doğrulama, yetkilendirme, replikasyon,
sharding, çok kiracılı kullanım, TLS, hız sınırlama, dağıtık koordinasyon.
Tek süreç, tek düğüm, tek yazar. İnternete açık bir yere koymayın.

Kod `#![deny(unsafe_code)]` altında; SIMD dahil her şey güvenli Rust
(`wide` crate'i üzerinden).

## Başlık rakamları (SIFT-1M, 128 boyut, k=10)

Ölçüm makinesi: 8 çekirdek, Windows 11. Tüm rakamlar `BENCHMARKS.md`'de
koşulları ve tarihleriyle birlikte duruyor.

| ölçüm | değer |
|---|---|
| recall@10 (ef=100, resmi ground truth) | **0.9970** |
| arama p50 / p99 | 805 µs / 1.14 ms |
| yazma p50 / p99 (mühürleme dışı) | 600 ns / 7.6 µs |
| 1M inşa (metadata + WAL group:20) | 120 s |
| bellek (vektör + graf, f32) | 729 MB |
| bellek (metadata yapıları) | 618 MB |
| disk (checkpoint) | 802 MB (841 B/vektör) |
| soğuk başlangıç (1M, boş WAL) | 10.2 s |

Filtreli aramada **kol örtüşmesi %100** (planlayıcının seçtiği kol ile
oracle'ın seçtiği kol 16/16 hücrede aynı) ve filtre recall'ı 1.000
(bir hücrede 0.999).

## Mimari özet

**Segment modeli.** Yazmalar önce bellekteki brute-force buffer'a gider.
Buffer eşiğe varınca *mühürlenir*: içeriği bağımsız, değişmez bir HNSW
segmentine dönüşür. Arama tüm segmentleri + buffer'ı gezip sonuçları
birleştirir. Silme, segment-yerel tombstone ile yapılır.

**Tek yazar + arka planda inşa.** Yazma yolunda tek bir yazar var (sunucuda
mpsc → tek yazıcı task). Pahalı işler — HNSW inşası ve merge — arka plan
worker'larına alındı; yazıcının bloke olduğu pencere 80.5 s'den mikrosaniye
mertebesine indi. Kuyruk büyürse yazma **reddedilmez, yavaşlatılır**
(Lucene `IndexWriter` stall'ü gibi). Okuyucular mühürleme ve merge sırasında
hiç durmaz.

**Merge tavanı.** Segment sayısı tavanı aşarsa en küçük iki segment
birleştirilir. Gerekçe latency değil (eşit-recall karşılaştırmasında tam
merge ~%20 kazandırıyor), sınırsız büyümeyi kesmek.

**Filtre planlayıcısı.** Üç kol: küçük eşleşme kümesinde doğrudan tarama,
büyük kümede filtresiz gezinti + over-fetch, tahmin yoksa gezinti-içi
filtre. Kritik tasarım kararı: **küçük kol kararı asla tahminle verilmez** —
Eq'te sayım zaten kesin, Range'de sınırlı sayım (`enumerate_up_to`)
kesinleştirir. Histogram tahmini yalnız *büyük* kolun over-fetch penceresini
boyutlandırır, orada hata bedeli recall değil latency. Bu yüzden histogram
çarpık dağılımda 49x sapsa bile kol seçimi doğru kalıyor.

**Kalıcılık.** WAL çerçevesi `[len][crc32][payload]`; senkronizasyon
politikası `none` / `group(T)` / `per_op` (varsayılan `group:20`). HTTP 200
yalnız politikanın vaat ettiği dayanıklılığı ifade eder. Checkpoint'te
segmentler değişmez dosyalara yazılır, manifest EN SON yazılır ve GC ondan
sonra çalışır — böylece hiçbir anda manifest'in gösterdiği dosya eksik
olamaz. Replay ilk tutarsızlıkta durur ve dosyayı sağlam önekte keser.

## Kurulum ve kullanım

Gereken: Rust (stable). Veri kümesi için SIFT-1M'i `data/sift/` altına açın
(`sift_base.fvecs`, `sift_query.fvecs`, `sift_groundtruth.ivecs`).

```bash
cargo build --release
```

```bash
cargo test --release
```

### Ölçümleri koşturma

En hızlı başlangıç — parametre süpürmesi (SIFT gerektirmez, birkaç dakika):

```bash
cargo run --release --bin report -- sweep 10000 128
```

Diğer modlar (`report -- <mod> <n> <sorgu>`):

| mod | ne ölçer |
|---|---|
| `sweep` | HNSW parametre süpürmesi (M, ef_construction, ef_search) |
| `sift` | recall/latency temel ölçümü |
| `filter` | Eq filtre seçicilik süpürmesi + kol örtüşmesi |
| `rangefilter` | Range tahmini, kol örtüşmesi, bakım maliyeti |
| `fullscale` | 1M uçtan uca (inşa, bellek, recall, filtre, merge, soğuk başlangıç) |
| `memverify` | metadata bellek tahminini gerçek RSS ile doğrular |
| `accumulation` | sürekli yük altında kuyruk birikmesi + backpressure |
| `mergewindow` | mühürleme/merge penceresinin yazma latency'sine etkisi |
| `postingcost` | sıralı posting listesinin id sırasına duyarlılığı |
| `durability`, `wal`, `delete`, `quant`, `coldprofile` | ilgili aşamaların ölçümleri |

### HTTP sunucusu

```bash
cargo run --release --bin server
```

Uçlar: `POST /vectors`, `DELETE /vectors/:id`, `POST /search`,
`POST /checkpoint`, `GET /stats`.

## Belgeler

- **`DECISIONS.md`** — 69 numaralı karar, her biri gerekçesiyle. Reddedilen
  fikirler ve ertelemeler de burada.
- **`BENCHMARKS.md`** — her ölçüm, koşulları ve tarihiyle.
- **`NOT-DONE.md`** — yapılmayanlar, karşılanmamış eşikler ve bilinen
  sınırlar.
- **`METHODOLOGY.md`** — bu projeden çıkan ölçüm dersleri.
- **`HANDOFF.md`** — devir notu: durum, açık kalemler, tuzaklar.
