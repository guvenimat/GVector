# Benchmark Kayıtları

## Aşama 9c — metadata bellek sıkıştırması (SONUÇLAR) — 2026-08-19

### 9c-0: tahminin RSS ile doğrulanması

Yapılar tek tek bırakılıp RSS farkı ölçüldü (`report -- memverify`).
`clear()` değil, yapı yenisiyle DEĞİŞTİRİLDİ (clear kapasiteyi tutar).

**Ölçüldü: 1.50M kayıtlı dizinde** (metadata taşıyan kayıt sayısı 1M;
fazlası `mergewindow` koşularının metadata'sız kayıtları):

| adım | RSS | düşüş | tahmin | tahmin/gerçek |
|---|---|---|---|---|
| başlangıç | 3088 MB | — | — | — |
| −numeric | 2891 MB | 197 MB | 168 MB | 0.85x |
| −postings | 2321 MB | 570 MB | 371 MB | 0.65x |
| −metadata | 1822 MB | 499 MB | 441 MB | 0.89x |
| **toplam** | | **1266 MB** | 980 MB | **0.77x** |

**Gerçek metadata payı %41.0** (eşik %25 → GO ayakta).

Sapma yönü: tahmin gerçeği EKSİK gösteriyor. "İşletim sistemi belleği geri
vermedi" bunu açıklayamaz — o durumda gerçek düşüş daha KÜÇÜK görünürdü.
Yani %41 bir ALT sınır ve GO kararı şişirilmiş bir tahmine dayanmıyordu.

### 9c-1: uygulama sonrası

Karşılaştırma AYNI tahmin ediciyle, AYNI 1M metadata kümesinde
(`fullscale` bölüm 2 — kayıt sayısı normalizasyonu gerekmiyor, çünkü her
iki koşuda da metadata taşıyan kayıt sayısı tam 1M):

| kalem | önce | sonra | kazanç |
|---|---|---|---|
| harita (id→metadata) | 421 MB | **158 MB** | **2.7x** |
| posting listeleri | 353 MB | **299 MB** | 1.2x |
| sayısal indeksler | 160 MB | 160 MB | (dokunulmadı) |
| **metadata toplam** | **934 MB** | **618 MB** | **−%34** |
| metadata payı | %51.5 | **%45.9** | |

Gerçek RSS (memverify, ölçüldü 1.23M kayıtlı dizinde): **3088 → 2504 MB**,
gerçek metadata payı **%41.0 → %33.3**.

Posting kazancının küçük kalmasının sebebi: baskın terim `HashSet`'lerin
içi değil, **ayrı (alan, değer) anahtarı sayısı** — sayısal alanlar çok
sayıda ayrık değer üretiyor, dış `HashMap`'in kendisi hakim.

### Doğruluk ve maliyet (kabul kriterleri)

| kriter | önce | sonra |
|---|---|---|
| kol örtüşmesi (fullscale / rangefilter) | 3/3, 13/13 | **3/3, 13/13 (%100)** |
| filtre recall (13 hücre) | 1.000 (lv 0.1: 0.999) | **birebir aynı** |
| 1M recall ef=100 | 0.9970 | **0.9970** (≥0.99 ✓) |
| 1M inşa (metadata + WAL group:20) | 170.4 s | **120.4 s** |
| metadata bakım payı (rangefilter 100K) | +%4 (9.9→10.2 s) | +%17 (4.9→5.7 s) |

### Sıralı posting listesinin bilinen maliyeti

`report -- postingcost 200000`: 200K kayıt TEK posting listesine.

| id sırası | süre | kayıt/s |
|---|---|---|
| artan | 144.6 ms | 1.383.172 |
| **rastgele** | **1.266 s** | **157.945** |

Sıralı `Vec`'e ekleme O(n) kaydırma yapar. id'ler artan sırada gelirse
konum daima sondadır (O(1)); rastgele sırada **8.8x** yavaşlar. 200K'da
liste 1.6 MB, yani önbelleğe sığıyor ve kaydırma bellek bandı hızında —
bu yüzden O(n²) büyümesi bu boyutta ısırmıyor. 1M'lik TEK liste kabaca
30 s eder. Ölçümlerimiz id'leri hep artan sırada ürettiği için bu fark
daha önce görünmüyordu.


## Aşama 9a-2 — `with_capacity` sonrası kriter 1 (ön-kayıt #61) — 2026-08-19

Yalnız ölçüm sonuçları. `data/fullscale` sıfırdan kuruldu (1M, 8 segment),
aynı protokol her iki fsync politikasında koşuldu, 130.000 yazma.

| ölçüm | **group:20** (ön-kayıtlı koşul) | **sync kapalı** (fark izolasyonu) |
|---|---|---|
| taban p50 | 900 ns | 2.7 µs |
| taban p99 | 10.3 µs | 16.5 µs |
| **BİRİNCİL: backpressure dışı en uzun yazma** | **4.435 ms** | **200.3 µs** |
| BİRİNCİL oran (kendi p99'una) | **426x** | **12x** |
| oran (Aşama 8 tabanı 7.8 µs'e göre) | 569x | 26x |
| ön-kayıtlı 50x eşiği | **GEÇMEDİ** | **GEÇTİ** |
| mühürlemenin yazıcıyı bloke ettiği süre | 53 µs | 58 µs |
| İKİNCİL: bekletilen yazma (backpressure) | **0** | **0** |

### `with_capacity`ın etkisi (realloc hipotezi KANITLANDI)

En yavaş 5 yazma, önce/sonra:

| politika | `with_capacity` ÖNCESİ | SONRASI |
|---|---|---|
| group:20 | 4.0 – 10.0 ms | 4.03 – 4.44 ms |
| sync kapalı | 1.8 – 4.4 ms | **98 – 200 µs** |

Sync kapalıyken sıçramalar **~20x küçüldü** (ms mertebesinden yüz µs'ye).
Buffer'ın kademeli büyümesi (≈64 MB realloc + memcpy) fsync dışı
sıçramaların kaynağıydı; kapasite baştan ayrılınca kalmadı.

### Kalan sıçrama tamamen fsync

group:20'de kalan beş sıçrama **4.03, 4.04, 4.06, 4.10, 4.44 ms** — dar bir
kümede ve 130.000 yazmaya dağılmış (#14824, #46099, #61013, #107181,
#129620), mühürleme noktasıyla (#21266) ilgisiz. Sync kapatılınca tamamen
kayboluyor. Yani kalan pay dayanıklılık politikasının bedeli, yazma
yolunun kusuru değil.

### İkincil kalem: veri yok

Her iki koşuda da backpressure hiç devreye girmedi (0 bekletme): ölçüm
130.000 yazma sürüyor ve kuyruk eşiği aşılmıyor. #61'in ikincil kalemi
için bu turda veri üretilemedi; dağılım ancak sürekli yük altında
(`accumulation` modu) gözlenebiliyor — orada 10 dakikada 21 bekletme,
toplam 594 s ölçülmüştü.


## Aşama 9a-2 — ön-kayıtlı iki kriterin ölçümü (SONUÇLAR) — 2026-08-19

Bu bölüm YALNIZ ölçüm sonuçlarıdır. Sonuçlardan çıkarılan kararlar ve
eşik kusurlarının analizi DECISIONS'ta ayrı kayıtlardır (bilerek ayrı
commit'lerde: "karar sonucu mu şekillendirdi" sorusu bulanıklaşmasın).

### Kriter 2 — birikme, 10 dakika (ön-kayıt #59)

1M SIFT, 600 s kesintisiz tam hız yazma, seal=125K, tavan=8, WAL kapalı,
5 s'de bir örnekleme. Toplam 2.875M kayıt yazıldı (~4.8K op/s sürdürülen).

| t (s) | segment | mühürlenen (kuyruk) |
|---|---|---|
| 5 | 0 | 3 |
| 110 | 5 | 3 |
| 200 | 9 | 3 |
| 320 | 12 | 3 |
| 440 | 13 | 3 |
| 530 | 16 | 3 |
| 600 | 16 | 3 |

| #59 kalem | değer | eşik | sonuç |
|---|---|---|---|
| **BİRİNCİL** kuyruk: ilk 1/3 → son 1/3 | 3.0 → 3.0 (**−%2**) | dengelenir | **OK** |
| BİRİNCİL kuyruk zirvesi | **3** | sabit üst sınır | — |
| İKİNCİL segment zirvesi (merge tavanının testi) | **16** | ≤ 12 | **AŞILDI** |
| (referans) #49'un kusurlu metriği: segment+mühürlenen | 7.7 → 17.7 | ≤ +%20 | AŞILDI |

Backpressure: 21 insert bekletildi, toplam 594.4 s.

### Kriter 1 — latency (ön-kayıt #40)

Ölçüm koşulu Aşama 8 / 9a-1 ile aynı: `data/fullscale` SIFTEN YENİDEN
KURULDU (1M, 8 segment — önceki koşular dizini 1.64M/10 segmente
büyütmüştü), WAL group:20, döngü içinde commit yok, izole süreç, warmup
5.000 yazma, 130.000 yazma ölçüldü.

| ölçüm | koşu 1 | koşu 2 |
|---|---|---|
| taban p50 | 1 µs | 1 µs |
| taban p99 | 9.9 µs | 10.6 µs |
| **en uzun tek yazma** | **6.031 ms** | **9.996 ms** |
| **mühürlemenin yazıcıyı bloke ettiği süre** | **2 µs** | **0 ns** |
| merge sayısı (arka plan) | 0 | 0 |
| oran (en uzun / taban p99) | **609x** | **943x** |
| ön-kayıtlı eşik 50x | **GEÇMEDİ** | **GEÇMEDİ** |

**Yazıcıyı bloke eden pencerenin seyri (asıl bulgu):**

| | Aşama 8 | 9a-1 | 9a-2 |
|---|---|---|---|
| yazıcının bloke olduğu en uzun pencere | 80.5 s | 28.5–30.6 s | **0 ns – 2 µs** |
| bileşeni | mühürleme 20.8 s + merge 59.7 s | mühürleme 28.5 s | — |

### Teşhis: 6–10 ms'lik sıçramalar nereden geliyor?

En yavaş 5 yazmanın sıra numarası, mühürlemenin düştüğü sıra numarasıyla
karşılaştırıldı (mühürleme ~#11266):

```
#5444 → 9.9959ms   #71802 → 5.6031ms   #70092 → 4.155ms
#115892 → 4.1223ms #38739 → 4.0845ms
```

Sıçramalar 130.000 yazmaya DAĞILMIŞ ve mühürleme noktasıyla ilgisiz.

WAL sync kapalı teşhis koşusu (ön-kayıtlı koşul DEĞİL, `GVDB_DIAG_NOWAL=1`):
sıçramalar 6–10 ms'den 1.8–4.4 ms'ye indi ama KAYBOLMADI → fsync katkının
bir kısmı, tamamı değil. Kalan hipotez: yazma buffer'ının kapasite
büyütmesi (125K × 128 float ≈ 64 MB realloc). Hipotez henüz kanıtlanmadı.

**Teşhis koşusunun beklenmedik sonucu:** WAL kapalıyken yazıcı hızlanınca
kuyruk eşiği aşıldı ve **backpressure devreye girdi** — en uzun yazma
**24.9 s** oldu (#121265). Bu bir kusur değil, #53'ün tasarlanmış
davranışı: kriter 2'yi geçiren mekanizmanın kendisi.


## Aşama 9a-2 — tek worker + backpressure, kriter 2 ikinci ölçüm — 2026-08-19

1M SIFT, 120 s kesintisiz tam hız yazma, seal=125K, tavan=8, WAL kapalı.

**(a) İlk backpressure sinyali (mühürlenen+segment > 16) — YANLIŞ (#56):**

| t (s) | segment | mühürlenen | yazma op/s |
|---|---|---|---|
| 5 | 0 | 10 | 271.293 |
| 10 | 0 | 17 | 153.707 |
| 15–65 | 0→3 | 17→14 | **0** |
| 120 | 6 | 12 | **0** |

Yazıcı 110 saniye boyunca tamamen durdu; 2 insert 60 s güvenlik sınırına
dayandı. Toplam sayı mühürleme bitince düşmediği için geri besleme yok.

**(b) Düzeltilmiş sinyal (yalnız kuyruk, eşik 2):**

| t (s) | segment | mühürlenen | yazma op/s |
|---|---|---|---|
| 5 | 0 | 3 | 75.000 |
| 40 | 2 | 3 | 25.000 |
| 80 | 4 | 3 | 25.000 |
| 120 | 6 | 3 | 7.595 |

| kriter | değer | eşik | sonuç |
|---|---|---|---|
| ilk 1/3 → son 1/3 ort. (segment+mühürlenen) | 3.8 → 7.9 (**+%110**) | ≤ +%20 | **AŞILDI** |
| zirve (segment+mühürlenen) | 9 | ≤ 12 | OK |
| kuyruk uzunluğu | **3'te sabit** | (ön-kayıtta yok) | — |

**KRİTER 2 SONUCU: KARŞILANMADI** (eşik yeniden yorumlanmadı, #58).
Kuyruk düz; büyüyen kısım segment sayısı (0→6), yani merge tavanına
yaklaşma. 2 dakikalık pencere tavana ulaşmadan bitti → yeni ön-kayıt #59
ile 10 dakikaya çıkarıldı (karar sonuç görüldükten sonra alındı).

Karşılaştırma — önceki tasarım (sınırsız thread, #52): kuyruk 60 s'de
0→35, segment 0'da kaldı, yazma 273K→11.7K, bellek ~2.3 GB.
Yeni tasarımda kuyruk 3, segment düzenli üretiliyor, yazma mühürleme
hızında dengeleniyor (~5K op/s), bellek ~2 buffer.


## Aşama 9a-2 — birikme ölçümü (ön-kayıt #49, KRİTER 2) — 2026-08-19

60 s kesintisiz TAM HIZ yazma; seal=125K, tavan=8, WAL kapalı (ilk deneme
WAL açıkken 120 s'de **4.3 GB** log yazıp ölçümü boğdu).

| t (s) | segment | mühürlenen (kuyruk) | yazma op/s |
|---|---|---|---|
| 5 | 0 | 10 | 273.337 |
| 15 | 0 | 21 | 102.022 |
| 30 | 0 | 28 | 52.253 |
| 45 | 0 | 33 | 28.738 |
| 60 | 0 | **35** | **11.709** |

| kriter | değer | eşik | sonuç |
|---|---|---|---|
| ilk 1/3 → son 1/3 ortalaması | 17.8 → 33.8 (**+%90**) | ≤ +%20 | **AŞILDI** |
| zirve (segment + mühürlenen) | **35** | ≤ 12 | **AŞILDI** |

**KRİTER 2 SONUCU: BİRİKİYOR → backpressure 9a-2'nin parçasıdır (ön-kayıt #49).**

Üç ek bulgu:
1. **Segment sayısı 60 saniye boyunca 0'da kaldı** — yani *hiçbir* mühürleme
   tamamlanamadı. 35 mühürleme aynı anda çalışıp 8 çekirdeği paylaşıyor.
2. **Yazma hızı 273K → 11.7K op/s'ye çöktü** (23x). Sistem kendini zaten
   yavaşlatıyor, ama bunu *bellek baskısıyla* yapıyor — tasarlanmış bir
   backpressure değil, kontrolsüz bir çökme.
3. **Bellek:** 35 mühürlenen buffer × 125K kayıt ≈ 2.3 GB; ilk denemede
   süreç 7.9 GB RSS'e çıktı.


## Aşama 9a-1 — merge arka plana alındı — 2026-08-19 (SIFT 1M, tam sistem)

Ölçüm protokolü 8a'daki gibi: izole süreç, warmup (5.000 yazma), iki ayrı
koşuda tekrar. Ölçüm koşulu Aşama 8 ile aynı (WAL group:20, döngü içinde
commit yok) — ön-kayıtlı 50x eşiği fsync'siz taban üzerinden tanımlı.

| ölçüm | Aşama 8 (merge senkron) | 9a-1 (merge arka planda) |
|---|---|---|
| **yazıcının bloke olduğu en uzun pencere** | **80.5 s** | **28.5 s / 30.6 s** |
| — bileşenleri | mühürleme 20.8 s + merge 59.7 s | yalnız mühürleme 28.5 / 30.6 s |
| merge süresi | (yazıcıda) 59.7 s | (arka planda) 53.5 s / 54.0 s |
| merge sayısı | 1 | 2 / 4 |
| taban yazma p50 | 600 ns | 1.2 µs / 1.3 µs |
| taban yazma p99 | 7.8 µs | 10.1 µs / 8.4 µs |
| oran (max / taban p99) | 10.3 M x | 2.8 M x / 3.6 M x |
| ön-kayıtlı 50x eşiği | geçilmedi | **geçilmedi (beklenen)** |

"Ölçüm biterken merge çalışıyor muydu: **EVET**" — yani merge ile yazma
akışının örtüştüğü doğrulandı; merge artık gerçekten paralel koşuyor.

**Okumalar:**
- Yazıcının bloke olduğu süre **80.5 s → ~29 s (2.8x daralma)**; kalan
  sürenin tamamı **mühürleme**, yani 9a-2'nin hedefi ölçümle doğrulandı.
- Dürüst bir bedel: mühürlemenin KENDİSİ 20.8 s → 28–30 s'ye çıktı (%40).
  Sebep, merge'in artık paralel koşup CPU paylaşması. Net kazanç yine de
  büyük (80.5 → 29), ama "merge'i arka plana almak bedava" değil.
- 50x eşiği bu adımda geçilmedi ve bu **ön-kayıtta zaten öngörülmüştü**
  ("9a-1'de geçilmez, raporlanır"). Eşik 9a-2'den sonra tekrar sınanacak.
- Segment sayısı geçici olarak tavanı aşıyor (koşu başlangıçlarında 9 ve 10;
  mühürleme merge'den hızlı). Yazma durunca worker tavana indiriyor
  (her iki koşu 8 segmentle bitti).


## Aşama 8a — int8 çoklu-okuyucu ölçeklenmesi — 2026-08-19

Makine: **AMD Ryzen 7 7800X3D, 8 fiziksel / 16 mantıksal çekirdek, L3 = 96 MB**
(3D V-Cache). Veri: `data/fullscale` (1.13M kayıt, 8 segment). Ölçüm:
warmup + 3 tekrarın medyanı; **iki ayrı süreçte tekrarlandı**.

| indeks | ef | 1 thread | 2 | 4 | 8 | ölçeklenme (8/1) |
|---|---|---|---|---|---|---|
| f32 | 50 | 1286 / 1273 | 2443 / 1920 | 4649 / 3793 | **7865 / 6872** | **6.12x / 5.40x** |
| f32 | 100 | 794 / 738 | 1229 / 1180 | 2371 / 2334 | **4380 / 4355** | **5.52x / 5.90x** |
| int8 | 50 | 1226 / 1236 | 2243 / 2317 | 3242 / 4374 | **3643 / 3399** | **2.97x / 2.75x** |
| int8 | 100 | 741 / 803 | 964 / 1195 | 1694 / 1753 | **2649 / 2751** | **3.58x / 3.42x** |

(hücreler: koşu 1 / koşu 2 — tekrarlanabilirlik kontrolü)

| | değer |
|---|---|
| çalışma kümesi | f32 847 MB → int8 424 MB (**2.00x**, planda öngörülen ~2x) |
| quantize süresi | 0.39 s (8 segment) |
| recall kaybı (f32→int8) | **0.0091** (ef=50) / **0.0101** (ef=100) — eşik 0.02 ✓ |
| 8-thread QPS oranı int8/f32 | **0.46–0.49x** (ef=50), **0.60–0.63x** (ef=100) |

**Sonuçlar:**
1. **f32 1M'de ölçekleniyor: 5.4–6.1x** (8 fiziksel çekirdekte). Aşama 8'in
   "1M'de okuma ölçeklenmiyor (8 thread = tek thread QPS'i)" bulgusu
   **YANLIŞTI** — bkz. DECISIONS #44.
2. **int8 daha AZ ölçekleniyor (2.75–3.58x) ve mutlak olarak ~2x YAVAŞ.**
   Sebep ADC: her mesafede dequantize (min + scale·kod) aritmetiği; Aşama 6
   micro-bench'i zaten 15.6 ns (ADC) vs 7.4 ns (f32 L2) demişti. Tek thread'de
   bellek avantajı bu maliyeti dengeliyor (1226 vs 1286), 8 thread'de CPU
   darboğaz olunca ADC ağır basıyor.
3. **L3 hipotezi:** 100K'daki 8.7x ölçeklenmenin nedeni çalışma kümesinin
   (92 MB) 96 MB L3'e sığmasıydı. 1M'de f32 847 MB, int8 424 MB — **ikisi de
   L3'ün kat kat üstünde**, yani quantization çalışma kümesini cache'e
   sokmuyor, yalnız DRAM trafiğini yarıya indiriyor. Bu, int8'in ölçeklenmeyi
   "geri getirmesi" beklentisinin neden boş çıktığını açıklıyor.

**Metodoloji notu:** İlk sürüm tekrarlanabilir değildi (aynı kod, aynı veri:
5.08x ve 1.14x). Neden: aynı süreçte ikinci bir büyük indeks açmak (bellek
baskısı + cache kirliliği). Düzeltme: warmup + 3 tekrarın medyanı, tek indeks,
ayrı süreçlerde doğrulama.

**Recall mutlak değeri uyarısı:** tablodaki 0.8242, `data/fullscale` dizininin
fullscale koşusundan kalan 130K kopya kayıt içermesinden (aynı vektörler farklı
id'lerle; resmi GT saf 1M için). Anlamlı olan f32→int8 **kaybı** (0.009–0.010),
mutlak değer değil.

## Aşama 8 — 1M uçtan uca gerçeklik — 2026-08-19 (SIFT1M tam set, tam sistem)

Konfigürasyon: 8 segment (seal=125K, tavan=8), 3 metadata alanı + 3 kümelenmiş
filtre etiketi, WAL=group:20, f32. Eşikler ön-kayıtlı (DECISIONS #40).

| Ölçüm | Değer |
|---|---|
| inşa (1M + metadata, WAL açık) | **170.4 s** (tek-graf 1M: 802 s → 4.7x hızlı) |
| checkpoint | 2.44 s, disk **802 MB** (841 B/vektör) |
| soğuk başlangıç (medyan, 3 tur) | **3.63 s** (1.13M kayıt) |
| soğuk başlangıç + 10K WAL | 3.63 s (replay etkisi ölçülemez) |
| **recall@10 (resmi GT, ef=100)** | **0.9970** — eşik ≥0.99 **TUTTU** |
| arama p50 / p99 (tek thread) | 954.6 µs / 1.17 ms |
| bellek (hesaplanan) | vektör+graf **882 MB**, metadata **934 MB** |
| bellek (zirve RSS) | **3167 MB** |

### Soğuk başlangıç bileşenleri (9b kararı için)

| bileşen | süre | pay |
|---|---|---|
| (a) segment dosyaları okuma + CRC (812 MB) | 196 ms | %5 |
| (b) segment parse (graf + vektör kopyası) | 722 ms | %20 |
| (c) metadata okuma + decode (83 MB) | 428 ms | %12 |
| (d) **türetilmiş indeksleri kurma** (posting + sayısal) | **2.28 s** | **%63** |
| toplam | 3.62 s | |

### Filtre kritik hücreleri (clustered × uzak sorgu, gerçek planlayıcı yolu)

| s | eşleşme | kol (oracle) | recall | p50 |
|---|---------|--------------|--------|-----|
| 0.001 | 1.000 | scan (scan) | 1.000 | 131 µs |
| 0.05 | 50.000 | scan (scan) | 1.000 | 9.36 ms |
| 0.3 | 300.000 | post (post) | 0.997 | **92.65 ms** |

Kol örtüşmesi **3/3 (%100)** — 100K'daki sonuç 1M'de korundu. Recall korundu
ama s=0.3 hücresinde latency 100K'daki 3.9 ms'den 92.7 ms'ye çıktı (23x, veri
10x): tarama kolunun maliyeti eşleşme sayısıyla doğrusal büyüyor.

### Merge penceresi (9a gerekçesi)

| | değer |
|---|---|
| taban yazma p50 / p99 (pencere dışı) | 600 ns / **7.8 µs** |
| **en uzun tek yazma** | **80.5 s** |
| — bileşenleri | mühürleme **20.8 s** + merge **59.7 s** |
| oran (max / taban p99) | 10.3 milyon x |

### Karışık yük: 8 okuyucu + 1 yazıcı (200 op/s throttle)

| politika | okuma QPS | tabana oran | yazma op/s | okuma p50 |
|---|---|---|---|---|
| yazıcısız taban | 945 | 1.00 | — | 8.48 ms |
| `none` | 950 | **1.01** | 200 | 8.42 ms |
| `group:20` | 937 | **0.99** | 200 | 8.56 ms |
| `per_op` | 937 | **0.99** | 200 | 8.55 ms |

**Aşama 5 sözleşmesi sınavı geçildi:** fsync politikası okuyuculara ölçülebilir
bir yük bindirmiyor (oran 0.99–1.01). Bu ORANSAL sonuç geçerliliğini koruyor.

> ⚠️ **DÜZELTME (2026-08-19, DECISIONS #44):** Bu tablodaki **mutlak QPS
> değerleri geçersizdir** ve buradan çıkarılan "1M'de okuma hiç ölçeklenmiyor"
> sonucu **YANLIŞTI**. İzole ölçüm (Aşama 8a) f32'nin 1M'de **5.4–6.1x**
> ölçeklendiğini gösterdi. Hata kaynağı: bu tablo, 5 dakikadır çalışan,
> 1M inşa + 130K yazma + merge + üç soğuk başlangıç yapmış, RSS'i 3.1 GB'a
> çıkmış bir süreçte `fullscale`'in son bölümü olarak alınmıştı. Aynı
> şüpheyle bu koşunun diğer MUTLAK rakamları da (merge penceresi, soğuk
> başlangıç bileşenleri) izole ölçümle teyit edilmelidir; merge penceresi
> 9a-1'de teyit edildi (yukarıdaki bölüm).

### Kaza testi (1M snapshot + dolu WAL)

145K kayıtlık WAL, %67'de kesildi → 103.734 kayıt sağlam önek; açılış **3.69 s**,
kurtarılan durum sağlam önekle **birebir eşit**. Not: replay kayıt sayısı
mühürleme eşiğini (125K) aşarsa replay içinde HNSW inşası tetiklenir ve
kurtarma süresi doğrusallıktan çıkar — geçersiz ilk koşuda (rastgele veri,
252K kayıt) bu 206 s olarak gözlendi. Mekanizma veri-bağımsız; checkpoint
sıklığı kurtarma süresini doğrudan belirliyor.

## Aşama 7b/7c — WAL: fsync politikası ve kurtarma — 2026-08-18

20.000 insert (SIFT, 128d + 1 metadata alanı), batch=64 (sunucu yazıcı
task'inin davranışı), mühürleme kapalı — saf WAL yolu ölçülüyor.

| politika | süre | throughput | fsync/op | WAL | replay |
|---|---|---|---|---|---|
| `none` | 71 ms | **281.609 op/s** | 0.000 | 10.9 MB | 36.8 ms |
| `group:20` | 632 ms | **31.669 op/s** | 0.016 | 10.9 MB | 30.5 ms |
| `per_op` | 40.1 s | **499 op/s** | 1.000 | 10.9 MB | 33.2 ms |

Dolu WAL replay (100K kayıt / 54.3 MB): **155 ms — 646.000 kayıt/s**.

Okumalar:
- fsync ~2 ms (Windows, `sync_data`) → per_op'un 499 op/s tavanı doğrudan
  bundan geliyor. Group commit aynı dayanıklılık vaadini **63x** throughput
  ile veriyor: varsayılan `group:20` bu yüzden (DECISIONS #36).
- `none` ile `group` arasındaki 8.9x fark, fsync'in gerçek bedeli.
- Replay hızlı: 100K kayıt 155 ms — soğuk başlangıcın (242 ms, Aşama 7a)
  yanında checkpoint aralığını uzun tutmak ucuz. WAL boyutu replay süresini
  doğrusal etkiliyor.

**Kaza matrisi (deterministik kesme, 5 test):** kayıt sınırı / başlık ortası /
gövde ortası / son bayt eksik / checkpoint sonrası kesme + bozuk gövde →
her durumda kurtarılan durum WAL'ın sağlam önekine EŞİT, panic yok, ikinci
replay idempotent (dosya kesildiği için). proptest: rastgele op dizisi ×
rastgele kesme noktası (24 vaka) ve tamamen rastgele baytlar → panic yok.

## Aşama 7a — Soğuk kalıcılık — 2026-08-18 (SIFT 100K, 3 metadata alanı, 8 segment)

| Ölçüm | Değer |
|---|---|
| inşa (3 metadata alanı ile) | 8.1 s |
| ilk checkpoint (8 segment yazımı) | 221 ms |
| ikinci checkpoint (yeni segment yok) | 98 ms |
| disk toplam | 79.7 MB (836 B/vektör) |
| soğuk başlangıç (8 segment + türetilmiş indeksler) | 242 ms |
| yeniden açılış sonrası recall@10 | 1.0000 |

Okumalar:
- İkinci checkpoint'in 98 ms'i **tamamen metadata tam yazımı** (100K × 3 alan)
  + manifest; segment yazımı sıfır çünkü dosyalar değişmez (DECISIONS #32).
  1M'de bu kalem ~10x büyür — Aşama 8'in ölçeceği kalemlerden biri.
- Soğuk başlangıç 242 ms'in içinde 8 segment GVDB yüklemesi + posting-list ve
  sayısal indekslerin metadata'dan yeniden kurulması var (diske yazılmıyorlar).
- 836 B/vektör: 512 B ham vektör + ~404 B graf'ın bir kısmı + metadata
  snapshot'ı; graf/vektör oranı Aşama 2 tablosuyla tutarlı.

HTTP uçtan uca doğrulama (dim=4, kalıcı mod): 3 insert + 1 delete →
`POST /checkpoint` (gen=1) → süreç öldürüldü → yeniden başlatıldı →
`GET /stats` 2 kayıt/gen=1, arama silineni döndürmüyor, Eq ve Range
filtreleri çalışıyor (türetilmiş indeksler kurtarıldı), silinen id yeniden
eklenebiliyor.

## Range histogramı — 2026-08-18 (SIFT 100K, 64 kova eşit genişlik, k=10)

Bakım maliyeti: metadata'sız inşa 9.9s → 3 alanlı (2 sayısal) 10.2s (**+%4**).
scan_limit = 5000. Tahmin [alt, üst] aralığı; "üst/gerçek" hata göstergesi.

| alan | s | gerçek | tahmin [alt,üst] | üst/gerçek | kol (oracle) | recall | p50 |
|------|---|--------|------------------|-----------|--------------|--------|-----|
| v(düzgün) | 0.01 | 1000 | [0,1608] | 1.61 | scan (scan) | 1.000 | 112µs |
| v(düzgün) | 0.1 | 10000 | [8039,11255] | 1.13 | post (post) | 1.000 | 1.67ms |
| v(düzgün) | 0.3 | 30000 | [27333,30549] | 1.02 | post (post) | 1.000 | 727µs |
| v(düzgün) | 0.5 | 50000 | [48235,51451] | 1.03 | post (post) | 1.000 | 481µs |
| lv(çarpık) | 0.01 | 1000 | [0,48788] | 48.8 | scan (scan) | 1.000 | 117µs |
| lv(çarpık) | 0.1 | 10000 | [0,48788] | 4.88 | post (post) | 0.999 | 550µs |
| lv(çarpık) | 0.3 | 30000 | [0,48788] | 1.63 | post (post) | 1.000 | 566µs |
| lv(çarpık) | 0.5 | 50000 | [48788,79485] | 1.59 | post (post) | 1.000 | 376µs |
| Eq∧Range korelasyonlu | 0.1 | 10000 | min-üst: 22510 | 2.25 | post (post) | 1.000 | 1.22ms |

**Kabul kriterleri karşılığı:**
- Düzgün dağılımda post-bandı tahmin hatası %2–13 (< %20 ✓).
- Çarpıkta ölçüldü ve öngörülen patoloji doğrulandı: log-normal'de kütlenin
  tamamına yakını ilk kovalarda (48788 kayıt tek kova komşuluğunda) —
  üst/gerçek 1.6–49x. Quantile histogramına geçmenin gerekçesi bu satırlar;
  AMA aşağıdaki nedenle şimdilik gerekmedi:
- **Kol örtüşmesi 13/13 (%100)** (≥ %95 ✓): küçük-kol kararı histogramla
  değil değer-sıralı map'te sınırlı sayımla (`enumerate_up_to(scan_limit)`)
  verildiği için tahmin hatası kol seçimine hiç sızmıyor. Histogram yalnız
  post-kolunun ŝ'ına (ef'' ölçeği) etki ediyor; oradaki hata yönü muhafazakâr
  (üst sınır → küçük ŝ sanılmaz → recall değil en fazla latency öder) ve
  <2k fallback'i güvenlik ağı.
- Korelasyonlu Eq∧Range: min-üst 2.25x şişkin (bağımsızlık değil Fréchet
  üst sınırı kullanılıyor; şişkinlik yine muhafazakâr yönde), kol doğru,
  recall 1.000.
- Bakım maliyeti +%4 inşada, insert başına O(log distinct); DECISIONS #31.


## Segment sayısı × latency/recall eğrisi — 2026-08-18 (SIFT 100K, filtresiz, ef=50)

Merge politikası girdisi: aynı veri farklı seal eşikleriyle bölündü.

| segment | p50 | p99 | recall@10 | inşa |
|---------|-----|-----|-----------|------|
| 1 | 57.3µs | 99.2µs | 0.9889 | 16.4s |
| 2 | 110µs | 177µs | 0.9980 | 12.0s |
| 4 | 197.9µs | 321.9µs | 1.0000 | 10.1s |
| 5 | 272.2µs | 557.1µs | 1.0000 | 9.8s |
| 8 | 385.3µs | 596µs | 1.0000 | 8.9s |
| 10 | 466µs | 705.8µs | 1.0000 | 8.4s |

Tavan bekçisi maliyeti (aynı veri, seal=10K):

| | inşa | segment | arama p50 | bellek |
|---|---|---|---|---|
| tavansız | 8.4s | 10 | 446µs | 89 MB |
| tavan=8 | 12.3s | 8 (2 merge) | 410µs | 88 MB |

Merge tepe belleği ≈ kalıcı + 2×kaynak segment (takas anına dek; 10K
segmentte +2×9 MB).

Okumalar:
- Eğri hafif alt-doğrusal ama doğrusala yakın: segment başına ~+45µs
  (10 segment = tek segmentin 8.1 katı, 10 katı değil). Küçük segmentin
  kısalan gezintisi maliyeti tam telafi etmiyor çünkü her segment kendi
  ef genişliğinde aranıyor.
- Recall bonusu gerçek: 1 segment 0.9889, ≥4 segment 1.0000 (toplamda
  5×ef aday havuzu). Merge bu bonusu geri öder.
- **Adil karşılaştırma** (eşit ~0.998 recall'da): merge edilmiş tek indeks
  ef=100 → 222µs (Aşama 2 süpürmesinden); 5 segment ef=50 → 272µs.
  Tam merge'in net kazancı ~%20, naif "5x" değil. Merge acil DEĞİL.
- İnşa süresi segment sayısıyla DÜŞÜYOR (16.4s → 8.4s): küçük graf ucuz
  kurulum. Merge = yeniden inşa olduğundan (grafları birleştirmenin ucuz
  yolu yok) yazma amplifikasyonu LSM'den pahalı → politika muhafazakâr
  olmalı: segment sayısı tavanı (ör. 8–10) aşılınca en eski/küçük çifti
  birleştir; her mühürlemede değil.


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
