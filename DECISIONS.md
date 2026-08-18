# Mimari Kararlar

## Aşama 7a — Soğuk kalıcılık — 2026-08-18

### 32. Segment dosyaları değişmez, adları generation taşır
`segment-<gen>-<idx>.gvdb` bir kez yazılır, bir daha ASLA üzerine yazılmaz;
sonraki checkpoint'ler onu yalnız manifest'ten referanslar. Üç kazanç:
(1) her checkpoint sadece YENİ segmentleri yazar — 100K'da ilk checkpoint
221ms, ikinci (yeni segment yok) 98ms; 1M'de checkpoint maliyetini belirleyen
şey bu. (2) Windows dosya kilitleriyle uyumlu: açık handle'lı dosyaya asla
yazmıyoruz. (3) Aşama 9b'nin mmap'i için önkoşul — haritalanan dosyanın
değişmezliği zaten garanti.

### 33. Manifest tek gerçek kaynak, atomik takas, EN SON yazılır
Yazma sırası: yeni segmentler → metadata snapshot → **manifest** → GC.
Her an diskteki manifest, referansladığı tüm dosyalar var olacak şekilde
tutarlı; kesinti hangi adımda olursa olsun ESKİ manifest geçerli kalır ve
yeni dosyalar yetim kalır (sonraki GC toplar). GC manifest'ten sonra
çalışır — ters sıra hâlâ referanslanan bir dosyayı silebilirdi.

**Windows dizin fsync yok:** dosya içeriği `sync_all` ile fsync'li, rename
atomik (MoveFileEx REPLACE_EXISTING), ama dizin girdisinin dayanıklılığı
işletim sistemine bırakılmış (Rust std dizin handle'ı açmaz). Sonuç:
"checkpoint diskte ama dizin girdisi kaybolmuş" senaryosu teorik olarak
mümkün — kurtarma bu yüzden her zaman WAL replay'iyle tamamlanacak (7b).

### 34. Tombstone'lar manifest'te, türetilmiş yapılar diskte YOK
- Tombstone'lar segment dosyasına yazılamaz (değişmezlik kuralı) ve WAL'a
  bırakılamaz (checkpoint WAL'ı rotasyona sokar). Manifest zaten atomik ve
  küçük; merge/compaction tombstone'ları düzenli temizliyor.
- Eq posting-list'leri ve sayısal alan indeksleri **diske yazılmaz**:
  metadata'dan tam olarak türetilebiliyorlar. Tek kaynak → tutarsızlık
  riski yapısal olarak yok. Bedeli açılışta yeniden kurma (100K + 3 alan:
  toplam soğuk başlangıç 242ms; 1M'de Aşama 8 ölçecek).
- Metadata snapshot'ı tam yazım (artımlı değil): sıcak yolu WAL taşıyacak.

### 35. Disk temsili API temsilinden ayrı: `MetaValueRepr`
`MetaValue` HTTP JSON şekli için `#[serde(untagged)]` — `{"renk":"mavi"}`
gibi doğal gövdeler bunu gerektiriyor. Ama untagged deserialization
`deserialize_any` ister ve bincode self-describing olmadığı için bunu
DESTEKLEMEZ (sessizce değil, derleme/çalışma zamanı hatası). Disk ve WAL
temsili bu yüzden ayrı ve etiketli bir enum. Ayrışma zaten sağlıklı: biri
dış sözleşme, diğeri iç format; bağımsız evrilebilirler. Regresyon testi
her MetaValue türünü roundtrip ediyor.

## Range histogramı — 2026-08-18

### 31. Range tahmini: eşit-genişlik histogram [alt,üst] + sınırlı sayım
- **Eşit genişlik 64 kova**, quantile değil: basit, O(1) güncelleme.
  Çarpık dağılım riski ölçüldü (log-normal: üst/gerçek 49x'e kadar) ama
  quantile'a geçilmedi çünkü tahmin hatası kol seçimine sızmıyor (aşağıda).
  Quantile, ancak post-kolu ef'' ölçeklemesi çarpık veride ölçülebilir
  latency kaybettirirse gündeme gelir — o ölçümde recall 0.999+ ve p50'ler
  düzgün dağılımdan farksızdı.
- **Tahmin tek sayı değil [alt, üst] aralığı** (kullanıcı tasarımı): tam
  içerilen kovalar alt, sınır kovaları dahil üst. Kova-içi düzgünlük
  varsayımı hiç yapılmıyor; belirsizlik açık taşınıyor ve planlayıcı hep
  muhafazakâr tarafı kullanıyor (küçük kol için üst, ŝ için üst → hata
  yönü latency'e, asla recall'a çarpmaz).
- **Kritik ekleme — sınırlı sayım**: histogramın yanında değer-sıralı
  BTreeMap (bits-sıralı f64). Küçük-kol kararı tahminle DEĞİL,
  `enumerate_up_to(scan_limit)` ile kesin veriliyor; eşleşen id'ler tarama
  koluna bedava çıkıyor. Kabul kriterindeki kol örtüşmesi bu sayede
  yapısal %100 (ölçüm: 13/13, çarpık ve korelasyonlu hücreler dahil).
- **VE bağlacı**: bağımsızlık varsayımı YOK; üst sınırların minimumu
  (Fréchet). Korelasyonlu Eq∧Range hücresinde 2.25x şişkin ama muhafazakâr.
- **Genişleme payı %12.5**: monoton değer akışında histogram yeniden
  kurulumunu amorti eder (testli).
- Bakım: insert/remove O(log distinct); 100K inşada +%4 (BENCHMARKS).
- Bellek bedeli: sorted map id başına ~24B/sayısal alan — histogram-yalnız
  tasarıma göre pahalı ama küçük-kol kesinliği + fallback enumerasyonunu
  satın alıyor.


## Segment tavan bekçisi — 2026-08-18

### 30. Merge: minimal tavan bekçisi, en-küçük-iki, tavan 8
segcurve ölçümü (BENCHMARKS): eğri doğrusala yakın (~+45µs/segment) ve
eşit-recall karşılaştırmasında tam merge'in kazancı ~%20 — merge'in
gerekçesi latency DEĞİL, sınırsız büyümeyi kesmek (40 segment ≈ 1.8ms
olurdu; eğri doyuma ulaşmıyor). Politika:
- **En küçük iki** segment birleştirilir (en-eski değil): yeniden inşa
  maliyeti n'e bağlı — en ucuz merge, ve boyutlar dengelenir. HNSW merge'i
  gerçek birleştirme değil yeniden inşadır (grafları birleştirmenin ucuz
  yolu yok); yazma amplifikasyonu LSM'den pahalı → politika muhafazakâr.
- Mekanizma mühürlemenin "iki girdi, bir çıktı" varyantı: kilitsiz inşa,
  tek write-kilidi altında atomik takas (retain+push aynı kilitte — okuyucu
  ya eski ikiliyi ya birleşiği görür, asla ikisini birden/hiçbirini değil).
- Merge doğal compaction: tombstone'lular birleşiğe taşınmaz.
- Tek-yazar sözleşmesi korunur: merge yazıcıyı inşa süresince meşgul eder
  (100K/tavan-8'de toplam +3.9s), okuyucular hiç durmaz. Tepe bellek:
  kalıcı + iki kaynak segment (takas anına dek; 10K'lık segmentte +2×9MB).
- Tavan 8: segcurve'de 8 segment ≈ 385µs — kabul edilebilir taban; tavana
  ancak mühürleme sonrası bakılır, mühürleme başına merge asla.


## Filtre planlayıcısı — 2026-08-18

### 28. Ölçüm bulgusu: kırılganlık recall'da değil latency'de (#26 revizyonu)
Seçicilik süpürmesi (BENCHMARKS, filtre bölümü) hipotezin tersini gösterdi:
gezinti-içi filtre recall'u KORUYOR (en kötü hücre 0.952) çünkü kabul kümesi
dolana dek genişlemeye devam ediyor — bedeli, kümelenmiş eşleşme + uzak
sorguda gezintinin tüm grafa yayılması (kabul/ziyaret oranı 0.19'dan 0.01'e
çöküyor, p50 25µs→1.3ms). Eski ikili fallback (found < k) bu patolojiyi hiç
yakalamıyordu (süpürmede 0 kez tetiklendi). Ölçekli ef kolu test edildi ve
REDDEDİLDİ: recall zaten yüksek, sadece latency ekliyor.

### 29. Üç kollu planlayıcı: tarama / post-filter (over-fetch) / gezinti-içi
- **Kardinalite tahmini O(1)**: Eq için (alan, değer) → yaşayan id kümesi
  posting-list'leri (insert/delete'te bakım; tutarlılık testli). O(n)
  metadata sayımı planlayıcı için reddedildi: 100K'da 14.4ms — planlanan
  aramanın yüzlerce katı. Range koşulları tahmine katılmaz.
- **Kol 1 — tarama**: est ≤ max(16k, 0.05n) → graf hiç açılmaz, en küçük
  posting listesinde exact top-k. Maliyet est ile sınırlı, sorgu konumundan
  bağımsız (100K'da 12µs–1ms).
- **Kol 2 — post-filter (over-fetch)**: est daha büyükse FİLTRESİZ graf
  araması `ef'' = clamp(5k/ŝ, ef, 8ef)` ile, filtre sonuçlara uygulanır.
  Kritik içgörü (100K ölçümü): gezinti-içi filtre kümelenmiş eşleşme + uzak
  sorguda grafın tamamına yayılıyor (35ms) ve ölçekle sessiz recall düşüşü
  başlıyordu (0.948, fallback hiç tetiklenmeden — visited/admitted çöküşü
  tek sinyaldi). Filtresiz gezinti bu patolojiye YAPISAL olarak bağışık.
  ŝ Eq-minimum üst sınırı; sonuç < 2k kalırsa (pencere ıskaladı ya da tahmin
  şişkin — VE bağlacı korelasyonu) exact taramaya düşülür. β=5: β=3'te sonuç
  sayısı yetip kalite kaçıyordu (0.979 hücresi), 2k eşiği bunu yakalamıyordu.
- **Kol 3 — gezinti-içi**: yalnız Eq'suz (Range-only) filtrelerde, eski
  found<k güvenlik ağıyla. Tahminsiz durumda tek seçenek.
- Denenip REDDEDİLENLER: ölçekli ef (gezinti-içi ile — recall'u zaten
  koruyordu, sadece latency ekledi); ziyaret bütçesi + tarama fallback'i
  (10K'da çalıştı, 100K'da yanlış kesmeler 30K'lık taramalara mal oldu —
  bütçe API'ı ölçüm enstrümantasyonu olarak duruyor).
- Kalibre sonuç (10K): 21 hücrenin HEPSİNDE recall 1.000; en kötü hücre
  1.03ms (tarama tabanı), eski en kötü 1.3ms + 0.952 recall idi.


## Metadata filtreleme — 2026-08-18

### 26. Gezinti-içi filtre + brute-force fallback
Üç seçenek vardı: post-filter (ara, sonra ele — düşük seçicilikte k'dan az
sonuç), pre-filter (önce eşleşenleri bul, aralarında ara — graf bağlantılılığı
kopar), gezinti-içi (eşleşmeyen node köprü olarak gezilir, sonuca girmez —
tombstone mekanizmasının genellemesi). Üçüncüsü seçildi; `search_layer`'a
opsiyonel slot predicate'i eklendi. Doğruluk garantisi: graf araması k'dan az
sonuç bulursa filtreli doğrusal taramaya düşülür (aşırı seçici filtrede yavaş
ama eksiksiz — testte tek-eşleşmeli senaryo bunu doğrular).

### 27. Metadata id düzeyinde, segmentlerden ayrı
`SegmentedIndex.metadata: HashMap<VectorId, Metadata>` — segmentler immutable
kalır, metadata silme/yeniden eklemede id ile akar (silme metadata'yı düşürür,
eski metadata yeni kayda sızmaz). Filtre modeli bilinçli dar: Eq + Range'in
VE bağlacı; VEYA/negasyon ihtiyaç doğarsa ağaca genişletilir.


## SIMD — 2026-08-18

### 25. Açık SIMD: `wide::f32x8`, unsafe'siz
`std::simd` nightly istiyor, intrinsics unsafe istiyor; `wide` güvenli API
ile ikisinden de kaçınıyor (`deny(unsafe_code)` korundu). Float toplama
sırasının değişmesi bilinçli kabul: mesafeler yalnız karşılaştırılıyor,
~1 ulp fark sonucu etkilemez. `target-cpu=native` .cargo/config.toml'da —
binary taşınabilirliği yerine yerel performans (öğrenme projesi).


## Aşama 6 — 2026-08-18

### 22. Quantization mimarisi: f32 ile inşa, dondurup quantize et, ADC ile ara
Graf f32 hassasiyetle kurulur (komşu seçimi tam hassasiyetten yararlanır),
`QuantizedHnsw::from_hnsw` grafı kopyalayıp vektörleri u8 koda çevirir;
f32 kaynağı düşürülünce bellekte yalnız kodlar kalır. Arama asimetrik
(ADC): query f32, kodlar anlık dequantize — çift taraflı quantize hataya
iki kez maruz kalırdı. Donmuş indekste insert/delete `Unsupported`:
segment modelinde (Aşama 5) yazma zaten buffer'a gider, quantize indeks
"mühürlü segmentin sıkıştırılmış hali" rolündedir.

### 23. Rerank YOK (saf quantization)
Seçenekler: (a) saf SQ — düşük bellek, makul recall; (b) SQ + diskten f32
okuyup top-k'yı yeniden sırala — yüksek recall, IO bağımlılığı.
**(a) seçildi.** Ölçüm: SIFT 100K'da kayıp 0.005–0.011, hedef 0.02'nin
yarısı bile değil — rerank'in kazanacağı recall pratikte yok denecek kadar
az, buna karşılık disk IO yolu, dosya yaşam döngüsü bağımlılığı ve p99
belirsizliği eklenecekti. Rerank, recall bütçesi gerçekten sıkışırsa
(ör. PQ'ya geçiş) yeniden değerlendirilir.

### 24. Per-dimension min/max kalibrasyon
Global min/max yerine boyut başına: SIFT gibi boyutlar arası dinamik
aralığı farklı veride hata payını boyut başına küçültür. Sabit boyut
(max==min) scale=0 üretir; kod 0, decode min — NaN üretmez (testli).


## Aşama 5 — 2026-08-18

### 18. Eşzamanlılık: segment modeli (onaylanan strateji 2)
COW/RCU yerine immutable segment + append-only buffer seçildi (kullanıcı
onayıyla). Gerekçe: yazma başına O(n) kopya yok; compaction "mühürleme"ye
dönüşüp arka plana alınabiliyor; gerçek vektör DB mimarisi. Sharded locking
reddedildi çünkü kilit birimi (shard) ile erişim birimi (gezinti yolu)
örtüşmüyor: her adım rastgele shard'a sıçrar, insert çift yönlü bağlantı +
çok node'lu budama ile aynı anda çok kilit ister → deadlock ya da dev kilit.

### 19. Kilit disiplini: pahalı iş asla kilit altında değil
Okuyucu segment listesini `Vec<Arc<Segment>>` olarak klonlar (read kilidi
mikrosaniyeler) ve HNSW aramasını kilitsiz yapar. Mühürleme (HNSW inşası,
saniyeler) hiçbir kilit tutulmadan koşar; yayınlama sırası "önce segment
ekle, sonra buffer'ı boşalt" — aradaki pencerede görülebilecek kopyaları
arama id-bazlı tekilleştirme emer (veri kaybı yerine kopya tercih edildi).

### 20. Tombstone'lar segment-YEREL
Global silinmiş-küme, silinip yeniden eklenen id'de eski kopyayı hortlatırdı
(reinsert kümeden çıkarınca segmentteki eski vektör yeniden görünür olurdu).
Segment-yerel kümede eski kopya kendi segmentinde kalıcı gölgede kalır.

### 21. Tek-yazar sözleşmesi
Mutasyonlar `&self` üzerinden kilitli çalışır (`insert_shared`) ama duplicate
kontrolü check-then-act olduğu için çoklu yazıcıda yarışabilir; sözleşme
"çok okuyucu + tek yazıcı"dır (aşamanın kabul kriteriyle uyumlu). Çoklu
yazıcı ileride id-uzayı bölüştürme ya da yazı kuyruğuyla eklenebilir.


## Aşama 4 — 2026-08-18

### 15. Silme: tombstone + gezinmede köprü, sonuçta dışlama
Silinen node graftan sökülmez (kenar onarımı pahalı ve bağlantılılığı
riske atar); `deleted` bayrağıyla işaretlenir. `search_layer` tombstone'ları
GEZMEYE devam eder (komşuları keşfedilir, köprü görevi sürer) ama sonuç
kümesine almaz. İnşa sırasında yeni node'lar tombstone'lara bağlanabilir —
compaction toptan temizler.

### 16. Entry point silinirse: yaşayan en yüksek seviyeli node yeni giriş
Tombstone waypoint olarak işleyebilirdi ama tüm aramaların ölü node'dan
başlaması kırılganlık ekler; silme anında `pick_new_entry` çalışır.
Tüm elemanlar silinirse entry None olur; sonraki insert sıfırdan kurar.

### 17. Compaction: eşikli tam yeniden inşa
Tombstone oranı `tombstone_threshold`'u (varsayılan 0.3) aşınca delete
otomatik compaction tetikler: yaşayanlar taze indekse yeniden insert edilir.
Yerinde slot geri kazanımı yerine tam inşa: basit, doğruluğu garanti,
ve HNSW inşası zaten hızlı (10K → ~2 s). Bedeli: compaction anlık duraklama
yaratır — Aşama 5'in eşzamanlılık modeli bunu arka plana alabilir.


## Aşama 3 — 2026-08-18

### 11. Dosya formatı: magic + versiyon + bincode meta + ham f32 bölümü + CRC32
Vektör verisi meta'nın DIŞINDA, 4 byte'a hizalı ham bölümde tutulur —
memmap ile kopyasız erişilebilsin diye. CRC32 dosyanın tamamını kapsar;
checksum meta parse'ından ÖNCE doğrulanır (bozuk baytı deserializer'a
hiç göstermemek fuzz yüzeyini küçültür). Yazma geçici dosya + atomik
rename ile: yarım yazım asıl dosyayı asla bozamaz.

### 12. memmap2 lazy load BEKLEMEDE: unsafe izni gerekiyor
`memmap2::Mmap::map` bir `unsafe fn` (harita yaşarken dosya değişirse UB) ve
crate `#![deny(unsafe_code)]` ile derleniyor. Kural gereği kaldırmadan önce
kullanıcıya soruldu; onay gelene dek `load(path, lazy)` iki yolda da güvenli
tam-okuma yapar. `VectorStorage::Mmap` altyapısı hazır (bytemuck cast,
copy-on-write insert), izinle tek satırlık aktivasyon kaldı.

### 13. RNG durumu diske yazılmaz
Yükleme sonrası seviye RNG'si `seed ^ n`'den yeniden türetilir. Yüklenmiş
indekse yapılan insert'ler deterministiktir ama kesintisiz inşayla birebir
aynı graf olmayabilir — arama doğruluğu bundan etkilenmez, kabul edildi.

### 14. load_from_bytes ayrı yüzey
Fuzz hedefi, testler ve dosya yolu aynı parse kodunu paylaşır; `rebuild`
her slot/entry referansını sınır kontrolünden geçirir — bozuk ama crc'si
tutan (kasıtlı üretilmiş) dosyada bile panic yerine Err.


## Aşama 2 — 2026-08-18

### 7. HNSW komşu seçimi: Algorithm 4 heuristic + keepPrunedConnections
Naif top-M yerine makaledeki heuristic: aday, seçili bir komşuya query'den
daha yakınsa elenir. Bu, küme içi gereksiz kenarları kırpıp kümeler arası
köprüleri korur; recall'un veri kümelenmesine dayanıklılığı buradan gelir.
Elenenlerle M'e tamamlama (keepPrunedConnections) açık — düşük dereceli
node kalmasın diye.

### 8. Budama sonrası graf yönlüdür
`shrink_links` bir node'un listesini kırptığında karşı taraftaki kenar
silinmez (hnswlib ile aynı davranış). Çift yönlülüğü zorlamak her budamada
karşı listeleri de taramayı gerektirir ve pratik fayda sağlamaz; testler
sadece derece limitini ve komşu geçerliliğini doğrular.

### 9. Seviye ataması ve parametre varsayılanları
`level = floor(-ln(U) * mL)`, `mL = 1/ln(M)` (makale 4.1 optimumu),
`M_max0 = 2M`. Varsayılan M=16, ef_c=200. Süpürme sonucu tatlı nokta:
M=16 + ef_search 25–50 (BENCHMARKS.md Aşama 2 tablosu).

### 10. `search_layer`'da visited için `Vec<bool>`
HashSet yerine slot başına bayrak: 100K node'da sorgu başına tek 100KB
allocation, dallanma başına hash maliyeti yok. Eşzamanlılık aşamasında
sorgu-yerel kaldığı için paylaşım sorunu yaratmaz.


## Aşama 0 — 2026-08-18

### 1. `VectorIndex::insert` imzası: `&mut self`
**Karar:** Trait'te mutasyonlar `&mut self` alır; interior mutability yok.

**Gerekçe:** İndeks algoritmaları (özellikle HNSW insert) tek-yazar varsayımıyla
en basit ve en test edilebilir halinde yazılır. Aşama 5'teki eşzamanlılık,
trait'in içine `RwLock`/atomics gömerek değil, indeksin **üstüne** bir katman
sarılarak eklenecek (COW/arc-swap ya da immutable segment modeli — o aşamada
karşılaştırılıp seçilecek). `&self + interior mutability` seçseydik her
implementasyon lock granülaritesi düşünmek zorunda kalır, tek-thread'li
brute-force bile gereksiz senkronizasyon taşırdı. `&mut self` + dış katman,
"algoritma" ile "eşzamanlılık politikası"nı ayrıştırır; Aşama 5'te strateji
değiştirmek trait'i kırmaz.

### 2. Cosine normalizasyon politikası: insert/query anında normalize, aramada dot product
**Karar:** `Metric::Cosine` ile kurulan indeks, vektörü insert anında ve
query'yi arama başında bir kez normalize eder; sıcak mesafe döngüsü `-dot` çalıştırır.

**Gerekçe:** HNSW bir aramada binlerce mesafe hesabı yapar; her hesapta iki norm
çıkarmak (sqrt dahil) maliyeti ~3x'e katlar. Normalizasyon vektör başına bir kez
ödenir ve sonuç sıralaması birebir aynıdır. Bedeli: orijinal (normalize edilmemiş)
vektör indeksten geri okunamaz — bu bir arama motoru için kabul edilebilir;
gerekirse orijinaller Aşama 3'ün kalıcılık katmanında ayrıca saklanabilir.
Sıfır vektör edge case'i: normalize edilmeden bırakılır (NaN üretilmez),
her şeye benzerliği 0 sayılır.

### 3. Mesafe sözleşmesi: "küçük = daha yakın", L2 karesi, benzerlikler negatiflenir
**Gerekçe:** Tek yönlü sıralama sözleşmesi sayesinde top-k/heap/recall kodu
metrikten bağımsız tek kez yazılır. `sqrt` monoton olduğundan L2'de atlanır.

### 4. Graf temsili: index tabanlı (`Vec<Vec<usize>>`), Rc/RefCell yok
**Gerekçe:** (Aşama 2'de uygulanacak, şimdiden bağlayıcı.) Slot indeksli düz
vektörler hem borrow-checker sürtünmesini sıfırlar hem cache dostudur hem de
Aşama 3'te serileştirmesi trivialdir.

### 5. Ölçüm altyapısı indekslerden bağımsız
`eval::exact_top_k` trait'in dışında sade bir doğrusal taramadır ve proje boyunca
ground truth üreticisi olarak kalır. Aşama 1'in brute-force indeksi bile buna
karşı test edilir; referans ile test edilen şey aynı kod olursa test anlamsızlaşır.

### 6. Tekrarlanabilirlik: `StdRng::seed_from_u64(42)`
Tüm rastgelelik (veri üretimi, ileride HNSW seviye ataması) sabit varsayılan
seed'li `StdRng`'den gelir; benchmark'lar deterministiktir.
