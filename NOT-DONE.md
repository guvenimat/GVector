# Yapılmayanlar, karşılanmamış eşikler ve bilinen sınırlar

Bu dosya kasten "kötü haberler" dosyasıdır. Bir sistemin neyi yapmadığını
ve hangi eşiği tutturamadığını bilmek, neyi yaptığını bilmek kadar önemli.
Kayıtların uzun hâli `DECISIONS.md`'de.

## Karşılanmamış kabul eşikleri

Ön-kayıt kuralı gereği bu eşikler **ölçümden önce** yazıldı ve sonradan
değiştirilmedi. Karşılanmayan sonuç "karşılanmadı" olarak duruyor.

| eşik | sonuç | yanındaki kusur kaydı |
|---|---|---|
| **#40 kriter 1** — mühürleme/merge penceresine denk gelen yazmaların p99'u taban p99'un 50 katını aşmamalı | **karşılanmadı** (609x / 943x; `with_capacity` sonrası 426x) | #60: eşik, backpressure'ın VAR OLMADIĞI bir sistemde tanımlandı. O tarihte uzun yazma daima kusur belirtisiydi; backpressure eklendikten sonra uzun yazma sistemin DOĞRU çalıştığının kanıtı da olabiliyor. Metrik iki farklı nedeni aynı sayıda topluyor. |
| **#49** — 2 dk yükte segment+mühürlenen sayısı dengelensin (≤+%20, zirve ≤12) | **karşılanmadı** (+%90, zirve 35) | #58: metrik iki FARKLI REJİMDEKİ sayıyı topluyor — segment sayısı zaten merge tavanıyla bağlı, kuyruk ise o tarihte sınırsızdı. |
| **#59 ikincil** — segment sayısı merge tavanı+4 içinde kalsın | **karşılanmadı** (10 dk'da zirve 16) | #62: bu 9a-2'nin değil merge tavanı mekanizmasının testi; ayrı kalem. |
| **#61 birincil** — backpressure dışındaki en uzun yazma / taban p99 | group:20'de **426x**, sync kapalıyken **12x** | Kalan sıçramanın tamamı fsync (üç kanıtla izole edildi). Hangi politikanın kabulü belirlediği ön-kayıtta yazılmamıştı — boşluk kaydedildi. |

Geçen eşikler de var ve onlar `BENCHMARKS.md`'de: #59 birincil (kuyruk 10
dakika boyunca 3'te sabit), 1M recall ≥0.99, kol örtüşmesi %100, 9c'nin
bellek düşüşü.

## Denendi ve reddedildi

- **Ölçekli ef kolu** — planlayıcıya üçüncü bir kol olarak eklenmesi
  düşünüldü, ölçüldü, reddedildi: recall zaten bozulmuyordu, yani kolun
  çözeceği bir problem yoktu.
- **int8 quantization'ın segment modeline entegrasyonu (8a)** — ön-kayıtlı
  eşik geçti ama **eşiğin dayandığı varsayım çürüdü**: int8 çok thread'de
  f32'den yavaş çıktı. Eşik ile karar ayrı şeylerdir; eşik geçse bile
  varsayım çürüdüyse karar farklı olabilir.
- **Gezinti-içi filtrenin varsayılan yol olması** — 100K ölçümünde
  kümelenmiş eşleşme + uzak sorguda gezintinin grafın tamamına yayıldığı
  (35 ms'e kadar) ve ölçekle sessiz recall düşüşü başladığı görüldü.
  Yerine filtresiz gezinti + over-fetch geldi; o yol bu patolojiye
  yapısal olarak bağışık.

## Ölçüldü, kazanç yetersiz bulundu, ertelendi

- **Agresif/tiered merge** — eşit-recall karşılaştırmasında kazanç ~%20.
  Merge tavanı bu yüzden latency gerekçesiyle değil, sınırsız büyümeyi
  kesmek için var.
- **Quantile histogram** — mevcut eşit-genişlik histogramının hatası çarpık
  dağılımda 49x'e çıkıyor, ama **hiçbir kol kararını değiştirmiyor**
  (küçük kol kararı kesin sayımla veriliyor). Hata büyük, etkisi yok.
- **mmap / `unsafe` açılması (9b)** — soğuk başlangıçta mmap'in tavanı
  0.6 s, eşiğin altında. `#![deny(unsafe_code)]` açılmadı. **Yeniden bakma
  koşulu:** vektör verisi RAM'e sığmaz hale gelirse.
- **Türetilmiş indekslerin snapshot'lanması (9d)** — beklenen kazanç ~1.5 s
  soğuk başlangıç; tek kullanıcılı bir sistemde görünmez. Ertelendi.
- **Sayısal indekslerin sıkıştırılması (9c)** — 197 MB, metadata toplamının
  ~%16'sı. Gerekçe **bellek payı değil risk/kazanç**: o yapı filtre
  planlayıcısının kesin-sayım kolunu besliyor, yani projedeki en incelikli
  doğruluk mekanizmasının altında duruyor. Kapanışa giderken en riskli
  bileşene el atmak yanlış zamanlama.
- **Sayısal alanlar için Eq posting'inin kaldırılması** — 9c'den sonra
  yazılan sonraki adım. Sayısal bir alanın her ayrık değeri HEM bir posting
  anahtarı HEM bir `BTreeMap` girdisi üretiyor; Range zaten sayısal
  indeksten karşılanıyor. Bu bir optimizasyon değil, tekrarın kaldırılması
  olurdu. Ertelendi.

## Bilinen sınırlar (çalışıyor ama böyle çalışıyor)

- **Segment birikmesi — sistemin tek sınırsız büyüyen boyutu.** Mühürleme
  ~25 s'de bir segment üretiyor, merge ~54 s'de bir eksiltiyor. Sürekli
  yüksek yazma yükünde segment sayısı büyür (10 dakikada 0 → 16). Bu, 9a'nın
  kendi başarısının yan etkisi: mühürleme hızlandı, merge hızlanmadı.
  Çözüm ayrı bir tasarım işi (paralel merge, ya da merge'in üç segmenti
  birden alması). **Yeniden bakma koşulu:** sürekli yazma yükü öngörülen
  kullanımın parçası olursa. Hedeflenen senaryolarda (RAG, iç araç) sürekli
  5K op/s yazma yok — gerçek ama şu an teorik bir sorun.
- **`metadata_memory_bytes()` sistematik olarak eksik gösteriyor** (ölçülen
  0.77x, üç kalemde de aynı yönde). Aynı fonksiyon `/stats`'ta kullanılıyor,
  yani kapasite planlaması yapan biri gerçek kullanımın ~%77'sini görür.
- **Sıralı posting listeleri id sırasına duyarlı.** Artan sırada ekleme
  O(1), rastgele sırada O(n) kaydırma — 200K'lık tek listede 8.8x fark.
  Ölçüldü ve kabul edildi; 1M'lik tek bir liste ~30 s eder.
- **Filtrede VEYA / negasyon yok.** Yalnız koşulların VE bağlacı. İhtiyaç
  çıkarsa ağaç yapısına genişletilir.
- **Silme tombstone ile; alan geri kazanımı merge'e bağlı.** Çok silme
  yapılan bir kullanımda merge tetiklenene kadar disk ve bellek şişer.
- **Tek düğüm, tek yazar.** Yazma throughput'unun tavanı tek yazıcı
  task'idir; ölçeklenme yatay değil dikeydir.
