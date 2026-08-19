# Devir Notu — proje KAPANDI (v0.1.0, 2026-08-19)

Bu dosya, oturum değiştiğinde kaybolmaması gereken **bağlamı** taşır.
Kararların gerekçeleri `DECISIONS.md`'de, ölçümler `BENCHMARKS.md`'de,
yapılmayanlar `NOT-DONE.md`'de, ölçüm dersleri `METHODOLOGY.md`'de.

## Durum

Aşama 0–10 tamamlandı, **v0.1.0 etiketlendi**. Proje kapanış durumunda:
yeni özellik geliştirilmiyor. Devam edilecekse aşağıdaki backlog'dan
seçilir.

| aşama | durum |
|---|---|
| 0–7 | bitti (HNSW, kalıcılık, WAL, filtreler, planlayıcı, segment tavanı) |
| 8 + 8a | bitti — 8'in hatalı bulgusu #44'te düzeltildi; int8 reddedildi (#45) |
| 9a-1, 9a-2 | bitti — pencere 80.5 s → 0–2 µs; tek worker + backpressure (#53) |
| 9b | NO-GO — `deny(unsafe_code)` açılmadı |
| 9c | bitti — metadata 934 → 618 MB (#64, #65) |
| 9d | ertelendi |
| 10 | bitti — README, NOT-DONE, METHODOLOGY, CHANGELOG, v0.1.0 |

## Backlog (yeniden bakma koşullarıyla)

| kalem | yeniden bakma koşulu |
|---|---|
| **Segment birikmesi** (#62) — sistemin tek sınırsız büyüyen boyutu; mühürleme 25 s'de üretiyor, merge 54 s'de eksiltiyor | sürekli yazma yükü öngörülen kullanımın parçası olursa |
| **Sayısal alanlar için Eq posting'ini kaldırmak** (#68) — tekrarın kaldırılması; kalan metadata payının iki büyük kalemi aynı kökten besleniyor | metadata belleği yeniden sınırlayıcı olursa |
| **Sayısal indekslerin sıkıştırılması** (#67) — 197 MB | bellek sınırlayıcı olursa VE kesin-sayım kolunun regresyon testleri güçlendirilebilirse |
| **9d — türetilmiş indeks snapshot'ı** | soğuk başlangıç süresi kullanıcıya görünür olursa (çok kullanıcılı / sık yeniden başlatma) |
| **mmap / unsafe (9b)** (#40) | vektör verisi RAM'e sığmaz hale gelirse |
| **`metadata_memory_bytes()` düzeltmesi** (#66) | `/stats` rakamıyla kapasite planlaması yapılacaksa (şu an gerçeğin ~%77'sini gösteriyor) |
| **#61'in ikincil kalemine eşik** — backpressure bekleme dağılımı | veri `accumulation` modundan toplanır, `mergewindow` üretmiyor |

## Sözleşmeler (gerileme yasak)

Tek yazar (mpsc → tek yazıcı task) · okuyucular mühürleme/merge sırasında
durmaz · HTTP 200 = politikanın vaat ettiği dayanıklılık (#36, varsayılan
group:20) · seed=42 · 1M recall ef=100 ≥ 0.99 · kol örtüşmesi %100 ·
filtre boş/eşdeğer-boşsa `search_shared` kısayolu · `#![deny(unsafe_code)]` ·
clippy uyarısız + fmt + tüm testler yeşil (şu an **118 unit + 6 kaza**).

## Ön-kayıt kuralı

Eşik, onu değerlendirecek ölçüm koşulmadan ÖNCE yazılır ve sonradan
değiştirilmez. Karşılanmayan eşik "karşılanmadı" olarak durur; gerekiyorsa
yanına kusur kaydı yazılır. Eşik yazarken şu soru DA sorulur:
**"Bu kriter, başka hangi kriterle çelişebilir?"** (#63 — 9a-2'de kriter 2'yi
geçiren mekanizma kriter 1'i geçilemez hale getirdi.)

## Bu projede öğrenilmiş tuzaklar

Uzun hâli `METHODOLOGY.md`'de. Kısa liste:

1. **Ölçüm izole süreçte yapılır** — uzun koşunun sonuna eklenen ölçüm
   ölçtüğünü sandığın şeyi ölçmez (#44). Taze süreç, warmup, 3 tekrar
   medyanı, iki koşuda doğrulama.
2. **Ölçüm çıktısını grep'leyerek okuma** — panic gizlenir, pipe `exit 0`
   döner.
3. **Eşzamanlılık testleri yarışın tetiklendiğini ölçmeli**
   (`during_merge > 0`, `stalls > 0`, `max_queue > 0`).
4. **`data/fullscale` kalıcıdır** — ölçüm modları id çakışmasına karşı
   korunmalı; karşılaştırmadan önce dizin sıfırdan kurulmalı (koşular
   biriktikçe 1M → 1.64M oluyor).
5. **Eşik ile karar ayrı şeylerdir** — eşik geçse bile dayandığı varsayım
   çürüdüyse karar farklı olabilir (#45).
6. **Resmi SIFT ground truth yalnız tam 1M taban için geçerli.**
7. **bincode untagged serde'yi deserialize edemez** (#35).
8. **Rust kilit tuzakları** (#54): iki kilidi tek ifadede alma (geçiciler
   satır sonuna kadar yaşar); `while let` scrutinee'si GÖVDE boyunca yaşar
   (`let ... else` kullan). Kilit sırası: **segments → sealing → buffer**.
   Deadlock'un belirtisi asılmadır — CI'da timeout var (#55).

## Yeniden koşulabilir ölçümler

```
cargo run --release --bin report -- sweep 10000 128         # hızlı başlangıç
cargo run --release --bin report -- fullscale 1000000 99    # 1M uçtan uca (~10 dk)
cargo run --release --bin report -- memverify 1000000 99    # metadata belleği (gerçek RSS)
cargo run --release --bin report -- accumulation 1000000 99 # birikme + backpressure (10 dk)
cargo run --release --bin report -- mergewindow 1000000 99  # yazma latency penceresi
cargo run --release --bin report -- postingcost 200000 99   # posting id sırası duyarlılığı
cargo run --release --bin report -- rangefilter 100000 99   # kol örtüşmesi + Range tahmini
```
