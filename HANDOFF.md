# Devir Notu — 2026-08-19

Bu dosya, oturum değiştiğinde kaybolmaması gereken **bağlamı** taşır.
Kararların gerekçeleri `DECISIONS.md`'de, ölçümler `BENCHMARKS.md`'de.
Yeni oturumun ilk işi: bu üç dosyayı okumak.

## Şu an neredeyiz

Aşama 0–8 tamamlandı (HNSW, kalıcılık, WAL, filtreler, planlayıcı, segment
tavanı, 1M gerçeklik ölçümü). Devam eden arc: **ölçümün yönlendirdiği işler**.

| iş | durum |
|---|---|
| 8a — int8 ölçeklenme ölçümü | **bitti** — int8 performans gerekçesiyle REDDEDİLDİ (#45) |
| 9a-1 — merge arka plana | **bitti** — pencere 80.5 s → ~29 s (#46–#48) |
| 9a-2 — mühürleme arka plana | tek worker + backpressure yapıldı (#53). Kriter 2 birincil **GEÇTİ**, kriter 1 **GEÇMEDİ** (#60) → yeni ön-kayıt #61 |
| 9c — metadata sıkıştırma | sırada (kapsam planda revize edildi) |
| 9d — türetilmiş indeks snapshot'ı | 9c'den SONRA (sıra gerekçesi planda) |
| 10 — README + v0.1.0 | en son |

Plan dosyası: `~/.claude/plans/bir-soru-fallback-e-i-ini-prancy-bachman.md`

## Açık ön-kayıtlar (DEĞİŞTİRİLEMEZ)

- **#40** — 9a latency eşiği: merge/mühürleme penceresine denk gelen
  yazmaların p99'u, taban p99'un (7.8 µs, **fsync'siz** ölçüldü) 50 katını
  aşmamalı. Ölçüm koşulu Aşama 8 ile aynı tutulmalı; fsync'li ölçüm eşiği
  otomatik aşar ve yanlış "kabul edilmedi" sonucu doğurur.
- **#49** — SONUÇ: karşılanmadı (metrik kusuru #58). Yerine **#59**
  (kuyruk birincil, 10 dk) → birincil GEÇTİ.
- **#61** — kriter 1'in ikinci sürümü: birincil = **backpressure dışındaki**
  en uzun yazma; ikincil = backpressure beklemelerinin dağılımı (bu turda
  yalnız ÖLÇÜLÜR, eşik yok); her iki fsync politikasında koşulur.
  Önce `with_capacity` işi yapılır.
- **9c eşiği (#40):** metadata payı > %25 → GO (ölçüldü: %51.5, f32 modunda,
  hesaplanan boyutlara göre).
- **9b:** NO-GO, `deny(unsafe_code)` bu arc'ta açılmaz (yeniden bakma koşulu
  #40'ta: veri RAM'e sığmazsa).

## Ön-kayıt kuralının kalıcı maddesi (#63)

Eşik yazarken şu soru DA sorulur: **"Bu kriter, başka hangi kriterle
çelişebilir?"** — 9a-2'de kriter 2'yi geçiren mekanizma (backpressure)
kriter 1'i geçilemez hale getirdi. Kriterler bağımsız yazılıyor ama
sistem bir bütün.

## Sözleşmeler (gerileme yasak)

Tek yazar (mpsc → tek yazıcı task) · okuyucular mühürleme/merge sırasında
durmaz · HTTP 200 = politikanın vaat ettiği dayanıklılık (#36, varsayılan
group:20) · seed=42 · 1M recall ef=100 ≥ 0.99 · kol örtüşmesi %100 ·
filtre boş/eşdeğer-boşsa `search_shared` kısayolu · `#![deny(unsafe_code)]` ·
clippy uyarısız + fmt + tüm testler yeşil (şu an **115 unit + 6 kaza**).

## Bu projede öğrenilmiş tuzaklar (tekrarlamamak için)

1. **Ölçüm izole süreçte yapılır.** Uzun bir koşunun sonuna eklenen
   throughput ölçümü ölçtüğünü sandığın şeyi ölçmez — Aşama 8'in "okuma
   ölçeklenmiyor" bulgusu böyle yanlış çıktı (#44). Protokol: taze süreç,
   warmup, 3 tekrar medyanı, iki ayrı koşuda doğrulama.
2. **Ölçüm çıktısını grep'leyerek okuma.** Panic gizlenir ve pipe `exit 0`
   döndürür. Filtresiz oku.
3. **Eşzamanlılık testleri yarışın tetiklendiğini ölçmeli.** `during_merge > 0`,
   `seal_in_flight() > 0`, `saw_sealing > 0` gibi — yoksa test sessizce
   zayıflar ve yeşil kalır.
4. **`data/fullscale` kalıcıdır**; ölçüm modları id çakışmasına karşı taban
   kaydırmalı (aksi halde ikinci koşu `DuplicateId` ile patlar).
5. **Eşik ile karar ayrı şeylerdir.** Eşik geçilse bile eşiğin dayandığı
   varsayım çürüdüyse karar farklı olabilir (#45 bunun örneği).
6. **Resmi SIFT ground truth yalnız tam 1M taban için geçerli**; alt kümede
   exact taramayla üretilmeli.
7. **bincode untagged serde'yi deserialize edemez** — disk/WAL temsili ayrı
   ve etiketli (`MetaValueRepr`, #35).

## SIRADAKİ İŞ (net)

1. **`with_capacity`** — yazma buffer'ı 125K kaydı baştan ayırsın; 64 MB'lık
   realloc sıçramasını kaldırır (ön-kayıt #61'in ön şartı).
2. **Kriter 1'i #61'e göre yeniden ölç:** backpressure dışındaki en uzun
   yazma + backpressure beklemelerinin dağılımı, HER İKİ fsync
   politikasında. Realloc hipotezi burada kanıtlanır ya da çürür.
3. Sonra 9c → 9d → 10 (plan dosyasındaki sıra).
4. **Ertelendi (#62):** segment birikmesi (mühürleme 25 s'de üretiyor,
   merge 54 s'de eksiltiyor). Yeniden bakma koşulu: sürekli yazma yükü
   öngörülen kullanımın parçası olursa.

## Yeniden koşulabilir ölçümler

```
cargo run --release --bin report -- fullscale 1000000 99   # 1M uçtan uca (~10 dk)
cargo run --release --bin report -- int8scale 1000000 99   # 8a
cargo run --release --bin report -- mergewindow 1000000 99 # 9a latency
cargo run --release --bin report -- accumulation 1000000 99 # 9a-2 birikme
cargo run --release --bin report -- coldprofile 1000 99    # soğuk başlangıç bileşenleri
```
