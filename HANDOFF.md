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
| 9a-2 — mühürleme arka plana | kod+testler bitti ama **KABUL EDİLMEDİ**: kriter 2 aşıldı → önce tek-worker + backpressure (#51, #52) |
| 9c — metadata sıkıştırma | sırada (kapsam planda revize edildi) |
| 9d — türetilmiş indeks snapshot'ı | 9c'den SONRA (sıra gerekçesi planda) |
| 10 — README + v0.1.0 | en son |

Plan dosyası: `~/.claude/plans/bir-soru-fallback-e-i-ini-prancy-bachman.md`

## Açık ön-kayıtlar (DEĞİŞTİRİLEMEZ)

- **#40** — 9a latency eşiği: merge/mühürleme penceresine denk gelen
  yazmaların p99'u, taban p99'un (7.8 µs, **fsync'siz** ölçüldü) 50 katını
  aşmamalı. Ölçüm koşulu Aşama 8 ile aynı tutulmalı; fsync'li ölçüm eşiği
  otomatik aşar ve yanlış "kabul edilmedi" sonucu doğurur.
- **#49** — 9a-2'nin İKİNCİ kriteri (birikme): 2 dk tam hız yazma altında
  segment+mühürlenen sayısı dengeleniyorsa (son 1/3 ort. ≤ ilk 1/3 ort.
  +%20) **ve** zirve ≤ 12 ise backpressure'sız kabul; aksi halde
  backpressure aynı arc'ta yapılır.
- **9c eşiği (#40):** metadata payı > %25 → GO (ölçüldü: %51.5, f32 modunda,
  hesaplanan boyutlara göre).
- **9b:** NO-GO, `deny(unsafe_code)` bu arc'ta açılmaz (yeniden bakma koşulu
  #40'ta: veri RAM'e sığmazsa).

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

1. `seal()` her çağrıda `thread::spawn` yapıyor → 35 eşzamanlı mühürleme,
   hiçbiri bitmiyor (#52). **Tek worker + kuyruk** yap (merge'deki
   `MergeContext` kalıbı hazır örnek).
2. **Backpressure:** kuyruk eşiği aşılınca yazma yolunda kısa bekleme.
3. Ön-kayıt #49'un İKİ kriterini de yeniden ölç (`mergewindow` +
   `accumulation`); 9a-2 ancak ikisi de geçerse kabul.
4. Sonra 9c → 9d → 10 (plan dosyasındaki sıra).

## Yeniden koşulabilir ölçümler

```
cargo run --release --bin report -- fullscale 1000000 99   # 1M uçtan uca (~10 dk)
cargo run --release --bin report -- int8scale 1000000 99   # 8a
cargo run --release --bin report -- mergewindow 1000000 99 # 9a latency
cargo run --release --bin report -- accumulation 1000000 99 # 9a-2 birikme
cargo run --release --bin report -- coldprofile 1000 99    # soğuk başlangıç bileşenleri
```
