# Mimari Kararlar

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
