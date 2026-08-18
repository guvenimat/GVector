# Mimari Kararlar

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
