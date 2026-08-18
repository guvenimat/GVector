//! Projenin ortak temel tipleri.

/// Vektörlerin dıştan verilen kimliği.
///
/// Newtype: çıplak `u64` yerine ayrı tip, id ile graf içi offset'lerin
/// (ileride `usize` slot indeksleri) karışmasını derleme zamanında engeller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VectorId(pub u64);

/// Sahipli vektör verisi. Boyut kontrolü indeksin sorumluluğunda;
/// bu tip sadece taşıyıcı.
pub type Vector = Vec<f32>;

/// Tek bir arama sonucu.
///
/// `distance` her zaman "küçük daha iyi" anlamındadır: L2 için gerçek mesafe,
/// cosine/dot için negatiflenmiş benzerlik (bkz. `distance` modülü). Böylece
/// sıralama mantığı metrikten bağımsız tek biçimde yazılır.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
}

impl SearchResult {
    pub fn new(id: VectorId, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl Eq for SearchResult {}

// f32 NaN yüzünden Ord değildir; mesafe fonksiyonlarımız NaN üretmez
// (sıfır vektör dahil, bkz. distance modülü) — total_cmp ile toplam sıralama
// tanımlıyoruz ki BinaryHeap'te doğrudan kullanılabilsin.
impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}
