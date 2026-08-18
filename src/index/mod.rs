//! `VectorIndex` trait'i ve implementasyonları.
//!
//! Aşama 1: `bruteforce` (doğruluk referansı, proje boyunca kalacak).
//! Aşama 2: `hnsw`.

pub mod bruteforce;
pub mod hnsw;
pub mod numeric;
pub mod quant;
pub mod segmented;

use crate::types::{SearchResult, VectorId};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum IndexError {
    #[error("boyut uyuşmazlığı: indeks {expected} bekliyor, {got} geldi")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("id zaten mevcut: {0:?}")]
    DuplicateId(VectorId),
    #[error("id bulunamadı: {0:?}")]
    NotFound(VectorId),
    #[error("desteklenmiyor: {0}")]
    Unsupported(&'static str),
    /// WAL/disk yazımı başarısız. Write-ahead sırası gereği bu hata
    /// döndüğünde mutasyon belleğe UYGULANMAMIŞTIR.
    #[error("kalıcılık hatası: {0}")]
    Storage(String),
}

/// Tüm indeks implementasyonlarının ortak arayüzü.
///
/// Mutasyon imzaları `&mut self` — gerekçe (özet, tamamı DECISIONS.md'de):
/// tekil bir indeks yapısı tek-yazar semantiğiyle en basit halinde kalır;
/// Aşama 5'teki eşzamanlılık trait'e interior mutability gömerek değil,
/// indeksin ÜSTÜNE bir katman (COW/arc-swap veya segment modeli) sarılarak
/// eklenecek. Böylece algoritma kodu lock/atomics düşünmeden yazılıp test edilir.
pub trait VectorIndex {
    /// Vektörü verilen id ile ekler. Cosine metrikli indeksler vektörü
    /// burada normalize etmekle yükümlüdür (bkz. distance modülü sözleşmesi).
    fn insert(&mut self, id: VectorId, vector: &[f32]) -> Result<(), IndexError>;

    /// En yakın `k` sonucu artan mesafe sırasıyla döndürür.
    /// `k > len()` ise mevcut tüm elemanları döndürür (hata değil).
    /// Boş indekste boş vektör döner.
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;

    /// Var olan bir kaydı siler; yoksa `NotFound` döner.
    fn delete(&mut self, id: VectorId) -> Result<(), IndexError>;

    /// Aranabilir (silinmemiş) eleman sayısı.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
