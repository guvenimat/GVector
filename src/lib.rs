#![deny(unsafe_code)]
//! Sıfırdan yazılmış bir vektör arama motoru.
//!
//! Modül sınırları bilinçli olarak indekslerden bağımsız tutuldu:
//! `distance`, `dataset` ve `eval` hem brute-force hem HNSW tarafından
//! aynen kullanılacak, böylece aşamalar arası ölçümler karşılaştırılabilir kalır.

pub mod dataset;
pub mod distance;
pub mod eval;
pub mod index;
pub mod types;
