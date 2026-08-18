//! Deserialization fuzz hedefi: rastgele baytlar HİÇBİR girdide panic
//! üretmemeli — geçersiz dosya her zaman Err ile dönmeli.
//!
//! Çalıştırma (nightly gerektirir):
//!   cargo install cargo-fuzz
//!   cargo +nightly fuzz run load_index
#![no_main]

use libfuzzer_sys::fuzz_target;
use vector_gvector::index::hnsw::HnswIndex;

fuzz_target!(|data: &[u8]| {
    // Sonuç ne olursa olsun (Ok/Err) kabul; tek başarısızlık kriteri panic.
    let _ = HnswIndex::load_from_bytes(data);
});
