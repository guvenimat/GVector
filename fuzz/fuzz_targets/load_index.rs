//! Deserialization fuzz target: random bytes must NEVER cause a panic for any
//! input — an invalid file must always come back as Err.
//!
//! Running it (requires nightly):
//!   cargo install cargo-fuzz
//!   cargo +nightly fuzz run load_index
#![no_main]

use libfuzzer_sys::fuzz_target;
use vector_gvector::index::hnsw::HnswIndex;

fuzz_target!(|data: &[u8]| {
    // Any outcome (Ok/Err) is acceptable; the only failure criterion is a panic.
    let _ = HnswIndex::load_from_bytes(data);
});
