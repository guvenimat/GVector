//! The `VectorIndex` trait and its implementations.
//!
//! Phase 1: `bruteforce` (the correctness reference, kept for the whole
//! project). Phase 2: `hnsw`.

pub mod bruteforce;
pub mod hnsw;
pub mod numeric;
pub mod quant;
pub mod segmented;

use crate::types::{SearchResult, VectorId};
use thiserror::Error;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum IndexError {
    #[error("dimension mismatch: index expects {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    #[error("id already exists: {0:?}")]
    DuplicateId(VectorId),
    #[error("id not found: {0:?}")]
    NotFound(VectorId),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    /// A WAL/disk write failed. Because of the write-ahead ordering, when this
    /// error is returned the mutation has NOT been applied to memory.
    #[error("persistence error: {0}")]
    Storage(String),
}

/// The common interface of every index implementation.
///
/// Mutating methods take `&mut self` — rationale (summary; the full version is
/// in DECISIONS.md): a single index structure stays simplest under
/// single-writer semantics. The concurrency introduced in phase 5 is not built
/// by burying interior mutability in the trait, but by wrapping a layer ON TOP
/// of the index (COW/arc-swap, or the segment model). That way the algorithm
/// code is written and tested without thinking about locks or atomics.
pub trait VectorIndex {
    /// Inserts a vector under the given id. Indexes with the cosine metric are
    /// responsible for normalizing the vector here (see the contract in the
    /// distance module).
    fn insert(&mut self, id: VectorId, vector: &[f32]) -> Result<(), IndexError>;

    /// Returns the nearest `k` results in ascending distance order.
    /// If `k > len()`, returns all available elements (not an error).
    /// Returns an empty vector for an empty index.
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;

    /// Deletes an existing record; returns `NotFound` if it does not exist.
    fn delete(&mut self, id: VectorId) -> Result<(), IndexError>;

    /// Number of searchable (non-deleted) elements.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
