#![deny(unsafe_code)]
//! A vector search engine written from scratch.
//!
//! The module boundaries are deliberately kept independent of the indexes:
//! `distance`, `dataset` and `eval` are used verbatim by both brute-force and
//! HNSW, so that measurements stay comparable across phases.

pub mod dataset;
pub mod distance;
pub mod eval;
pub mod index;
pub mod meta;
pub mod storage;
pub mod types;
