//! Dataset loading: fvecs/ivecs (the SIFT1M format), subset extraction, and
//! seeded random data generation.
//!
//! The fvecs/ivecs format: vectors are laid out back to back as
//! `[d: u32 little-endian][d elements]`; there is no other header. The f32
//! and i32 elements are little-endian as well.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::{self, Read};
use std::path::Path;

/// The project-wide default seed, for benchmark reproducibility.
pub const DEFAULT_SEED: u64 = 42;

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("malformed file: {0}")]
    Malformed(String),
}

/// Reads an fvecs file. Verifies that every vector has the same dimension.
pub fn read_fvecs(path: &Path) -> Result<Vec<Vec<f32>>, DatasetError> {
    let bytes = std::fs::read(path)?;
    parse_fvecs(&bytes)
}

/// Reads an ivecs file (for ground-truth neighbour id lists).
pub fn read_ivecs(path: &Path) -> Result<Vec<Vec<i32>>, DatasetError> {
    let bytes = std::fs::read(path)?;
    parse_ivecs(&bytes)
}

/// Parses fvecs from memory (kept separate from IO for testability).
pub fn parse_fvecs(bytes: &[u8]) -> Result<Vec<Vec<f32>>, DatasetError> {
    parse_vecs(bytes, f32::from_le_bytes)
}

pub fn parse_ivecs(bytes: &[u8]) -> Result<Vec<Vec<i32>>, DatasetError> {
    parse_vecs(bytes, i32::from_le_bytes)
}

// Shared parser: the element type differs only in how a value is built from
// 4 bytes.
fn parse_vecs<T>(
    mut bytes: &[u8],
    from_le: impl Fn([u8; 4]) -> T,
) -> Result<Vec<Vec<T>>, DatasetError> {
    let mut out = Vec::new();
    let mut expected_dim: Option<usize> = None;
    while !bytes.is_empty() {
        if bytes.len() < 4 {
            return Err(DatasetError::Malformed(
                "no 4 bytes left for the dimension field (truncated file)".into(),
            ));
        }
        let dim = u32::from_le_bytes(bytes[..4].try_into().expect("4 byte")) as usize;
        if dim == 0 || dim > 1 << 20 {
            return Err(DatasetError::Malformed(format!(
                "implausible dimension field: {dim}"
            )));
        }
        match expected_dim {
            None => expected_dim = Some(dim),
            Some(e) if e != dim => {
                return Err(DatasetError::Malformed(format!(
                    "inconsistent dimension: expected {e}, saw {dim}"
                )));
            }
            _ => {}
        }
        let need = 4 + dim * 4;
        if bytes.len() < need {
            return Err(DatasetError::Malformed(format!(
                "vector data truncated: {need} bytes needed, {} left",
                bytes.len()
            )));
        }
        let v = bytes[4..need]
            .chunks_exact(4)
            .map(|c| from_le(c.try_into().expect("4 byte")))
            .collect();
        out.push(v);
        bytes = &bytes[need..];
    }
    Ok(out)
}

/// Reads the first `n` vectors from a reader — so that extracting a 10K/100K
/// subset out of the 1M file does not pull the whole file into memory.
pub fn read_fvecs_subset<R: Read>(reader: &mut R, n: usize) -> Result<Vec<Vec<f32>>, DatasetError> {
    let mut out = Vec::with_capacity(n);
    let mut dim_buf = [0u8; 4];
    for _ in 0..n {
        match reader.read_exact(&mut dim_buf) {
            Ok(()) => {}
            // the file ended before n: return what we have (the requested
            // subset may be larger than the file)
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;
        if dim == 0 || dim > 1 << 20 {
            return Err(DatasetError::Malformed(format!(
                "implausible dimension field: {dim}"
            )));
        }
        let mut data = vec![0u8; dim * 4];
        reader
            .read_exact(&mut data)
            .map_err(|_| DatasetError::Malformed("vector data truncated (subset read)".into()))?;
        out.push(
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().expect("4 byte")))
                .collect(),
        );
    }
    Ok(out)
}

/// Seeded random data generator: `n` vectors of `dim` dimensions with
/// components in [-1, 1). The same seed always produces the same data —
/// benchmark reproducibility.
pub fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode_fvecs(vecs: &[Vec<f32>]) -> Vec<u8> {
        let mut out = Vec::new();
        for v in vecs {
            out.extend((v.len() as u32).to_le_bytes());
            for x in v {
                out.extend(x.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn fvecs_roundtrip() {
        let vecs = vec![vec![1.0f32, -2.5, 3.25], vec![0.0, 7.0, -0.5]];
        let parsed = parse_fvecs(&encode_fvecs(&vecs)).unwrap();
        assert_eq!(parsed, vecs);
    }

    #[test]
    fn fvecs_empty_input_ok() {
        assert!(parse_fvecs(&[]).unwrap().is_empty());
    }

    #[test]
    fn fvecs_truncated_is_error_not_panic() {
        let mut bytes = encode_fvecs(&[vec![1.0f32, 2.0]]);
        bytes.truncate(bytes.len() - 3);
        assert!(matches!(
            parse_fvecs(&bytes),
            Err(DatasetError::Malformed(_))
        ));
    }

    #[test]
    fn fvecs_inconsistent_dim_is_error() {
        let bytes = encode_fvecs(&[vec![1.0f32, 2.0], vec![1.0f32, 2.0, 3.0]]);
        assert!(matches!(
            parse_fvecs(&bytes),
            Err(DatasetError::Malformed(_))
        ));
    }

    #[test]
    fn subset_reads_first_n() {
        let vecs: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32; 4]).collect();
        let bytes = encode_fvecs(&vecs);
        let subset = read_fvecs_subset(&mut &bytes[..], 3).unwrap();
        assert_eq!(subset, vecs[..3]);
        // if the requested n exceeds what is available, all are returned, not an error
        let all = read_fvecs_subset(&mut &bytes[..], 100).unwrap();
        assert_eq!(all.len(), 10);
    }

    #[test]
    fn random_vectors_deterministic() {
        assert_eq!(random_vectors(5, 8, 42), random_vectors(5, 8, 42));
        assert_ne!(random_vectors(5, 8, 42), random_vectors(5, 8, 43));
    }
}
