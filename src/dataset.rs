//! Veri seti yükleme: fvecs/ivecs (SIFT1M formatı), alt küme çıkarma,
//! seed'li rastgele veri üretimi.
//!
//! fvecs/ivecs formatı: her vektör `[d: u32 little-endian][d eleman]`
//! şeklinde art arda dizilir; başka header yoktur. f32 ve i32 elemanlar
//! da little-endian'dır.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::io::{self, Read};
use std::path::Path;

/// Benchmark tekrarlanabilirliği için projedeki varsayılan seed.
pub const DEFAULT_SEED: u64 = 42;

#[derive(Debug, thiserror::Error)]
pub enum DatasetError {
    #[error("io hatası: {0}")]
    Io(#[from] io::Error),
    #[error("bozuk dosya: {0}")]
    Malformed(String),
}

/// fvecs dosyasını okur. Tüm vektörlerin aynı boyutta olmasını doğrular.
pub fn read_fvecs(path: &Path) -> Result<Vec<Vec<f32>>, DatasetError> {
    let bytes = std::fs::read(path)?;
    parse_fvecs(&bytes)
}

/// ivecs dosyasını okur (ground truth komşu id listeleri için).
pub fn read_ivecs(path: &Path) -> Result<Vec<Vec<i32>>, DatasetError> {
    let bytes = std::fs::read(path)?;
    parse_ivecs(&bytes)
}

/// Bellekten fvecs ayrıştırma (test edilebilirlik için IO'dan ayrık).
pub fn parse_fvecs(bytes: &[u8]) -> Result<Vec<Vec<f32>>, DatasetError> {
    parse_vecs(bytes, f32::from_le_bytes)
}

pub fn parse_ivecs(bytes: &[u8]) -> Result<Vec<Vec<i32>>, DatasetError> {
    parse_vecs(bytes, i32::from_le_bytes)
}

// Ortak ayrıştırıcı: eleman tipi sadece 4 byte'tan değer üretme şeklinde ayrışır.
fn parse_vecs<T>(
    mut bytes: &[u8],
    from_le: impl Fn([u8; 4]) -> T,
) -> Result<Vec<Vec<T>>, DatasetError> {
    let mut out = Vec::new();
    let mut expected_dim: Option<usize> = None;
    while !bytes.is_empty() {
        if bytes.len() < 4 {
            return Err(DatasetError::Malformed(
                "boyut alanı için 4 byte kalmadı (kesik dosya)".into(),
            ));
        }
        let dim = u32::from_le_bytes(bytes[..4].try_into().expect("4 byte")) as usize;
        if dim == 0 || dim > 1 << 20 {
            return Err(DatasetError::Malformed(format!(
                "mantıksız boyut alanı: {dim}"
            )));
        }
        match expected_dim {
            None => expected_dim = Some(dim),
            Some(e) if e != dim => {
                return Err(DatasetError::Malformed(format!(
                    "tutarsız boyut: {e} beklenirken {dim} görüldü"
                )));
            }
            _ => {}
        }
        let need = 4 + dim * 4;
        if bytes.len() < need {
            return Err(DatasetError::Malformed(format!(
                "vektör verisi kesik: {need} byte gerekli, {} kaldı",
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

/// Reader üzerinden ilk `n` vektörü okur — 1M'lik dosyadan 10K/100K alt küme
/// çıkarırken dosyanın tamamını belleğe almamak için.
pub fn read_fvecs_subset<R: Read>(reader: &mut R, n: usize) -> Result<Vec<Vec<f32>>, DatasetError> {
    let mut out = Vec::with_capacity(n);
    let mut dim_buf = [0u8; 4];
    for _ in 0..n {
        match reader.read_exact(&mut dim_buf) {
            Ok(()) => {}
            // dosya n'den önce bitti: elde olanı döndür (alt küme isteği aşkın olabilir)
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;
        if dim == 0 || dim > 1 << 20 {
            return Err(DatasetError::Malformed(format!(
                "mantıksız boyut alanı: {dim}"
            )));
        }
        let mut data = vec![0u8; dim * 4];
        reader
            .read_exact(&mut data)
            .map_err(|_| DatasetError::Malformed("vektör verisi kesik (subset okuma)".into()))?;
        out.push(
            data.chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().expect("4 byte")))
                .collect(),
        );
    }
    Ok(out)
}

/// Seed'li rastgele veri üreteci: `n` adet `dim` boyutlu, [-1, 1) bileşenli vektör.
/// Aynı seed her zaman aynı veriyi üretir — benchmark tekrarlanabilirliği.
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
        // istenen n eldekinden fazlaysa hepsi döner, hata olmaz
        let all = read_fvecs_subset(&mut &bytes[..], 100).unwrap();
        assert_eq!(all.len(), 10);
    }

    #[test]
    fn random_vectors_deterministic() {
        assert_eq!(random_vectors(5, 8, 42), random_vectors(5, 8, 42));
        assert_ne!(random_vectors(5, 8, 42), random_vectors(5, 8, 43));
    }
}
