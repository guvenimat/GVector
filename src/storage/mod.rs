//! Kalıcılık altyapısı (Aşama 7a): dayanıklı yazım, çerçeveli dosya formatı,
//! manifest.
//!
//! Tasarım ilkeleri:
//! - **Segment dosyaları değişmezdir.** Adları yazıldıkları generation'ı taşır
//!   (`segment-<gen>-<idx>.gvdb`) ve bir daha asla üzerine yazılmaz. Bu hem
//!   Windows dosya kilitleriyle uyumludur (açık handle'lı dosyaya yazmayız)
//!   hem de her checkpoint'in yalnız YENİ segmentleri yazmasını sağlar —
//!   1M ölçeğinde checkpoint maliyetini belirleyen şey budur.
//! - **Manifest tek gerçek kaynaktır** ve atomik takas edilir (tmp + fsync +
//!   rename). Yarım yazılmış manifest asla görünmez.
//! - **Türetilebilir yapılar diske yazılmaz**: Eq posting-list'leri ve sayısal
//!   alan indeksleri metadata'dan tam olarak yeniden kurulur. Tek kaynak →
//!   tutarsızlık riski yapısal olarak yok.

use crate::distance::Metric;
use crate::index::hnsw::HnswParams;
use crate::meta::{MetaValue, Metadata};
use crate::types::VectorId;
use std::io::Write;
use std::path::{Path, PathBuf};

pub const MANIFEST_NAME: &str = "MANIFEST";
const MANIFEST_MAGIC: [u8; 4] = *b"GVMF";
const METADATA_MAGIC: [u8; 4] = *b"GVMD";
/// Aşama 7 formatı. Aşama 3'ün tek-indeks formatıyla geriye uyum aranmaz.
pub const STORAGE_VERSION: u32 = 2;

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("io hatası: {0}")]
    Io(#[from] std::io::Error),
    #[error("bozuk dosya ({path}): {reason}")]
    Corrupt { path: String, reason: String },
    #[error("desteklenmeyen format versiyonu: {0} (bu sürüm {STORAGE_VERSION} okur)")]
    UnsupportedVersion(u32),
    #[error("serileştirme hatası: {0}")]
    Encode(#[from] bincode::Error),
    #[error("indeks hatası: {0}")]
    Index(#[from] crate::index::IndexError),
    #[error("segment yükleme hatası: {0}")]
    Segment(#[from] crate::index::hnsw::PersistError),
}

fn corrupt(path: impl AsRef<Path>, reason: impl Into<String>) -> StorageError {
    StorageError::Corrupt {
        path: path.as_ref().display().to_string(),
        reason: reason.into(),
    }
}

/// Dayanıklı dosya yazımı: tmp'ye yaz → fsync → rename.
///
/// Windows notu (DECISIONS #32): dizinin kendisi fsync'lenemez — Rust std
/// dizin handle'ı açmaz ve Windows dizin fsync'i genel olarak desteklemez.
/// Dosya içeriği fsync'li, rename atomik (MoveFileEx REPLACE_EXISTING); ancak
/// dizin girdisinin dayanıklılığı işletim sistemine bırakılmış durumda. Bu,
/// güç kesintisinde "checkpoint görünmez ama WAL sağlam" senaryosunu mümkün
/// kılar — kurtarma bu yüzden her zaman WAL replay'iyle tamamlanır.
pub fn write_file_durable(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Çerçeveli dosya: magic + versiyon + uzunluk + bincode payload + CRC32.
/// CRC gövdenin tamamını kapsar ve payload'a DOKUNULMADAN önce doğrulanır —
/// bozuk baytı deserializer'a hiç göstermemek fuzz yüzeyini küçültür.
fn encode_framed<T: serde::Serialize>(magic: [u8; 4], value: &T) -> Result<Vec<u8>, StorageError> {
    let payload = bincode::serialize(value)?;
    let mut buf = Vec::with_capacity(payload.len() + 20);
    buf.extend(magic);
    buf.extend(STORAGE_VERSION.to_le_bytes());
    buf.extend((payload.len() as u64).to_le_bytes());
    buf.extend(&payload);
    let mut h = crc32fast::Hasher::new();
    h.update(&buf);
    buf.extend(h.finalize().to_le_bytes());
    Ok(buf)
}

fn decode_framed<T: serde::de::DeserializeOwned>(
    magic: [u8; 4],
    bytes: &[u8],
    path: &Path,
) -> Result<T, StorageError> {
    if bytes.len() < 20 {
        return Err(corrupt(path, "dosya header için bile kısa"));
    }
    if bytes[0..4] != magic {
        return Err(corrupt(path, "magic uyuşmuyor"));
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().expect("4 byte"));
    if version != STORAGE_VERSION {
        return Err(StorageError::UnsupportedVersion(version));
    }
    let body = &bytes[..bytes.len() - 4];
    let stored = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().expect("4 byte"));
    let mut h = crc32fast::Hasher::new();
    h.update(body);
    if h.finalize() != stored {
        return Err(corrupt(path, "crc32 uyuşmuyor (bozuk/kesik dosya)"));
    }
    let len = u64::from_le_bytes(bytes[8..16].try_into().expect("8 byte")) as usize;
    let end = 16usize
        .checked_add(len)
        .ok_or_else(|| corrupt(path, "payload uzunluğu taşıyor"))?;
    if end > body.len() {
        return Err(corrupt(path, "payload uzunluğu dosyayı aşıyor"));
    }
    Ok(bincode::deserialize(&bytes[16..end])?)
}

// ---------------------------------------------------------------------------
// Disk temsili: MetaValue'nun etiketli ikizi
//
// `MetaValue` HTTP JSON şekli için `#[serde(untagged)]` — `{"renk": "mavi"}`
// gibi doğal gövdeler bunu gerektiriyor. Ama untagged deserialization
// `deserialize_any` ister; bincode self-describing olmadığı için bunu
// desteklemez. Disk (ve Aşama 7b'de WAL) temsili bu yüzden ayrı ve ETİKETLİ.
// API şekli ile depolama şeklini ayırmak zaten sağlıklı: biri sözleşme,
// diğeri iç format, bağımsız evrilebilirler.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) enum MetaValueRepr {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl From<&MetaValue> for MetaValueRepr {
    fn from(v: &MetaValue) -> Self {
        match v {
            MetaValue::Bool(b) => MetaValueRepr::Bool(*b),
            MetaValue::Int(i) => MetaValueRepr::Int(*i),
            MetaValue::Float(f) => MetaValueRepr::Float(*f),
            MetaValue::Str(s) => MetaValueRepr::Str(s.clone()),
        }
    }
}

impl From<MetaValueRepr> for MetaValue {
    fn from(v: MetaValueRepr) -> Self {
        match v {
            MetaValueRepr::Bool(b) => MetaValue::Bool(b),
            MetaValueRepr::Int(i) => MetaValue::Int(i),
            MetaValueRepr::Float(f) => MetaValue::Float(f),
            MetaValueRepr::Str(s) => MetaValue::Str(s),
        }
    }
}

pub(crate) type MetaRepr = Vec<(String, MetaValueRepr)>;

pub(crate) fn meta_to_repr(m: &Metadata) -> MetaRepr {
    m.iter().map(|(k, v)| (k.clone(), v.into())).collect()
}

pub(crate) fn repr_to_meta(r: MetaRepr) -> Metadata {
    r.into_iter().map(|(k, v)| (k, v.into())).collect()
}

/// id → metadata snapshot'ı. Tam yazım (artımlı değil): sıcak yolu WAL
/// taşıdığı için snapshot yalnız checkpoint'te üretilir.
#[derive(serde::Serialize, serde::Deserialize)]
struct MetadataSnapshot {
    entries: Vec<(u64, MetaRepr)>,
}

pub fn encode_metadata(entries: &[(VectorId, Metadata)]) -> Result<Vec<u8>, StorageError> {
    let snap = MetadataSnapshot {
        entries: entries
            .iter()
            .map(|(id, m)| (id.0, meta_to_repr(m)))
            .collect(),
    };
    encode_framed(METADATA_MAGIC, &snap)
}

pub fn decode_metadata(
    bytes: &[u8],
    path: &Path,
) -> Result<Vec<(VectorId, Metadata)>, StorageError> {
    let snap: MetadataSnapshot = decode_framed(METADATA_MAGIC, bytes, path)?;
    Ok(snap
        .entries
        .into_iter()
        .map(|(id, r)| (VectorId(id), repr_to_meta(r)))
        .collect())
}

// ---------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------

/// Bir segmentin manifest kaydı.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SegmentRef {
    /// Dizine göreli dosya adı; değişmez.
    pub file: String,
    /// Dosyanın tamamının CRC32'si (yazım anında hesaplanır, sonra taşınır —
    /// eski segmentleri her checkpoint'te yeniden okumamak için).
    pub crc32: u32,
    /// Segmentteki toplam kayıt (tombstone'lular dahil) — gözlem/GC için.
    pub records: u64,
    /// Segment-yerel tombstone'lar. Segment dosyası değişmez olduğu için
    /// silmeler burada yaşar (bkz. modül başlığı).
    pub tombstones: Vec<u64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Manifest {
    pub generation: u64,
    pub dim: u64,
    pub metric: Metric,
    pub hnsw_params: HnswParams,
    pub seal_threshold: u64,
    pub max_segments: u64,
    pub segments: Vec<SegmentRef>,
    pub metadata_file: Option<String>,
    pub metadata_crc: u32,
    /// Aktif WAL dosyası (Aşama 7b). Yoksa sıcak kalıcılık kapalı demektir.
    pub wal_file: Option<String>,
    pub created_unix_secs: u64,
}

impl Manifest {
    pub fn segment_file_name(generation: u64, idx: usize) -> String {
        format!("segment-{generation:06}-{idx:02}.gvdb")
    }

    pub fn metadata_file_name(generation: u64) -> String {
        format!("meta-{generation:06}.gvmeta")
    }

    pub fn wal_file_name(generation: u64) -> String {
        format!("wal-{generation:06}.log")
    }

    /// Manifest'i atomik olarak yazar.
    pub fn write(&self, dir: &Path) -> Result<(), StorageError> {
        let bytes = encode_framed(MANIFEST_MAGIC, self)?;
        write_file_durable(&dir.join(MANIFEST_NAME), &bytes)?;
        Ok(())
    }

    /// Manifest'i okur ve doğrular. Dosya yoksa `Ok(None)` — boş dizin
    /// "kurtarılacak bir şey yok" demektir, hata değil.
    pub fn read(dir: &Path) -> Result<Option<Manifest>, StorageError> {
        let path = dir.join(MANIFEST_NAME);
        match std::fs::read(&path) {
            Ok(bytes) => Ok(Some(decode_framed(MANIFEST_MAGIC, &bytes, &path)?)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Manifest'te referanslanan dosyaların kümesi (GC için).
    pub fn referenced_files(&self) -> Vec<String> {
        let mut out: Vec<String> = self.segments.iter().map(|s| s.file.clone()).collect();
        out.extend(self.metadata_file.clone());
        out.extend(self.wal_file.clone());
        out.push(MANIFEST_NAME.to_string());
        out
    }
}

/// Manifest'te geçmeyen segment/meta/wal dosyalarını siler.
/// Silinen dosya sayısını döndürür. Manifest yazıldıktan SONRA çağrılmalı:
/// sıra tersine dönerse hâlâ referanslanan bir dosya silinebilirdi.
pub fn gc_unreferenced(dir: &Path, manifest: &Manifest) -> Result<usize, StorageError> {
    let keep = manifest.referenced_files();
    let mut removed = 0usize;
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let is_ours = name.starts_with("segment-")
            || name.starts_with("meta-")
            || name.starts_with("wal-")
            || name.ends_with(".tmp");
        if is_ours && !keep.contains(&name) {
            // Windows'ta açık handle varsa silme başarısız olabilir; bu
            // ölümcül değil — bir sonraki GC dener.
            if std::fs::remove_file(entry.path()).is_ok() {
                removed += 1;
            }
        }
    }
    Ok(removed)
}

/// Dosya yolunu okuyup CRC'sini doğrular.
pub fn read_verified(dir: &Path, file: &str, expect_crc: u32) -> Result<Vec<u8>, StorageError> {
    let path = dir.join(file);
    let bytes = std::fs::read(&path)?;
    let mut h = crc32fast::Hasher::new();
    h.update(&bytes);
    let actual = h.finalize();
    if actual != expect_crc {
        return Err(corrupt(
            &path,
            format!("crc uyuşmuyor: manifest {expect_crc:08x}, dosya {actual:08x}"),
        ));
    }
    Ok(bytes)
}

pub fn crc32(bytes: &[u8]) -> u32 {
    let mut h = crc32fast::Hasher::new();
    h.update(bytes);
    h.finalize()
}

pub fn now_unix_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Test/deney için geçici dizin (scratch): süreç kimliği + sayaçla benzersiz.
pub fn temp_dir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("gvdb-{tag}-{}-{n}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dizin");
    dir
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> Manifest {
        Manifest {
            generation: 3,
            dim: 8,
            metric: Metric::L2,
            hnsw_params: HnswParams::default(),
            seal_threshold: 100,
            max_segments: 8,
            segments: vec![SegmentRef {
                file: Manifest::segment_file_name(3, 0),
                crc32: 0xdead_beef,
                records: 42,
                tombstones: vec![1, 2, 3],
            }],
            metadata_file: Some(Manifest::metadata_file_name(3)),
            metadata_crc: 0x1234_5678,
            wal_file: Some(Manifest::wal_file_name(3)),
            created_unix_secs: 1_700_000_000,
        }
    }

    #[test]
    fn manifest_roundtrip() {
        let dir = temp_dir("manifest");
        let m = sample_manifest();
        m.write(&dir).unwrap();
        let back = Manifest::read(&dir).unwrap().unwrap();
        assert_eq!(back.generation, 3);
        assert_eq!(back.segments.len(), 1);
        assert_eq!(back.segments[0].tombstones, vec![1, 2, 3]);
        assert_eq!(back.wal_file.as_deref(), Some("wal-000003.log"));
    }

    #[test]
    fn manifest_missing_is_none_not_error() {
        let dir = temp_dir("manifest-empty");
        assert!(Manifest::read(&dir).unwrap().is_none());
    }

    #[test]
    fn manifest_corruption_detected_not_panic() {
        let dir = temp_dir("manifest-corrupt");
        sample_manifest().write(&dir).unwrap();
        let path = dir.join(MANIFEST_NAME);
        let good = std::fs::read(&path).unwrap();
        for pos in [0, 5, good.len() / 2, good.len() - 1] {
            let mut bad = good.clone();
            bad[pos] ^= 0x01;
            std::fs::write(&path, &bad).unwrap();
            assert!(Manifest::read(&dir).is_err(), "bozulma @{pos} yakalanmalı");
        }
        // kesik dosya
        for cut in [0, 10, good.len() / 2] {
            std::fs::write(&path, &good[..cut]).unwrap();
            assert!(Manifest::read(&dir).is_err(), "kesik @{cut} yakalanmalı");
        }
    }

    #[test]
    fn metadata_snapshot_roundtrip_all_value_kinds() {
        // untagged/bincode tuzağının regresyon testi: her MetaValue türü
        // disk temsilinden aynen geri gelmeli.
        let m: Metadata = [
            ("b".to_string(), MetaValue::Bool(true)),
            ("i".to_string(), MetaValue::Int(-7)),
            ("f".to_string(), MetaValue::Float(2.5)),
            ("s".to_string(), MetaValue::Str("değer".into())),
        ]
        .into();
        let bytes = encode_metadata(&[(VectorId(9), m.clone())]).unwrap();
        let back = decode_metadata(&bytes, Path::new("test")).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].0, VectorId(9));
        assert_eq!(back[0].1, m);
    }

    #[test]
    fn gc_removes_only_unreferenced() {
        let dir = temp_dir("gc");
        let m = sample_manifest();
        m.write(&dir).unwrap();
        std::fs::write(dir.join(&m.segments[0].file), b"x").unwrap();
        std::fs::write(dir.join("segment-000001-00.gvdb"), b"eski").unwrap();
        std::fs::write(dir.join("meta-000001.gvmeta"), b"eski").unwrap();
        std::fs::write(dir.join("baska.txt"), b"dokunma").unwrap();
        let removed = gc_unreferenced(&dir, &m).unwrap();
        assert_eq!(removed, 2);
        assert!(dir.join(&m.segments[0].file).exists());
        assert!(dir.join("baska.txt").exists(), "yabancı dosyaya dokunulmaz");
        assert!(dir.join(MANIFEST_NAME).exists());
    }

    #[test]
    fn read_verified_catches_wrong_crc() {
        let dir = temp_dir("verify");
        std::fs::write(dir.join("segment-000001-00.gvdb"), b"veri").unwrap();
        let good = crc32(b"veri");
        assert!(read_verified(&dir, "segment-000001-00.gvdb", good).is_ok());
        assert!(read_verified(&dir, "segment-000001-00.gvdb", good ^ 1).is_err());
    }

    #[test]
    fn durable_write_replaces_atomically() {
        let dir = temp_dir("durable");
        let p = dir.join("f.bin");
        write_file_durable(&p, b"ilk").unwrap();
        write_file_durable(&p, b"ikinci").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"ikinci");
        assert!(!p.with_extension("tmp").exists(), "tmp temizlenmeli");
    }
}
