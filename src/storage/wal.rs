//! Write-ahead log (Aşama 7b): yazma buffer'ının sıcak dayanıklılığı.
//!
//! Segmentler checkpoint ile dayanıklı; aradaki yazmalar yalnız bellekte
//! yaşıyordu. WAL bu boşluğu kapatır: her mutasyon önce loga, sonra belleğe.
//!
//! **Çerçeve:** `[len: u32 LE][crc32: u32 LE][payload: bincode]`.
//! CRC yalnız payload'ı kapsar; uzunluk başlığı ayrı okunur ki kesik kuyruk
//! (dosya kaydın ortasında bitmiş) ile bozuk gövde ayırt edilebilsin.
//!
//! **Kurtarma sözleşmesi:** replay ilk tutarsızlıkta DURUR — kısmi kayıt,
//! CRC uyuşmazlığı ya da mantıksız uzunluk. Hayalet op asla türetilmez;
//! o noktadan sonrası yok sayılır ve dosya ORADA KESİLİR. Kesmezsek bir
//! sonraki append bozuk kuyruğun üstüne yazar ve dosya kalıcı olarak
//! tutarsız kalırdı.

use crate::meta::Metadata;
use crate::storage::{meta_to_repr, repr_to_meta, MetaRepr, StorageError};
use crate::types::VectorId;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Tek kaydın üst sınırı — bozuk uzunluk başlığında GB'lık ayırma yapmamak
/// için. 128 boyutlu vektör + metadata bunun binde biri.
const MAX_RECORD_BYTES: u32 = 64 * 1024 * 1024;

/// fsync politikası. Hangi politikada HTTP 200'ün ne anlama geldiği
/// DECISIONS #36'da tanımlı.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// fsync yok; dayanıklılık yalnız checkpoint'te. WAL yine de yazılır,
    /// yani SÜREÇ çökmesine dayanır — MAKİNE çökmesine dayanmaz.
    None,
    /// Her kayıttan sonra fsync. En güvenli, en yavaş.
    PerOp,
    /// Grup commit: pencere içindeki kayıtlar tek fsync paylaşır.
    /// Yanıt yine fsync'i bekler (bkz. DECISIONS #36).
    Group { window_ms: u64 },
}

impl SyncPolicy {
    pub fn parse(s: &str) -> Option<SyncPolicy> {
        match s {
            "none" => Some(SyncPolicy::None),
            "per_op" | "per-op" => Some(SyncPolicy::PerOp),
            _ => s
                .strip_prefix("group:")
                .and_then(|ms| ms.parse().ok())
                .map(|window_ms| SyncPolicy::Group { window_ms }),
        }
    }

    pub fn label(&self) -> String {
        match self {
            SyncPolicy::None => "none".into(),
            SyncPolicy::PerOp => "per_op".into(),
            SyncPolicy::Group { window_ms } => format!("group:{window_ms}"),
        }
    }
}

/// WAL kaydı. Checkpoint işareti YOK: checkpoint WAL'ı rotasyona sokup yeni
/// dosya açtığı için "bu noktadan öncesi segmentlerde" bilgisi dosya
/// sınırının kendisidir.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum WalRecord {
    Insert {
        id: u64,
        vector: Vec<f32>,
        meta: MetaRepr,
    },
    Delete {
        id: u64,
    },
}

impl WalRecord {
    pub fn insert(id: VectorId, vector: &[f32], meta: &Metadata) -> Self {
        WalRecord::Insert {
            id: id.0,
            vector: vector.to_vec(),
            meta: meta_to_repr(meta),
        }
    }

    pub fn delete(id: VectorId) -> Self {
        WalRecord::Delete { id: id.0 }
    }
}

/// Replay sonucu — kaç kayıt uygulandı, nerede durdu ve neden.
#[derive(Debug, Clone, Default)]
pub struct ReplayReport {
    pub applied: usize,
    pub bytes_ok: u64,
    /// Sağlam önekin bittiği offset; dosya bundan sonrası atıldıysa Some.
    pub truncated_at: Option<u64>,
    pub reason: Option<String>,
}

pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
    policy: SyncPolicy,
    bytes: u64,
    records: u64,
    /// Son fsync'ten beri yazılmış (henüz dayanıklı olmayan) kayıt var mı?
    dirty: bool,
    last_sync: Instant,
}

impl Wal {
    /// Var olan dosyaya ekleyerek açar; yoksa oluşturur.
    pub fn open_append(path: PathBuf, policy: SyncPolicy) -> Result<Wal, StorageError> {
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let bytes = file.metadata()?.len();
        Ok(Wal {
            path,
            writer: BufWriter::new(file),
            policy,
            bytes,
            records: 0,
            dirty: false,
            last_sync: Instant::now(),
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn file_name(&self) -> String {
        self.path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default()
    }

    pub fn len_bytes(&self) -> u64 {
        self.bytes
    }

    pub fn policy(&self) -> SyncPolicy {
        self.policy
    }

    /// Kaydı loga ekler. `PerOp`'ta fsync burada olur; `Group`'ta pencere
    /// dolduysa olur; `None`'da hiç olmaz.
    pub fn append(&mut self, rec: &WalRecord) -> Result<(), StorageError> {
        let payload = bincode::serialize(rec)?;
        if payload.len() as u64 > MAX_RECORD_BYTES as u64 {
            return Err(StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "kayıt üst sınırı aşıyor",
            )));
        }
        let crc = crate::storage::crc32(&payload);
        self.writer
            .write_all(&(payload.len() as u32).to_le_bytes())?;
        self.writer.write_all(&crc.to_le_bytes())?;
        self.writer.write_all(&payload)?;
        self.bytes += 8 + payload.len() as u64;
        self.records += 1;
        self.dirty = true;
        match self.policy {
            SyncPolicy::PerOp => self.sync()?,
            SyncPolicy::Group { window_ms } => {
                if self.last_sync.elapsed() >= Duration::from_millis(window_ms) {
                    self.sync()?;
                }
            }
            SyncPolicy::None => {
                // Yalnız buffer'ı OS'e ver; fsync checkpoint'e kalır.
                self.writer.flush()?;
            }
        }
        Ok(())
    }

    /// Zorla fsync (grup penceresini kapatır; graceful shutdown ve checkpoint
    /// bunu çağırır). Zaten temizse ucuz.
    pub fn sync(&mut self) -> Result<(), StorageError> {
        self.writer.flush()?;
        if self.dirty {
            self.writer.get_ref().sync_data()?;
            self.dirty = false;
        }
        self.last_sync = Instant::now();
        Ok(())
    }

    /// Grup penceresi dolduysa fsync'ler. Yazıcı task'i batch sonunda çağırır.
    pub fn sync_if_due(&mut self) -> Result<bool, StorageError> {
        match self.policy {
            SyncPolicy::Group { window_ms } => {
                if self.dirty && self.last_sync.elapsed() >= Duration::from_millis(window_ms) {
                    self.sync()?;
                    return Ok(true);
                }
                Ok(false)
            }
            _ => Ok(false),
        }
    }

    /// Henüz dayanıklı olmayan kayıt var mı?
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Batch sonu commit: politikanın vaat ettiği dayanıklılığı sağlar.
    /// `None`'da fsync YOK (sözleşme bu; yalnız OS'e teslim), diğerlerinde
    /// fsync. Yazıcı task'i yanıtları göndermeden önce bunu çağırır —
    /// group commit'in "200 = fsync'lendi" sözleşmesi buna dayanır.
    pub fn commit(&mut self) -> Result<(), StorageError> {
        match self.policy {
            SyncPolicy::None => {
                self.writer.flush()?;
                Ok(())
            }
            _ => self.sync(),
        }
    }
}

/// WAL dosyasını sırayla okur. İlk tutarsızlıkta durur ve dosyayı sağlam
/// önekin sonunda keser. Dosya yoksa boş rapor döner (hata değil).
pub fn replay(path: &Path) -> Result<(Vec<WalRecord>, ReplayReport), StorageError> {
    let mut report = ReplayReport::default();
    let mut file = match File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), report)),
        Err(e) => return Err(e.into()),
    };
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let (records, rep) = replay_bytes(&bytes);
    report = rep;
    if let Some(cut) = report.truncated_at {
        // Bozuk/kesik kuyruğu at: sonraki append'ler temiz devam etsin.
        let f = OpenOptions::new().write(true).open(path)?;
        f.set_len(cut)?;
        f.sync_all()?;
    }
    Ok((records, report))
}

/// Bayt tamponundan replay — testler ve fuzz bu yolu paylaşır (dosya IO yok).
pub fn replay_bytes(bytes: &[u8]) -> (Vec<WalRecord>, ReplayReport) {
    let mut out = Vec::new();
    let mut report = ReplayReport::default();
    let mut off = 0usize;
    loop {
        if off == bytes.len() {
            report.bytes_ok = off as u64;
            return (out, report); // temiz son
        }
        let stop = |reason: &str| -> Option<String> { Some(reason.to_string()) };
        if bytes.len() - off < 8 {
            report.reason = stop("kesik kayıt başlığı");
            break;
        }
        let len = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4 byte"));
        let crc = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().expect("4 byte"));
        if len == 0 || len > MAX_RECORD_BYTES {
            report.reason = stop("mantıksız kayıt uzunluğu");
            break;
        }
        let end = off + 8 + len as usize;
        if end > bytes.len() {
            report.reason = stop("kesik kayıt gövdesi");
            break;
        }
        let payload = &bytes[off + 8..end];
        if crate::storage::crc32(payload) != crc {
            report.reason = stop("crc uyuşmuyor");
            break;
        }
        match bincode::deserialize::<WalRecord>(payload) {
            Ok(rec) => {
                out.push(rec);
                report.applied += 1;
                off = end;
            }
            Err(_) => {
                // CRC tuttu ama çözülemedi: format uyuşmazlığı. Yine dur.
                report.reason = stop("kayıt çözülemedi");
                break;
            }
        }
    }
    report.bytes_ok = off as u64;
    report.truncated_at = Some(off as u64);
    (out, report)
}

/// Kaydın metadata'sını uygulama tipine çevirir.
pub fn record_meta(meta: MetaRepr) -> Metadata {
    repr_to_meta(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::meta::MetaValue;
    use crate::storage::temp_dir;

    fn sample_records() -> Vec<WalRecord> {
        let meta: Metadata = [
            ("renk".to_string(), MetaValue::Str("mavi".into())),
            ("yil".to_string(), MetaValue::Int(2020)),
        ]
        .into();
        vec![
            WalRecord::insert(VectorId(1), &[1.0, 2.0, 3.0], &meta),
            WalRecord::insert(VectorId(2), &[4.0, 5.0, 6.0], &Metadata::new()),
            WalRecord::delete(VectorId(1)),
        ]
    }

    fn write_all(dir: &Path, policy: SyncPolicy) -> PathBuf {
        let path = dir.join("wal-test.log");
        let mut wal = Wal::open_append(path.clone(), policy).unwrap();
        for r in sample_records() {
            wal.append(&r).unwrap();
        }
        wal.sync().unwrap();
        path
    }

    #[test]
    fn roundtrip_all_policies() {
        for policy in [
            SyncPolicy::None,
            SyncPolicy::PerOp,
            SyncPolicy::Group { window_ms: 5 },
        ] {
            let dir = temp_dir("wal-rt");
            let path = write_all(&dir, policy);
            let (recs, rep) = replay(&path).unwrap();
            assert_eq!(recs, sample_records(), "politika {policy:?}");
            assert_eq!(rep.applied, 3);
            assert!(rep.truncated_at.is_none(), "temiz dosya kesilmemeli");
        }
    }

    #[test]
    fn append_after_reopen_preserves_records() {
        let dir = temp_dir("wal-append");
        let path = write_all(&dir, SyncPolicy::PerOp);
        {
            let mut wal = Wal::open_append(path.clone(), SyncPolicy::PerOp).unwrap();
            wal.append(&WalRecord::delete(VectorId(2))).unwrap();
            wal.sync().unwrap();
        }
        let (recs, _) = replay(&path).unwrap();
        assert_eq!(recs.len(), 4);
        assert_eq!(recs[3], WalRecord::delete(VectorId(2)));
    }

    #[test]
    fn truncated_tail_stops_at_prefix_and_file_is_cut() {
        let dir = temp_dir("wal-trunc");
        let path = write_all(&dir, SyncPolicy::PerOp);
        let good = std::fs::read(&path).unwrap();
        // Kaydın ORTASINDA kes
        for cut in [good.len() - 1, good.len() - 5, good.len() / 2] {
            std::fs::write(&path, &good[..cut]).unwrap();
            let (recs, rep) = replay(&path).unwrap();
            assert!(recs.len() < 3, "kesik dosyadan tam kayıt çıktı: {cut}");
            assert!(rep.truncated_at.is_some());
            // dosya sağlam önekte kesilmiş olmalı
            let after = std::fs::metadata(&path).unwrap().len();
            assert_eq!(after, rep.bytes_ok);
            // ikinci replay aynı sonucu vermeli ve artık kesmemeli
            let (recs2, rep2) = replay(&path).unwrap();
            assert_eq!(recs2, recs);
            assert!(rep2.truncated_at.is_none(), "ikinci replay temiz olmalı");
        }
    }

    #[test]
    fn record_boundary_truncation_keeps_all_prior() {
        let dir = temp_dir("wal-boundary");
        let path = write_all(&dir, SyncPolicy::PerOp);
        let good = std::fs::read(&path).unwrap();
        // İlk kaydın tam sınırında kes
        let first_len = u32::from_le_bytes(good[0..4].try_into().unwrap()) as usize;
        let boundary = 8 + first_len;
        std::fs::write(&path, &good[..boundary]).unwrap();
        let (recs, rep) = replay(&path).unwrap();
        assert_eq!(recs.len(), 1, "sınırda kesme tam kaydı korumalı");
        assert!(rep.truncated_at.is_none(), "sınır temiz son sayılır");
    }

    #[test]
    fn corrupt_payload_stops_without_ghost_ops() {
        let dir = temp_dir("wal-corrupt");
        let path = write_all(&dir, SyncPolicy::PerOp);
        let good = std::fs::read(&path).unwrap();
        // İkinci kaydın gövdesini boz
        let first_len = u32::from_le_bytes(good[0..4].try_into().unwrap()) as usize;
        let second_body = 8 + first_len + 8;
        let mut bad = good.clone();
        bad[second_body + 1] ^= 0xff;
        std::fs::write(&path, &bad).unwrap();
        let (recs, rep) = replay(&path).unwrap();
        assert_eq!(recs.len(), 1, "bozuk kayıttan sonrası uygulanmamalı");
        assert_eq!(rep.reason.as_deref(), Some("crc uyuşmuyor"));
        assert_eq!(rep.truncated_at, Some(8 + first_len as u64));
    }

    #[test]
    fn garbage_length_header_is_rejected() {
        let dir = temp_dir("wal-garbage");
        let path = dir.join("w.log");
        // Devasa uzunluk başlığı: ayırma denemeden reddedilmeli
        let mut bytes = u32::MAX.to_le_bytes().to_vec();
        bytes.extend(0u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let (recs, rep) = replay(&path).unwrap();
        assert!(recs.is_empty());
        assert_eq!(rep.reason.as_deref(), Some("mantıksız kayıt uzunluğu"));
    }

    #[test]
    fn empty_and_missing_files_are_clean() {
        let dir = temp_dir("wal-empty");
        let (recs, rep) = replay(&dir.join("yok.log")).unwrap();
        assert!(recs.is_empty() && rep.truncated_at.is_none());
        let p = dir.join("bos.log");
        std::fs::write(&p, b"").unwrap();
        let (recs, rep) = replay(&p).unwrap();
        assert!(recs.is_empty() && rep.truncated_at.is_none());
    }

    #[test]
    fn policy_parse_roundtrip() {
        for s in ["none", "per_op", "group:20"] {
            let p = SyncPolicy::parse(s).unwrap();
            assert_eq!(SyncPolicy::parse(&p.label()).unwrap(), p);
        }
        assert!(SyncPolicy::parse("saçma").is_none());
    }
}
