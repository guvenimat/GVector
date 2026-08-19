//! Write-ahead log (phase 7b): hot durability for the write buffer.
//!
//! Segments are made durable by checkpoints; the writes in between lived only
//! in memory. The WAL closes that gap: every mutation goes to the log first,
//! then to memory.
//!
//! **Framing:** `[len: u32 LE][crc32: u32 LE][payload: bincode]`.
//! The CRC covers the payload only; the length header is read separately so
//! that a truncated tail (the file ended mid-record) can be told apart from a
//! corrupted body.
//!
//! **Recovery contract:** replay STOPS at the first inconsistency — a partial
//! record, a CRC mismatch or an implausible length. A phantom operation is
//! never synthesized; everything past that point is discarded and the file is
//! TRUNCATED THERE. Without truncating, the next append would write on top of
//! a corrupted tail and the file would stay permanently inconsistent.

use crate::meta::Metadata;
use crate::storage::{meta_to_repr, repr_to_meta, MetaRepr, StorageError};
use crate::types::VectorId;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Upper bound for a single record — so a corrupted length header cannot
/// trigger a gigabyte allocation. A 128-dimensional vector plus metadata is a
/// thousandth of this.
const MAX_RECORD_BYTES: u32 = 64 * 1024 * 1024;

/// The fsync policy. What an HTTP 200 means under each policy is defined in
/// DECISIONS #36.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// No fsync; durability only at checkpoints. The WAL is still written, so
    /// it survives a PROCESS crash — but not a MACHINE crash.
    None,
    /// fsync after every record. Safest, slowest.
    PerOp,
    /// Group commit: records within a window share a single fsync.
    /// The response still waits for that fsync (see DECISIONS #36).
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

/// A WAL record. There is NO checkpoint marker: because a checkpoint rotates
/// the WAL and opens a new file, the information "everything before this point
/// is in the segments" is carried by the file boundary itself.
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

/// The replay result — how many records were applied, where it stopped, why.
#[derive(Debug, Clone, Default)]
pub struct ReplayReport {
    pub applied: usize,
    pub bytes_ok: u64,
    /// Offset where the intact prefix ends; Some if the rest was discarded.
    pub truncated_at: Option<u64>,
    pub reason: Option<String>,
}

pub struct Wal {
    path: PathBuf,
    writer: BufWriter<File>,
    policy: SyncPolicy,
    bytes: u64,
    records: u64,
    /// Are there records written since the last fsync (not yet durable)?
    dirty: bool,
    last_sync: Instant,
}

impl Wal {
    /// Opens for appending to an existing file; creates it if absent.
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

    /// Appends a record to the log. Under `PerOp` the fsync happens here;
    /// under `Group` it happens if the window has elapsed; under `None` never.
    pub fn append(&mut self, rec: &WalRecord) -> Result<(), StorageError> {
        let payload = bincode::serialize(rec)?;
        if payload.len() as u64 > MAX_RECORD_BYTES as u64 {
            return Err(StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "record exceeds the size limit",
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
                // Only hand the buffer to the OS; the fsync is left to the checkpoint.
                self.writer.flush()?;
            }
        }
        Ok(())
    }

    /// Forces an fsync (closing the group window; graceful shutdown and
    /// checkpoint call this). Cheap if already clean.
    pub fn sync(&mut self) -> Result<(), StorageError> {
        self.writer.flush()?;
        if self.dirty {
            self.writer.get_ref().sync_data()?;
            self.dirty = false;
        }
        self.last_sync = Instant::now();
        Ok(())
    }

    /// fsyncs if the group window has elapsed. The writer task calls this at
    /// the end of a batch.
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

    /// Are there records that are not durable yet?
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// End-of-batch commit: provides exactly the durability the policy
    /// promises. Under `None` there is NO fsync (that is the contract; the data
    /// is only handed to the OS); under the others there is. The writer task
    /// calls this before sending responses — the "200 = fsynced" contract of
    /// group commit rests on it.
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

/// Reads the WAL file sequentially. Stops at the first inconsistency and
/// truncates the file at the end of the intact prefix. Returns an empty report
/// if the file does not exist (not an error).
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
        // Discard the corrupt/truncated tail so later appends continue cleanly.
        let f = OpenOptions::new().write(true).open(path)?;
        f.set_len(cut)?;
        f.sync_all()?;
    }
    Ok((records, report))
}

/// Replay from a byte buffer — tests and fuzzing share this path (no file IO).
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
            report.reason = stop("truncated record header");
            break;
        }
        let len = u32::from_le_bytes(bytes[off..off + 4].try_into().expect("4 byte"));
        let crc = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().expect("4 byte"));
        if len == 0 || len > MAX_RECORD_BYTES {
            report.reason = stop("implausible record length");
            break;
        }
        let end = off + 8 + len as usize;
        if end > bytes.len() {
            report.reason = stop("truncated record body");
            break;
        }
        let payload = &bytes[off + 8..end];
        if crate::storage::crc32(payload) != crc {
            report.reason = stop("crc mismatch");
            break;
        }
        match bincode::deserialize::<WalRecord>(payload) {
            Ok(rec) => {
                out.push(rec);
                report.applied += 1;
                off = end;
            }
            Err(_) => {
                // CRC matched but decoding failed: a format mismatch. Stop anyway.
                report.reason = stop("record could not be decoded");
                break;
            }
        }
    }
    report.bytes_ok = off as u64;
    report.truncated_at = Some(off as u64);
    (out, report)
}

/// Converts a record's metadata into the application type.
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
        // Cut in the MIDDLE of a record
        for cut in [good.len() - 1, good.len() - 5, good.len() / 2] {
            std::fs::write(&path, &good[..cut]).unwrap();
            let (recs, rep) = replay(&path).unwrap();
            assert!(
                recs.len() < 3,
                "a full record came out of a truncated file: {cut}"
            );
            assert!(rep.truncated_at.is_some());
            // the file must have been truncated at the intact prefix
            let after = std::fs::metadata(&path).unwrap().len();
            assert_eq!(after, rep.bytes_ok);
            // a second replay must give the same result and no longer truncate
            let (recs2, rep2) = replay(&path).unwrap();
            assert_eq!(recs2, recs);
            assert!(
                rep2.truncated_at.is_none(),
                "the second replay must be clean"
            );
        }
    }

    #[test]
    fn record_boundary_truncation_keeps_all_prior() {
        let dir = temp_dir("wal-boundary");
        let path = write_all(&dir, SyncPolicy::PerOp);
        let good = std::fs::read(&path).unwrap();
        // Cut exactly at the boundary of the first record
        let first_len = u32::from_le_bytes(good[0..4].try_into().unwrap()) as usize;
        let boundary = 8 + first_len;
        std::fs::write(&path, &good[..boundary]).unwrap();
        let (recs, rep) = replay(&path).unwrap();
        assert_eq!(
            recs.len(),
            1,
            "a cut at the boundary must preserve the full record"
        );
        assert!(
            rep.truncated_at.is_none(),
            "a boundary counts as a clean end"
        );
    }

    #[test]
    fn corrupt_payload_stops_without_ghost_ops() {
        let dir = temp_dir("wal-corrupt");
        let path = write_all(&dir, SyncPolicy::PerOp);
        let good = std::fs::read(&path).unwrap();
        // Corrupt the body of the second record
        let first_len = u32::from_le_bytes(good[0..4].try_into().unwrap()) as usize;
        let second_body = 8 + first_len + 8;
        let mut bad = good.clone();
        bad[second_body + 1] ^= 0xff;
        std::fs::write(&path, &bad).unwrap();
        let (recs, rep) = replay(&path).unwrap();
        assert_eq!(
            recs.len(),
            1,
            "nothing after a corrupt record may be applied"
        );
        assert_eq!(rep.reason.as_deref(), Some("crc mismatch"));
        assert_eq!(rep.truncated_at, Some(8 + first_len as u64));
    }

    #[test]
    fn garbage_length_header_is_rejected() {
        let dir = temp_dir("wal-garbage");
        let path = dir.join("w.log");
        // A gigantic length header: must be rejected without attempting to allocate
        let mut bytes = u32::MAX.to_le_bytes().to_vec();
        bytes.extend(0u32.to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();
        let (recs, rep) = replay(&path).unwrap();
        assert!(recs.is_empty());
        assert_eq!(rep.reason.as_deref(), Some("implausible record length"));
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
        assert!(SyncPolicy::parse("nonsense").is_none());
    }
}
