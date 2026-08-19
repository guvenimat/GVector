//! Segment-based concurrent index (phase 5) — the Lucene/Qdrant model.
//!
//! Structure:
//! - **Sealed segments**: read-only HNSW indexes (`Arc<Segment>`). Their
//!   contents never change; deletions are written to a segment-local tombstone
//!   set.
//! - **Write buffer**: a small brute-force index. Inserts go here; once the
//!   threshold is crossed the buffer is "sealed" into an HNSW and appended to
//!   the segment list.
//! - A search walks every segment plus the buffer and merges the results by id.
//!
//! Kilit disiplini (neden aramalar pratikte bloklanmaz):
//! - A reader holds the read lock on the segment list only long enough to clone
//!   the `Vec<Arc<Segment>>` (a few pointer copies) and then performs the HNSW
//!   search lock-free — the contents behind an Arc are immutable.
//! - The buffer search runs under a read lock, but the buffer is small
//!   (< threshold) and a brute-force scan takes microseconds; the writer's
//!   buffer write lock is as short as an O(1) append. The expensive work — HNSW
//!   construction (sealing) — is done holding NO lock at all; only appending the
//!   result to the list is locked.
//! - Sealing order: the segment is appended FIRST, the buffer is drained after.
//!   In between, a reader can see the same id from two sources; because merging
//!   deduplicates by id this is safe (a duplicate was preferred over a loss).
//!
//! The single-writer assumption: `insert`/`delete` take `&mut self` (that is
//! the VectorIndex contract anyway). Readers can search from any thread via
//! `&self`; since `SegmentedIndex: Sync`, an `Arc<SegmentedIndex>` plus one
//! writer thread suffices instead of `Arc<RwLock<...>>`. So that the writer can
//! work through `&self` as well, the mutations were written with internal locks
//! and are also exposed as `insert_shared`/`delete_shared` (the stress test uses
//! them).

use crate::distance::Metric;
use crate::index::bruteforce::BruteForceIndex;
use crate::index::hnsw::{HnswIndex, HnswParams};
use crate::index::numeric::NumericFieldIndex;
use crate::index::{IndexError, VectorIndex};
use crate::meta::{Filter, MetaKey, MetaStore, MetaValue, Metadata, Predicate};
use crate::storage::wal::{self, ReplayReport, SyncPolicy, Wal, WalRecord};
use crate::storage::{self, Manifest, SegmentRef, StorageError};
use crate::types::{SearchResult, VectorId};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

/// A segment's on-disk counterpart. The file name carries the generation it was
/// written in and is IMMUTABLE: a segment is written once, and later checkpoints
/// merely reference it from the manifest (see the storage module header).
#[derive(Debug, Clone)]
struct StoredFile {
    name: String,
    crc32: u32,
}

/// A sealed, immutable HNSW plus its own tombstone set.
///
/// Tombstones are segment-LOCAL: when an id is deleted and re-inserted, the old
/// copy stays shadowed in its own segment forever while the new copy lives in
/// the buffer (and later in another segment) — with a global deleted-set, a
/// re-insertion would resurrect the old copy.
/// A buffer that is BEING sealed (9a-2). A snapshotted, immutable brute-force
/// dataset plus its own tombstone set — that is, `Segment`'s sibling without the
/// HNSW. While the HNSW is built in the background, search, delete and the
/// duplicate-id check must all walk this source too (DECISIONS #50).
struct Sealing {
    data: BruteForceIndex,
    tombstones: RwLock<HashSet<VectorId>>,
}

struct Segment {
    index: HnswIndex,
    tombstones: RwLock<HashSet<VectorId>>,
    /// File name + CRC if written to disk; None for a freshly sealed segment or
    /// segmentlerde None (bir sonraki checkpoint yazar).
    stored: RwLock<Option<StoredFile>>,
}

pub struct SegmentedIndex {
    dim: usize,
    metric: Metric,
    /// HNSW parameters for sealed segments.
    hnsw_params: HnswParams,
    /// The buffer is sealed once it reaches this size. Atomic because
    /// measurements need to disable sealing temporarily (otherwise an "fsync
    /// policy" measurement silently turns into an "HNSW construction"
    /// measurement).
    seal_threshold: AtomicUsize,
    /// `Arc` so it can be cloned into the background merge thread (9a-1).
    /// Thanks to auto-deref, existing `self.segments.read()` calls were
    /// unaffected.
    segments: Arc<RwLock<Vec<Arc<Segment>>>>,
    /// Buffers currently being sealed (9a-2). Normally 0 or 1 elements; if the
    /// write rate outruns sealing they accumulate (pre-registration #49 measures
    /// exactly this).
    sealing: Arc<RwLock<Vec<Arc<Sealing>>>>,
    buffer: RwLock<BruteForceIndex>,
    /// Query width (on sealed segments).
    ef_search: usize,
    /// id → metadata. Kept apart from the vector data: segments are immutable,
    /// but metadata administration (deletion, re-insertion) flows at the id
    /// level.
    metadata: RwLock<MetaStore>,
    /// Eq posting lists: (field, value) → a SORTED list of live ids.
    ///
    /// 9c: a sorted `Vec` instead of `HashSet<VectorId>` (DECISIONS #64). In the
    /// 1M measurement, postings were the largest metadata item at 570 MB, and
    /// the bloat was in the `HashSet`s themselves: each set carries its own
    /// table, load-factor slack and header. A sorted `Vec` holds exactly 8 bytes
    /// per id; membership is O(log n) via binary search — and since the work in
    /// the Eq arm is walking the list end to end anyway, this loses nothing.
    /// The planner's O(1) cardinality estimate and the scan arm's id source.
    /// Maintained on insert/delete; Range predicates are out of scope
    /// (DECISIONS #28).
    postings: RwLock<HashMap<(String, MetaKey), Vec<VectorId>>>,
    /// The Range index for numeric fields: a histogram (the ŝ interval) plus a
    /// value-ordered map (bounded counting). See the numeric module /
    /// DECISIONS #31.
    numeric: RwLock<HashMap<String, NumericFieldIndex>>,
    /// Planner thresholds (query-planning parameters, not graph parameters).
    /// The values were derived from the selectivity measurements (BENCHMARKS,
    /// the filter sweep).
    planner: PlannerConfig,
    /// The ceiling guard: if the segment count exceeds this after sealing, the
    /// two SMALLEST segments are merged. The rationale is not a latency win (at
    /// equal recall a full merge buys ~20% — BENCHMARKS segcurve) but cutting
    /// off unbounded growth: the curve is linear (~+45µs/segment), so 40
    /// segments would be ~1.8ms. The two smallest: rebuild cost scales with n,
    /// so this is the cheapest merge, and sizes stay balanced (an oldest-first
    /// policy would pointlessly rebuild
    /// yeniden kurabilirdi).
    max_segments: usize,
    /// The persistence directory (if attached). None for in-memory use.
    storage_dir: RwLock<Option<PathBuf>>,
    /// A monotonic checkpoint counter; file-name uniqueness rests on it.
    generation: AtomicU64,
    /// Unix time of the last successful checkpoint (0 = never).
    last_checkpoint: AtomicU64,
    /// Hot durability (phase 7b). None = checkpoint durability only.
    wal: RwLock<Option<Wal>>,
    /// Duration of the last sealing and the last merge (µs) plus the merge
    /// count. For the 9a measurement: even once merging moves to its own task,
    /// SEALING stays on the writer, so the two windows must be known separately
    /// (to avoid charging one to the other).
    last_seal_us: AtomicU64,
    /// An Arc because the merge statistics are updated from the background
    /// thread.
    merge_stats: Arc<MergeStats>,
    /// Sealing statistics (9a-2: in the background) plus the in-flight count.
    seal_stats: Arc<MergeStats>,
    seal_in_flight: Arc<AtomicUsize>,
    /// Backpressure counters (#53): how many inserts were stalled and for how
    /// many µs in total. For observability: a stall is a "silent" slowdown, and
    /// what cannot be measured cannot be diagnosed.
    stall_count: AtomicU64,
    stall_us: AtomicU64,
    /// At most one merge at a time; if the ceiling is exceeded again the worker
    /// loop continues (queueing) — no new thread is spawned.
    merge_in_flight: Arc<AtomicBool>,
    /// Report of the WAL replay performed at startup (observability / /stats).
    replay_report: RwLock<ReplayReport>,
}

/// Planner configuration. The values were derived from the 10K + 100K
/// selectivity sweeps (BENCHMARKS).
///
/// Why in-traversal filtering left the production path: the 100K measurement
/// showed that with clustered matches and a distant query, in-traversal
/// filtering spreads across the entire graph (up to 35ms) AND that a silent
/// recall decline sets in with scale (0.948). Unfiltered traversal is
/// structurally immune to that pathology: the traversal never looks at the
/// filter and walks the same ~µs path; the filter is applied to the results via
/// over-fetch.
#[derive(Debug, Clone)]
pub struct PlannerConfig {
    /// est ≤ scan_factor·k → the scan arm (a small absolute match count).
    pub scan_factor: usize,
    /// est ≤ scan_fraction·n → the scan arm. 0.05: below this band the expected
    /// match count of over-fetch cannot guarantee k, whereas the cost of a scan
    /// is bounded by est and predictable at every query position.
    pub scan_fraction: f64,
    /// Over-fetch in the post-filter arm: ef'' = overfetch_beta·k/ŝ.
    /// β=5: the expected match count is 5k. With β=3 the NUMBER of results was
    /// sufficient, but for a mid-band clustered query part of the true top-k
    /// fell outside the window and recall dropped to 0.979 (the 10K
    /// measurement); β=5 widens the window for quality.
    pub overfetch_beta: f64,
    /// Upper cap on ef'' = overfetch_cap_factor·ef (estimation error enters
    /// ef'' as a multiplier; the cap bounds it — from user feedback).
    pub overfetch_cap_factor: usize,
}

impl Default for PlannerConfig {
    fn default() -> Self {
        Self {
            scan_factor: 16,
            scan_fraction: 0.05,
            overfetch_beta: 5.0,
            overfetch_cap_factor: 8,
        }
    }
}

/// The thread-movable form of the merge trigger: when the sealing worker
/// finishes it runs the ceiling guard through this (`&self` cannot be moved).
#[derive(Clone)]
struct MergeContext {
    segments: Arc<RwLock<Vec<Arc<Segment>>>>,
    stats: Arc<MergeStats>,
    in_flight: Arc<AtomicBool>,
    dim: usize,
    metric: Metric,
    max_segments: usize,
    params: HnswParams,
}

impl MergeContext {
    /// Starts a background merge if the ceiling is exceeded. At most one merge
    /// at a time; if the ceiling is exceeded again the worker loop continues.
    fn spawn_if_needed(&self) {
        if self.segments.read().expect("kilit").len() <= self.max_segments {
            return;
        }
        if self.in_flight.swap(true, Ordering::SeqCst) {
            return; // the running worker will see the new state
        }
        let ctx = self.clone();
        std::thread::spawn(move || loop {
            while ctx.segments.read().expect("kilit").len() > ctx.max_segments {
                let t = std::time::Instant::now();
                merge_smallest_pair_bg(&ctx.segments, ctx.dim, ctx.metric, &ctx.params);
                ctx.stats
                    .last_us
                    .store(t.elapsed().as_micros() as u64, Ordering::Relaxed);
                ctx.stats.count.fetch_add(1, Ordering::Relaxed);
            }
            ctx.in_flight.store(false, Ordering::SeqCst);
            // A new segment may have been added while releasing the flag:
            // double-check.
            if ctx.segments.read().expect("kilit").len() <= ctx.max_segments
                || ctx.in_flight.swap(true, Ordering::SeqCst)
            {
                break;
            }
        });
    }
}

/// The movable context of the sealing worker (#53).
///
/// WHY A SINGLE WORKER: in the first 9a-2 design, `seal()` called
/// `thread::spawn` on every invocation. In the 1M accumulation measurement, 35
/// sealings ran concurrently within 60 seconds and, sharing 8 cores, NONE of
/// them could finish (the segment count stayed at 0), memory rose to 2.3 GB and
/// the write rate collapsed from 273K to 11.7K op/s (BENCHMARKS 9a-2,
/// DECISIONS #52). Concurrent construction does not reduce the total work, it
/// only slows all of it down; a sequential single worker does the same work in
/// the same time, but each sealing finishes IN TURN, so the queue is drained and
/// memory is returned.
///
/// The `sealing` list is the queue itself: the worker always processes the FIRST
/// (oldest) element — order is preserved, and since the search/delete paths
/// already walk that list, no separate queue structure is needed.
#[derive(Clone)]
struct SealContext {
    segments: Arc<RwLock<Vec<Arc<Segment>>>>,
    sealing: Arc<RwLock<Vec<Arc<Sealing>>>>,
    stats: Arc<MergeStats>,
    in_flight: Arc<AtomicUsize>,
    dim: usize,
    metric: Metric,
    params: HnswParams,
    merge: MergeContext,
}

impl SealContext {
    /// Starts the worker if it is not running. Does nothing if it is: the worker
    /// loop continues until the queue drains (the pattern used for merging).
    fn spawn_if_needed(&self) {
        if self.in_flight.swap(1, Ordering::SeqCst) == 1 {
            return; // the running worker will see the new element
        }
        let ctx = self.clone();
        std::thread::spawn(move || loop {
            // LOCK LIFETIME: the next element is taken in its own expression so
            // the guard drops at the end of that line. `while let Some(x) =
            // sealing.read()
            // ...first().cloned()` YAZILMAZ: `while let`, `loop { match EXPR
            // { ... } }` olarak desugar edilir ve match scrutinee'sinin
            // temporaries live for the whole block INCLUDING THE BODY — so the
            // read
            // kilidi tutulurken `build_one` write kilidi ister ve worker
            // deadlocks itself. (Rust 2024 fixed this for `if let`, but
            // `while let` still has the old behaviour; upgrading the edition
            // does not save you.) A plain `while COND` is safe: the condition's
            // temporaries drop as soon as it is evaluated — which is why the
            // merge worker was sound.
            loop {
                let next = ctx.sealing.read().expect("kilit").first().cloned();
                let Some(next) = next else { break };
                ctx.build_one(&next);
            }
            ctx.in_flight.store(0, Ordering::SeqCst);
            // A new sealing may have entered the queue while releasing the flag.
            if ctx.sealing.read().expect("kilit").is_empty()
                || ctx.in_flight.swap(1, Ordering::SeqCst) == 1
            {
                break;
            }
        });
    }

    /// Completes one sealing: builds the HNSW WITHOUT HOLDING A LOCK, then moves
    /// it into the segments under a single write lock and carries over, via
    /// diff-replay, the tombstones that landed during construction.
    fn build_one(&self, sealing: &Arc<Sealing>) {
        let t = std::time::Instant::now();
        let mut p = self.params.clone();
        p.seed = p.seed.wrapping_add(sealing.data.len() as u64);
        let mut hnsw = HnswIndex::new(self.dim, self.metric, p);
        for (id, v) in sealing.data.entries() {
            hnsw.insert(id, v).expect("a sealing insert cannot fail");
        }
        let built = Arc::new(Segment {
            index: hnsw,
            tombstones: RwLock::new(HashSet::new()),
            stored: RwLock::new(None),
        });
        {
            let mut segs = self.segments.write().expect("kilit");
            let mut seal_list = self.sealing.write().expect("kilit");
            let carried = sealing.tombstones.read().expect("kilit").clone();
            *built.tombstones.write().expect("kilit") = carried;
            segs.push(Arc::clone(&built));
            seal_list.retain(|s| !Arc::ptr_eq(s, sealing));
        }
        self.stats
            .last_us
            .store(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        self.stats.count.fetch_add(1, Ordering::Relaxed);
        // Ceiling guard: a new segment was added, trigger a merge if needed.
        self.merge.spawn_if_needed();
    }
}

/// The number of records to pre-allocate for the buffer.
///
/// Allocating up to the threshold is the right behaviour (#61: it removes the
/// realloc spike), but the threshold is also used as `usize::MAX` to mean "no
/// sealing in practice" — and allocating the raw threshold in that case PANICS
/// with a capacity overflow. So the allocation is bounded in bytes: above that
/// bound `Vec` falls back to its old incremental growth (the realloc spike
/// returns only in that extreme case, which is measurement/test usage).
fn prealloc_records(seal_threshold: usize, dim: usize) -> usize {
    const MAX_PREALLOC_BYTES: usize = 512 << 20;
    let per_record = dim.max(1) * std::mem::size_of::<f32>();
    seal_threshold.min(MAX_PREALLOC_BYTES / per_record)
}

/// Inserts into a sorted posting list (a no-op if already present).
///
/// `binary_search` gives the position and `insert` shifts from there. The shift
/// is O(n), but posting lists are updated only a few times per record on the
/// insert path and a `Vec` shift is a memmove over contiguous memory — in the
/// measurements the insert cost did not change appreciably (BENCHMARKS 9c).
fn posting_insert(list: &mut Vec<VectorId>, id: VectorId) {
    if let Err(pos) = list.binary_search(&id) {
        list.insert(pos, id);
    }
}

/// Removes from a sorted posting list (a no-op if absent).
fn posting_remove(list: &mut Vec<VectorId>, id: VectorId) {
    if let Ok(pos) = list.binary_search(&id) {
        list.remove(pos);
    }
}

/// Merge statistics (updated by the background thread, observed by readers).
#[derive(Debug, Default)]
struct MergeStats {
    last_us: AtomicU64,
    count: AtomicU64,
}

/// Rebuilds the two smallest segments (by live count) into a single segment.
/// Runs **on a background thread** (9a-1) — it takes the shared segment list
/// rather than `&self`, so the writer task is never blocked.
///
/// THE CRITICAL RACE (the real difficulty of this function): a sealed segment
/// accepts no inserts but it does accept **deletes**. While the rebuild runs for
/// seconds, new tombstones can land on the source segments; because the rebuild
/// does not see them, those records would be copied into the merged segment as
/// LIVE. At swap time, under the write lock, the DIFFERENCE between the sources'
/// CURRENT tombstones and the snapshot taken at the start of the rebuild is
/// applied to the merged segment (diff-replay). Skip this step and deleted
/// records silently come back — and no latency measurement would ever catch
/// it.
///
/// Kilit disiplini: `delete_vector_only` tombstone'u `segments` READ kilidini
/// while holding it; the swap takes the `segments` WRITE lock. Because the two
/// are mutually exclusive, the interleaving "I read the diff, then a new
/// tombstone arrived, then
/// takas ettim" penceresi YOKTUR.
fn merge_smallest_pair_bg(
    segments: &Arc<RwLock<Vec<Arc<Segment>>>>,
    dim: usize,
    metric: Metric,
    hnsw_params: &HnswParams,
) {
    // 1. Pick the victims and take a SNAPSHOT of their tombstones (short read
    //    lock).
    let (a, b, a_snap, b_snap) = {
        let segs = segments.read().expect("kilit");
        if segs.len() < 2 {
            return;
        }
        let live = |s: &Arc<Segment>| s.index.len() - s.tombstones.read().expect("kilit").len();
        let mut order: Vec<usize> = (0..segs.len()).collect();
        order.sort_by_key(|&i| live(&segs[i]));
        let a = segs[order[0]].clone();
        let b = segs[order[1]].clone();
        let a_snap = a.tombstones.read().expect("kilit").clone();
        let b_snap = b.tombstones.read().expect("kilit").clone();
        (a, b, a_snap, b_snap)
    };

    // 2. Lock-free rebuild (the real cost; readers and the writer keep running).
    //    Records tombstoned in the snapshot are not carried over.
    let mut params = hnsw_params.clone();
    let total = a.index.len() + b.index.len();
    params.seed = params.seed.wrapping_add(total as u64).wrapping_add(1);
    let mut merged = HnswIndex::new(dim, metric, params);
    for (seg, snap) in [(&a, &a_snap), (&b, &b_snap)] {
        for (id, v) in seg.index.live_entries() {
            if !snap.contains(&id) {
                merged.insert(id, v).expect("a merge insert cannot fail");
            }
        }
    }

    // 3. Atomic swap + diff-replay, under a SINGLE write lock.
    let mut segs = segments.write().expect("kilit");
    // Are the sources still in the list? (Another merge may have taken them.)
    if !segs.iter().any(|s| Arc::ptr_eq(s, &a)) || !segs.iter().any(|s| Arc::ptr_eq(s, &b)) {
        return; // swap cancelled: the rebuild was wasted but consistency holds
    }
    // NEW tombstones that landed during the rebuild are carried to the merged
    // segment.
    let mut carried: HashSet<VectorId> = HashSet::new();
    for (seg, snap) in [(&a, &a_snap), (&b, &b_snap)] {
        let now = seg.tombstones.read().expect("kilit");
        carried.extend(now.difference(snap).copied());
    }
    let merged = Arc::new(Segment {
        index: merged,
        tombstones: RwLock::new(carried),
        stored: RwLock::new(None), // the merged segment will go to a new file
    });
    segs.retain(|s| !Arc::ptr_eq(s, &a) && !Arc::ptr_eq(s, &b));
    segs.push(merged);
}

/// The numeric projection of a MetaValue (those that enter the Range index).
fn numeric_value(v: &MetaValue) -> Option<f64> {
    match v {
        MetaValue::Int(i) => Some(*i as f64),
        MetaValue::Float(f) => Some(*f),
        _ => None,
    }
}

/// The planner's arm decision. `Scan` carries the ids along (they come out for
/// free); `Post` carries its fallback source so an exact count can be done in
/// the <2k case.
enum Arm {
    /// One predicate matches exactly zero — the result is empty.
    Empty,
    /// A definitively small match set: scan directly.
    Scan(Vec<VectorId>),
    /// Unfiltered traversal + over-fetch; ŝ from the upper-bound estimate.
    Post {
        s_hat: f64,
        fallback: FallbackSource,
    },
    /// No estimate (no Eq and no numeric index): in-traversal filtering.
    Legacy,
}

enum FallbackSource {
    Ids(Vec<VectorId>),
    Range { key: String, lo: f64, hi: f64 },
}

impl SegmentedIndex {
    pub fn new(dim: usize, metric: Metric, hnsw_params: HnswParams, seal_threshold: usize) -> Self {
        let ef_search = hnsw_params.ef_search;
        Self {
            dim,
            metric,
            hnsw_params,
            seal_threshold: AtomicUsize::new(seal_threshold),
            segments: Arc::new(RwLock::new(Vec::new())),
            sealing: Arc::new(RwLock::new(Vec::new())),
            buffer: RwLock::new(BruteForceIndex::with_capacity(
                dim,
                metric,
                prealloc_records(seal_threshold, dim),
            )),
            ef_search,
            metadata: RwLock::new(MetaStore::new()),
            postings: RwLock::new(HashMap::new()),
            numeric: RwLock::new(HashMap::new()),
            planner: PlannerConfig::default(),
            max_segments: 8,
            storage_dir: RwLock::new(None),
            generation: AtomicU64::new(0),
            last_checkpoint: AtomicU64::new(0),
            wal: RwLock::new(None),
            last_seal_us: AtomicU64::new(0),
            merge_stats: Arc::new(MergeStats::default()),
            seal_stats: Arc::new(MergeStats::default()),
            seal_in_flight: Arc::new(AtomicUsize::new(0)),
            stall_count: AtomicU64::new(0),
            stall_us: AtomicU64::new(0),
            merge_in_flight: Arc::new(AtomicBool::new(false)),
            replay_report: RwLock::new(ReplayReport::default()),
        }
    }

    /// Changes the segment ceiling (for tests/experiments).
    pub fn set_max_segments(&mut self, max: usize) {
        self.max_segments = max.max(2);
    }

    /// Changes the sealing threshold. Used to disable sealing during
    /// measurements (usize::MAX): otherwise an "fsync policy" measurement
    /// silently measures something else because an HNSW build cuts in.
    pub fn set_seal_threshold(&self, n: usize) {
        self.seal_threshold.store(n.max(1), Ordering::Relaxed);
    }

    pub fn seal_threshold(&self) -> usize {
        self.seal_threshold.load(Ordering::Relaxed)
    }

    /// Insert with metadata. The metadata-free `insert_shared` passes an empty
    /// map.
    pub fn insert_with_meta(
        &self,
        id: VectorId,
        vector: &[f32],
        meta: Metadata,
    ) -> Result<(), IndexError> {
        // Write-ahead ordering (DECISIONS #36): (1) validation — no mutation,
        // (2) WAL append + fsync per the policy, (3) apply to memory. In the
        // reverse order you would get "we returned an error to the client but
        // the record stayed in memory and the next checkpoint made it
        // permanent".
        self.validate_insert(id, vector)?;
        if let Some(w) = self.wal.write().expect("kilit").as_mut() {
            w.append(&WalRecord::insert(id, vector, &meta))
                .map_err(|e| IndexError::Storage(e.to_string()))?;
        }
        self.apply_insert(id, vector, meta)
    }

    /// The in-memory side of an insert (no WAL). Replay uses this path — since
    /// the WAL is not attached during replay, records are not duplicated.
    fn apply_insert(&self, id: VectorId, vector: &[f32], meta: Metadata) -> Result<(), IndexError> {
        let should_seal = {
            let mut buffer = self.buffer.write().expect("kilit");
            buffer.insert(id, vector)?;
            buffer.len() >= self.seal_threshold.load(Ordering::Relaxed)
        }; // the write lock drops; sealing will run lock-free
        if !meta.is_empty() {
            self.index_metadata(id, meta);
        }
        if should_seal {
            self.seal(); // the ceiling guard lives inside seal()
                         // If the sealing queue is accumulating, slow the writer
                         // down here (#53). AFTER sealing: the wait only kicks in
                         // when the queue grows, not on every insert.
            self.apply_backpressure();
        }
        Ok(())
    }

    /// Insert validation: dimension and id collision. Mutates nothing — kept
    /// separate so it can be called before writing to the WAL.
    fn validate_insert(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        if vector.len() != self.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if self.buffer.read().expect("kilit").contains(id) {
            return Err(IndexError::DuplicateId(id));
        }
        // The 9a-2 "two buffers" risk (DECISIONS #50): if an id living in the
        // buffer being sealed were inserted a second time into the new buffer,
        // the collision would only surface once sealing finished — and by then
        // both copies would be permanent. That is why the duplicate check walks
        // THAT source too.
        if self.sealing_contains_live(id) {
            return Err(IndexError::DuplicateId(id));
        }
        let segments = self.segments.read().expect("kilit");
        for seg in segments.iter() {
            if seg.index.contains(id) && !seg.tombstones.read().expect("kilit").contains(&id) {
                return Err(IndexError::DuplicateId(id));
            }
        }
        Ok(())
    }

    /// Applies metadata to the store and to the derived indexes (posting lists,
    /// numeric fields). The insert path and the rebuild-from-snapshot path share
    /// this: since derived structures are never written to disk, this is their
    /// single source.
    fn index_metadata(&self, id: VectorId, meta: Metadata) {
        let mut postings = self.postings.write().expect("kilit");
        for (key, value) in &meta {
            posting_insert(postings.entry((key.clone(), value.key())).or_default(), id);
        }
        drop(postings);
        // Numeric values also enter the Range index (Int/Float).
        let mut numeric = self.numeric.write().expect("kilit");
        for (key, value) in &meta {
            if let Some(v) = numeric_value(value) {
                numeric.entry(key.clone()).or_default().insert(v, id);
            }
        }
        drop(numeric);
        self.metadata.write().expect("kilit").insert(id, meta);
    }

    /// Cardinality estimate: the minimum of the posting counts of the Eq
    /// predicates (an upper bound for an AND conjunction — the intersection can
    /// be smaller, never larger). None when there is no Eq predicate (we keep no
    /// histogram for Range here).
    /// The returned set: the smallest posting list (the candidate source of the
    /// scan arm).
    fn estimate(&self, filter: &Filter) -> Option<(usize, Vec<VectorId>)> {
        let keys = filter.eq_keys();
        if keys.is_empty() {
            return None;
        }
        let postings = self.postings.read().expect("kilit");
        let mut best: Option<&Vec<VectorId>> = None;
        for (k, mk) in keys {
            match postings.get(&(k.to_string(), mk)) {
                // If any Eq predicate has no matches at all, the result is empty.
                None => return Some((0, Vec::new())),
                Some(list) => {
                    if best.is_none_or(|b| list.len() < b.len()) {
                        best = Some(list);
                    }
                }
            }
        }
        best.map(|s| (s.len(), s.clone()))
    }

    /// The arm decision (DECISIONS #29 + #31). The interval estimate is used
    /// conservatively:
    /// - The small-arm decision is never made from an estimate: for Eq the count
    ///   is already exact, and for Range a bounded count (`enumerate_up_to`)
    ///   makes it exact. The bounded count is only attempted while the lower
    ///   bound ≤ limit (if even the lower bound is large, the truth is certainly
    ///   large and counting would be wasted).
    /// - ŝ for the large arm is the minimum of the UPPER bounds (the Fréchet
    ///   upper bound for an AND conjunction). An upper bound errs toward a small
    ///   ŝ and hence a large ef''; when wrong, the price is latency rather than
    ///   recall (and the <2k fallback exists anyway).
    fn plan(&self, filter: &Filter, k: usize) -> Arm {
        let n = self.len_shared().max(1);
        let scan_limit =
            (self.planner.scan_factor * k).max((self.planner.scan_fraction * n as f64) as usize);
        let eq = self.estimate(filter);
        if let Some((0, _)) = eq {
            return Arm::Empty;
        }
        let mut best_upper: Option<usize> = eq.as_ref().map(|(e, _)| *e);
        let mut best_small: Option<Vec<VectorId>> = eq
            .as_ref()
            .filter(|(e, _)| *e <= scan_limit)
            .map(|(_, list)| list.clone());
        let mut range_fallback: Option<(String, f64, f64, usize)> = None;
        {
            let numeric = self.numeric.read().expect("kilit");
            for p in &filter.must {
                if let Predicate::Range { key, min, max } = p {
                    if let Some(fi) = numeric.get(key) {
                        let (lower, upper) = fi.estimate(*min, *max);
                        if upper == 0 {
                            return Arm::Empty;
                        }
                        if best_upper.is_none_or(|b| upper < b) {
                            best_upper = Some(upper);
                        }
                        if range_fallback
                            .as_ref()
                            .is_none_or(|(_, _, _, u)| upper < *u)
                        {
                            range_fallback = Some((key.clone(), *min, *max, upper));
                        }
                        if best_small.is_none() && lower <= scan_limit {
                            if let Some(ids) = fi.enumerate_up_to(*min, *max, scan_limit) {
                                best_small = Some(ids);
                            }
                        }
                    }
                }
            }
        }
        if let Some(ids) = best_small {
            return Arm::Scan(ids);
        }
        match best_upper {
            Some(upper) => Arm::Post {
                s_hat: (upper as f64 / n as f64).clamp(1e-6, 1.0),
                fallback: match eq {
                    Some((_, set)) => FallbackSource::Ids(set),
                    None => {
                        let (key, lo, hi, _) =
                            range_fallback.expect("if upper exists so does the range source");
                        FallbackSource::Range { key, lo, hi }
                    }
                },
            },
            None => Arm::Legacy,
        }
    }

    /// For measurement/tests: the name of the arm the planner chose.
    pub fn debug_plan_arm(&self, filter: &Filter, k: usize) -> &'static str {
        match self.plan(filter, k) {
            Arm::Empty => "empty",
            Arm::Scan(_) => "scan",
            Arm::Post { .. } => "post",
            Arm::Legacy => "legacy",
        }
    }

    /// For measurement: the estimated cardinality interval of a numeric field
    /// over [lo, hi].
    pub fn debug_range_estimate(&self, key: &str, lo: f64, hi: f64) -> (usize, usize) {
        self.numeric
            .read()
            .expect("kilit")
            .get(key)
            .map(|fi| fi.estimate(lo, hi))
            .unwrap_or((0, 0))
    }

    /// The scan arm: the full filter plus direct distance computation over the
    /// candidate ids (the smallest posting list). Exact — the graph is never
    /// opened.
    fn scan_candidates(
        &self,
        query: &[f32],
        k: usize,
        candidates: &[VectorId],
        filter: &Filter,
    ) -> Vec<SearchResult> {
        let normalized_query;
        let query: &[f32] = if self.metric.requires_normalization() {
            normalized_query = crate::distance::normalized(query);
            &normalized_query
        } else {
            query
        };
        let meta = self.metadata.read().expect("kilit");
        let buffer = self.buffer.read().expect("kilit");
        let segments: Vec<Arc<Segment>> = self
            .segments
            .read()
            .expect("kilit")
            .iter()
            .cloned()
            .collect();
        // Cost notes (the first version was ~4x a brute-force scan):
        // - For a single-Eq filter the posting list is ALREADY the exact match
        //   set: the per-id metadata map lookup is skipped.
        // - Source-outer loop: instead of "try every source" per id, the
        //   remaining ids are eliminated per source — sequential access to the
        //   same map,
        //   bulunan id bir daha denenmez.
        // - A top-k heap: O(est·log k), rather than sorting the whole list.
        let mut heap: std::collections::BinaryHeap<SearchResult> =
            std::collections::BinaryHeap::with_capacity(k + 1);
        let push = |id: VectorId, d: f32, heap: &mut std::collections::BinaryHeap<SearchResult>| {
            let cand = SearchResult::new(id, d);
            if heap.len() < k {
                heap.push(cand);
            } else if let Some(worst) = heap.peek() {
                if cand < *worst {
                    heap.pop();
                    heap.push(cand);
                }
            }
        };
        let mut remaining: Vec<VectorId> = if filter.must.len() > 1 {
            candidates
                .iter()
                .copied()
                .filter(|&id| filter.matches_id(&meta, id))
                .collect()
        } else {
            candidates.to_vec()
        };
        // Newest source to oldest: the live copy is always in the newest place
        // (sil→yeniden-ekle zinciri buffer'a, oradan daha yeni segmente gider).
        remaining.retain(|&id| {
            if let Some(v) = buffer.vector_of(id) {
                push(id, self.metric.distance(query, v), &mut heap);
                false
            } else {
                true
            }
        });
        // 9a-2: the buffers being sealed (newest to oldest)
        for sl in self.sealing_snapshot().iter().rev() {
            if remaining.is_empty() {
                break;
            }
            let tombs = sl.tombstones.read().expect("kilit");
            remaining.retain(|&id| {
                if let Some(v) = sl.data.vector_of(id) {
                    if !tombs.contains(&id) {
                        push(id, self.metric.distance(query, v), &mut heap);
                    }
                    false
                } else {
                    true
                }
            });
        }
        for seg in segments.iter().rev() {
            if remaining.is_empty() {
                break;
            }
            let tombs = seg.tombstones.read().expect("kilit");
            remaining.retain(|&id| {
                if let Some(v) = seg.index.vector_of(id) {
                    if !tombs.contains(&id) {
                        push(id, self.metric.distance(query, v), &mut heap);
                    }
                    false // found in this segment (live or shadowed) — search ends
                } else {
                    true
                }
            });
        }
        let mut out = heap.into_vec();
        out.sort();
        out
    }

    /// Filtered search — a three-arm planner (rationale: the BENCHMARKS filter
    /// sweep; DECISIONS #28–29):
    /// 1. If the Eq estimate is small (≤ max(scan_factor·k, scan_fraction·n)):
    ///    scan the posting list directly without opening the graph — exact and
    ///    cheap.
    /// 2. Otherwise UNFILTERED traversal + over-fetch (ef'' = β·k/ŝ, capped),
    ///    with the filter applied to the results; if fewer than 2k remain it
    ///    falls back to an exact scan. Unfiltered traversal is structurally
    ///    immune to the clustered-match pathology of in-traversal filtering
    ///    (walking the entire graph).
    /// 3. If there is no Eq predicate (no estimate), in-traversal filtering plus
    ///    the found<k safety net — the only option.
    pub fn search_filtered(&self, query: &[f32], k: usize, filter: &Filter) -> Vec<SearchResult> {
        if filter.must.is_empty() {
            return self.search_shared(query, k);
        }
        if k == 0 {
            return Vec::new();
        }
        // Shortcut: a single Eq that every live record matches → the filter is
        // behaviourally empty and the unfiltered path is exactly equivalent (the
        // default-filter case coming from a UI).
        if filter.must.len() == 1 {
            if let Some((est, _)) = self.estimate(filter) {
                if est >= self.len_shared().max(1) {
                    return self.search_shared(query, k);
                }
            }
        }
        match self.plan(filter, k) {
            Arm::Empty => Vec::new(),
            Arm::Scan(candidates) => self.scan_candidates(query, k, &candidates, filter),
            Arm::Post { s_hat, fallback } => {
                // The post-filter arm: UNFILTERED traversal (immune to the
                // pathology) + over-fetch + filtering of the results. ŝ is an
                // upper bound; if the true selectivity is small, fewer than 2k
                // results remain and the exact fallback runs.
                let ef_over = ((self.planner.overfetch_beta * k as f64 / s_hat) as usize).clamp(
                    self.ef_search.max(2 * k),
                    (self.planner.overfetch_cap_factor * self.ef_search).max(4 * k),
                );
                let mut all: Vec<SearchResult> = Vec::new();
                {
                    let meta = self.metadata.read().expect("kilit");
                    let allow = |id: VectorId| filter.matches_id(&meta, id);
                    let segments: Vec<Arc<Segment>> = self
                        .segments
                        .read()
                        .expect("kilit")
                        .iter()
                        .cloned()
                        .collect();
                    for seg in &segments {
                        let tombs = seg.tombstones.read().expect("kilit");
                        let res = seg.index.search_with_ef(query, ef_over, ef_over);
                        all.extend(
                            res.into_iter()
                                .filter(|r| !tombs.contains(&r.id) && allow(r.id)),
                        );
                    }
                    // 9a-2: the buffers being sealed
                    for sl in self.sealing_snapshot() {
                        let tombs = sl.tombstones.read().expect("kilit");
                        let allow_live = |id: VectorId| !tombs.contains(&id) && allow(id);
                        all.extend(sl.data.search_filtered(query, k, &allow_live));
                    }
                    let buffer = self.buffer.read().expect("kilit");
                    all.extend(buffer.search_filtered(query, k, &allow));
                } // the locks drop — the fallback scan_candidates re-acquires them
                all.sort();
                let mut seen = HashSet::with_capacity(all.len());
                all.retain(|r| seen.insert(r.id));
                if all.len() < 2 * k {
                    // Either the over-fetch window missed the match region (the
                    // query is far from the matches) or the upper-bound estimate
                    // was inflated (AND correlation): fall back to an exact scan.
                    // The fallback candidates come from the source — an Eq
                    // posting list or a full Range enumeration.
                    let candidates = match fallback {
                        FallbackSource::Ids(ids) => ids,
                        FallbackSource::Range { key, lo, hi } => self
                            .numeric
                            .read()
                            .expect("kilit")
                            .get(&key)
                            .map(|fi| fi.enumerate_all(lo, hi))
                            .unwrap_or_default(),
                    };
                    return self.scan_candidates(query, k, &candidates, filter);
                }
                all.truncate(k);
                all
            }
            // No estimate (no Eq, and a Range without a numeric index):
            // in-traversal filtering plus the found<k safety net is the only
            // option.
            Arm::Legacy => {
                let meta = self.metadata.read().expect("kilit");
                let allow = |id: VectorId| filter.matches_id(&meta, id);
                let segments: Vec<Arc<Segment>> = self
                    .segments
                    .read()
                    .expect("kilit")
                    .iter()
                    .cloned()
                    .collect();
                let mut all: Vec<SearchResult> = Vec::new();
                for seg in &segments {
                    let tombs = seg.tombstones.read().expect("kilit");
                    let allow_live = |id: VectorId| !tombs.contains(&id) && allow(id);
                    let want = k + tombs.len().min(k);
                    all.extend(seg.index.search_filtered_with_ef(
                        query,
                        want,
                        self.ef_search.max(want),
                        &allow_live,
                    ));
                }
                {
                    for sl in self.sealing_snapshot() {
                        let tombs = sl.tombstones.read().expect("kilit");
                        let allow_live = |id: VectorId| !tombs.contains(&id) && allow(id);
                        all.extend(sl.data.search_filtered(query, k, &allow_live));
                    }
                    let buffer = self.buffer.read().expect("kilit");
                    all.extend(buffer.search_filtered(query, k, &allow));
                }
                all.sort();
                let mut seen = HashSet::with_capacity(all.len());
                all.retain(|r| seen.insert(r.id));
                all.truncate(k);
                all
            }
        }
    }

    /// Shared (&self) insert — must be called from the single writer thread.
    /// Multiple writers would not cause a data race (everything is locked), but
    /// the duplicate-id check could race between two writers; the contract is a
    /// single writer.
    pub fn insert_shared(&self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        self.insert_with_meta(id, vector, Metadata::new())
    }

    /// Sealing (9a-2: in the background).
    ///
    /// Converts the buffer into an HNSW segment. The expensive build runs
    /// lock-free; throughout it readers keep seeing the old segments plus the
    /// full buffer (no vector ever becomes invisible).
    ///
    /// ONLY this happens on the writer task: swap the buffer with a new empty
    /// takas et ve eskisini `sealing` listesine koy (mikrosaniyeler).
    /// The expensive part, the HNSW build, goes to a background thread.
    ///
    /// From this moment the "two buffers" state exists: the snapshot being
    /// sealed plus the new write buffer. Search, delete and the duplicate-id
    /// check all walk the three sources (DECISIONS #50).
    fn seal(&self) {
        let t_seal = std::time::Instant::now();
        // 1. Atomic swap: take the buffer, put an empty one in its place. Under
        //    the single-writer contract no other insert can happen meanwhile.
        let sealed_data = {
            let mut buffer = self.buffer.write().expect("kilit");
            if buffer.is_empty() {
                return;
            }
            // The new buffer is pre-sized too: otherwise the realloc ladder
            // would start over after every sealing (#61).
            let fresh = BruteForceIndex::with_capacity(
                self.dim,
                self.metric,
                prealloc_records(self.seal_threshold.load(Ordering::Relaxed), self.dim),
            );
            std::mem::replace(&mut *buffer, fresh)
        };
        let sealing = Arc::new(Sealing {
            data: sealed_data,
            tombstones: RwLock::new(HashSet::new()),
        });
        // 2. Publish: search will now walk this source too, so the data is never
        //    invisible
        //    olmuyor.
        self.sealing
            .write()
            .expect("kilit")
            .push(Arc::clone(&sealing));
        self.last_seal_us
            .store(t_seal.elapsed().as_micros() as u64, Ordering::Relaxed);

        // 3. A SINGLE worker drains the queue (#53). No new thread is spawned
        //    here: unbounded spawning was what left 35 concurrent sealings none
        //    of which could finish (#52).
        self.seal_context().spawn_if_needed();
    }

    /// The movable context of the sealing worker.
    fn seal_context(&self) -> SealContext {
        SealContext {
            segments: Arc::clone(&self.segments),
            sealing: Arc::clone(&self.sealing),
            stats: Arc::clone(&self.seal_stats),
            in_flight: Arc::clone(&self.seal_in_flight),
            dim: self.dim,
            metric: self.metric,
            params: self.hnsw_params.clone(),
            merge: self.merge_context(),
        }
    }

    /// Backpressure (#53, in the form pre-registration #49 anticipated).
    ///
    /// Writes are NOT REJECTED, they are slowed down (Lucene's `IndexWriter`
    /// stall): once the queue threshold is crossed, the writer waits in 1 ms
    /// sleeps. The single-writer contract is preserved; the wait holds no lock,
    /// so readers are unaffected.
    ///
    /// WHY THE SIGNAL IS THE QUEUE ALONE (the first attempt was wrong, #56): the
    /// threshold was originally written as "sealing + segments > 2×ceiling". But
    /// that sum does not drop when a sealing FINISHES — the element merely moves
    /// from the queue into the segments and the sum stays constant. Only a merge
    /// lowers it, and a merge runs only once the segment count exceeds the
    /// ceiling. The result: in the 1M measurement the writer fell to 0 op/s for
    /// 110 seconds and hit the 60 s safety limit. The dimension that needs
    /// bounding is not the segment count (the merge ceiling already bounds that)
    /// but **the queue itself, which grows without bound**.
    ///
    /// Threshold 2: one can wait in line while another is being built. Memory
    /// thus stays bounded at ~2 buffers and the sustainable write rate settles
    /// at the sealing rate — which is exactly the desired behaviour.
    fn apply_backpressure(&self) {
        const QUEUE_LIMIT: usize = 2;
        let queued = |s: &Self| s.sealing.read().expect("kilit").len();
        if queued(self) <= QUEUE_LIMIT {
            return;
        }
        let t = std::time::Instant::now();
        while queued(self) > QUEUE_LIMIT {
            // Safety valve: if the worker unexpectedly fails to progress, the
            // writer
            // sonsuza kadar beklemesin.
            if t.elapsed() > std::time::Duration::from_secs(60) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        self.stall_count.fetch_add(1, Ordering::Relaxed);
        self.stall_us
            .fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
    }

    /// Backpressure observability: how many inserts were stalled, for how many µs.
    pub fn stall_stats(&self) -> (u64, u64) {
        (
            self.stall_count.load(Ordering::Relaxed),
            self.stall_us.load(Ordering::Relaxed),
        )
    }

    /// A snapshot list of the buffers currently being sealed (9a-2).
    fn sealing_snapshot(&self) -> Vec<Arc<Sealing>> {
        self.sealing
            .read()
            .expect("kilit")
            .iter()
            .cloned()
            .collect()
    }

    /// Is an id LIVE in one of the buffers being sealed?
    fn sealing_contains_live(&self, id: VectorId) -> bool {
        self.sealing
            .read()
            .expect("kilit")
            .iter()
            .any(|s| s.data.contains(id) && !s.tombstones.read().expect("kilit").contains(&id))
    }

    /// The movable context of the merge trigger.
    fn merge_context(&self) -> MergeContext {
        MergeContext {
            segments: Arc::clone(&self.segments),
            stats: Arc::clone(&self.merge_stats),
            in_flight: Arc::clone(&self.merge_in_flight),
            dim: self.dim,
            metric: self.metric,
            max_segments: self.max_segments,
            params: self.hnsw_params.clone(),
        }
    }

    /// Is a sealing running in the background? (9a-2)
    pub fn seal_in_flight(&self) -> usize {
        self.seal_in_flight.load(Ordering::SeqCst)
    }

    /// Duration of the last sealing (µs) — after 9a-2 this is time spent IN THE
    /// BACKGROUND.
    pub fn last_seal_build_us(&self) -> u64 {
        self.seal_stats.last_us.load(Ordering::Relaxed)
    }

    /// Waits until every in-flight sealing has finished.
    /// With a single worker, "in flight" means the worker is active OR the queue
    /// is non-empty; ending the wait before both are clear would return early.
    pub fn wait_for_seal(&self) {
        while self.seal_in_flight.load(Ordering::SeqCst) > 0
            || !self.sealing.read().expect("kilit").is_empty()
        {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Waits until all sealing and merge work has finished.
    pub fn wait_for_background(&self) {
        loop {
            self.wait_for_seal();
            self.wait_for_merge();
            if self.seal_in_flight.load(Ordering::SeqCst) == 0
                && self.sealing.read().expect("kilit").is_empty()
                && !self.merge_in_flight()
            {
                break;
            }
        }
    }

    /// Is a merge running in the background?
    pub fn merge_in_flight(&self) -> bool {
        self.merge_in_flight.load(Ordering::SeqCst)
    }

    /// For tests/measurements: waits until the background merge finishes.
    pub fn wait_for_merge(&self) {
        while self.merge_in_flight.load(Ordering::SeqCst) {
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Duration of the last sealing (µs) — the window that blocks the writer.
    pub fn last_seal_us(&self) -> u64 {
        self.last_seal_us.load(Ordering::Relaxed)
    }

    /// Duration of the last merge (µs) and the total merge count. After 9a-1
    /// this is time spent on the BACKGROUND thread — the writer is not blocked
    /// during it.
    pub fn last_merge_us(&self) -> u64 {
        self.merge_stats.last_us.load(Ordering::Relaxed)
    }

    pub fn merge_count(&self) -> u64 {
        self.merge_stats.count.load(Ordering::Relaxed)
    }

    /// Changes the WAL policy (continuing with the same file). Used by
    /// measurements to compare all three policies on a single index.
    pub fn set_wal_policy(&self, policy: SyncPolicy) -> Result<(), IndexError> {
        let mut guard = self.wal.write().expect("kilit");
        if let Some(old) = guard.take() {
            let path = old.path().to_path_buf();
            drop(old);
            *guard = Some(
                Wal::open_append(path, policy).map_err(|e| IndexError::Storage(e.to_string()))?,
            );
        }
        Ok(())
    }

    /// Computed memory cost of the metadata structures (bytes):
    /// (the metadata map, the Eq posting lists, the numeric field indexes).
    /// The 9c threshold is evaluated against these computed values — metadata
    /// cannot be isolated within RSS (DECISIONS #40).
    pub fn metadata_memory_bytes(&self) -> (usize, usize, usize) {
        // 9c: the compact representation knows its own size (dictionary + record
        // bodies).
        let meta_bytes = self.metadata.read().expect("kilit").memory_bytes();
        let postings = self.postings.read().expect("kilit");
        let mut post_bytes = postings.capacity() * 64;
        for ((k, _), list) in postings.iter() {
            // A sorted Vec: exactly 8 bytes per id (16 plus slack in a HashSet).
            post_bytes += k.len() + list.capacity() * 8;
        }
        let numeric = self.numeric.read().expect("kilit");
        let num_bytes: usize = numeric
            .iter()
            .map(|(k, fi)| k.len() + fi.memory_bytes())
            .sum();
        (meta_bytes, post_bytes, num_bytes)
    }

    /// A MEASUREMENT HOOK (9c-0): empties the derived metadata structures ONE BY
    /// ONE.
    ///
    /// Why it exists: `metadata_memory_bytes()` works from rough estimates
    /// (capacity × a fixed factor) and 9c's GO decision rests on that estimate.
    /// The way to validate an estimate is to drop the structure and measure the
    /// RSS delta. `clear()` is NOT ENOUGH (it keeps the capacity) — the structure
    /// is REPLACED with a new one.
    ///
    /// This renders the index unusable; it is only called from measurement
    /// modes.
    pub fn clear_for_measurement(&self, what: &str) {
        match what {
            "numeric" => *self.numeric.write().expect("kilit") = HashMap::new(),
            "postings" => *self.postings.write().expect("kilit") = HashMap::new(),
            "metadata" => *self.metadata.write().expect("kilit") = MetaStore::new(),
            other => panic!("unknown measurement hook: {other}"),
        }
    }

    /// Shared (&self) deletion — from the single writer thread.
    /// The metadata is dropped too (so old metadata cannot leak into a
    /// re-insertion).
    pub fn delete_shared(&self, id: VectorId) -> Result<(), IndexError> {
        // Write-ahead: first "is this id live" (a mutation-free check), then the
        // WAL, then the actual deletion.
        if !self.contains_live(id) {
            return Err(IndexError::NotFound(id));
        }
        if let Some(w) = self.wal.write().expect("kilit").as_mut() {
            w.append(&WalRecord::delete(id))
                .map_err(|e| IndexError::Storage(e.to_string()))?;
        }
        self.apply_delete(id)
    }

    /// Does this id belong to a live record? (the buffer, or a segment without
    /// a tombstone for it)
    fn contains_live(&self, id: VectorId) -> bool {
        if self.buffer.read().expect("kilit").contains(id) {
            return true;
        }
        if self.sealing_contains_live(id) {
            return true;
        }
        self.segments.read().expect("kilit").iter().any(|seg| {
            seg.index.contains(id) && !seg.tombstones.read().expect("kilit").contains(&id)
        })
    }

    /// The in-memory side of a deletion (no WAL); replay uses this path.
    fn apply_delete(&self, id: VectorId) -> Result<(), IndexError> {
        let res = self.delete_vector_only(id);
        if res.is_ok() {
            // Posting lists contain only live ids: read the record's keys before
            // dropping its metadata, and remove it from those lists.
            if let Some(meta) = self.metadata.write().expect("kilit").remove(id) {
                let mut postings = self.postings.write().expect("kilit");
                for (key, value) in &meta {
                    if let Some(list) = postings.get_mut(&(key.clone(), value.key())) {
                        posting_remove(list, id);
                    }
                }
                drop(postings);
                let mut numeric = self.numeric.write().expect("kilit");
                for (key, value) in &meta {
                    if let Some(v) = numeric_value(value) {
                        if let Some(fi) = numeric.get_mut(key) {
                            fi.remove(v, id);
                        }
                    }
                }
            }
        }
        res
    }

    fn delete_vector_only(&self, id: VectorId) -> Result<(), IndexError> {
        // The buffer first: if it is there, a real deletion (brute-force
        // swap-remove).
        {
            let mut buffer = self.buffer.write().expect("kilit");
            match buffer.delete(id) {
                Ok(()) => return Ok(()),
                Err(IndexError::NotFound(_)) => {}
                Err(e) => return Err(e),
            }
        }
        // 9a-2: then the buffers being sealed (newest to oldest). Because the
        // snapshot is immutable, a deletion here is a TOMBSTONE as well; when the
        // build finishes, diff-replay carries it into the merged segment.
        {
            let sealing = self.sealing.read().expect("kilit");
            for sl in sealing.iter().rev() {
                if sl.data.contains(id) {
                    let mut tombs = sl.tombstones.write().expect("kilit");
                    return if tombs.insert(id) {
                        Ok(())
                    } else {
                        Err(IndexError::NotFound(id))
                    };
                }
            }
        }
        // Then the segments (in a re-insertion chain the live copy is the newest
        // one; since re-insertions go to the buffer, at most one live copy can
        // exist across the segments).
        let segments = self.segments.read().expect("kilit");
        for seg in segments.iter().rev() {
            if seg.index.contains(id) {
                let mut tombs = seg.tombstones.write().expect("kilit");
                if tombs.insert(id) {
                    return Ok(());
                }
                // zaten tombstone'luysa daha eski segmentlere bakmaya gerek yok:
                // if there were a live copy it would have been removed when this
                // tombstone was written
                return Err(IndexError::NotFound(id));
            }
        }
        Err(IndexError::NotFound(id))
    }

    /// Essentially lock-free search: the segment list is cloned, the HNSW
    /// searches run holding no lock, and the buffer search takes a short read
    /// lock.
    pub fn search_shared(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.search_shared_with_ef(query, k, self.ef_search)
    }

    /// The ef-parameterized form of `search_shared` (for parameter sweeps and
    /// measurements; it does not affect the graph, only the query width).
    pub fn search_shared_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        if k == 0 {
            return Vec::new();
        }
        let segments: Vec<Arc<Segment>> = self
            .segments
            .read()
            .expect("kilit")
            .iter()
            .cloned()
            .collect();

        let mut all: Vec<SearchResult> = Vec::new();
        for seg in &segments {
            let tombs = seg.tombstones.read().expect("kilit");
            // Ask for extra candidates since tombstones can eliminate results;
            // even if the tombstone count exceeds k, ef already sets the upper
            // bound.
            let want = k + tombs.len().min(k);
            let res = seg.index.search_with_ef(query, want, ef.max(want));
            all.extend(res.into_iter().filter(|r| !tombs.contains(&r.id)));
        }
        // 9a-2: the buffers being sealed are walked too — otherwise that data
        // would be INVISIBLE for the duration of the sealing.
        for sealing in self.sealing_snapshot() {
            let tombs = sealing.tombstones.read().expect("kilit");
            all.extend(
                sealing
                    .data
                    .search(query, k + tombs.len().min(k))
                    .into_iter()
                    .filter(|r| !tombs.contains(&r.id)),
            );
        }
        {
            let buffer = self.buffer.read().expect("kilit");
            all.extend(buffer.search(query, k));
        }
        // Deduplicate by id (for the duplicates that exist during the sealing
        // window): copies of the same id hold the same vector, so it does not
        // matter which one survives.
        all.sort();
        let mut seen = HashSet::with_capacity(all.len());
        all.retain(|r| seen.insert(r.id));
        all.truncate(k);
        all
    }

    pub fn len_shared(&self) -> usize {
        let segments = self.segments.read().expect("kilit");
        let seg_live: usize = segments
            .iter()
            .map(|s| s.index.len() - s.tombstones.read().expect("kilit").len())
            .sum();
        // 9a-2: the buffers being sealed carry live records too.
        let sealing_live: usize = self
            .sealing
            .read()
            .expect("kilit")
            .iter()
            .map(|s| s.data.len() - s.tombstones.read().expect("kilit").len())
            .sum();
        seg_live + sealing_live + self.buffer.read().expect("kilit").len()
    }

    // ---- Persistence: the cold path (phase 7a) ----

    /// Attaches a persistence directory (turning an in-memory index into a
    /// persistent one).
    pub fn attach_storage(&self, dir: PathBuf) {
        *self.storage_dir.write().expect("kilit") = Some(dir);
    }

    pub fn storage_dir(&self) -> Option<PathBuf> {
        self.storage_dir.read().expect("kilit").clone()
    }

    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Relaxed)
    }

    /// Unix time of the last successful checkpoint; 0 = none yet.
    pub fn last_checkpoint_unix(&self) -> u64 {
        self.last_checkpoint.load(Ordering::Relaxed)
    }

    /// Checkpoint: seal the buffer → write the new segments → write the metadata
    /// snapshot → swap the manifest atomically → clean up unreferenced files.
    ///
    /// The write order is critical: the manifest is written LAST and GC runs
    /// AFTER it. That way the manifest on disk is at every instant consistent
    /// with all the files it references; whichever step is interrupted, the old
    /// manifest stays valid (new files are orphaned and a later GC collects
    /// them).
    ///
    /// Called from the writer task, per the single-writer contract.
    pub fn checkpoint(&self) -> Result<u64, StorageError> {
        let dir = self.storage_dir().ok_or_else(|| {
            StorageError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no persistence directory attached (attach_storage/open_or_create)",
            ))
        })?;
        std::fs::create_dir_all(&dir)?;
        // Seal the buffer so that after the checkpoint all data lives in
        // segments and the WAL rotation (7b) orphans no record.
        //
        // 9a-2: sealing is now ASYNCHRONOUS. Without waiting, the data in the
        // `sealing` list would be in no segment, the manifest would not see it,
        // and the WAL rotation would orphan it → DATA LOSS. So a checkpoint waits
        // for the sealing and merge work to finish (a rare and predictable pause,
        // not a window on the write path).
        self.seal();
        self.wait_for_background();
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;

        let segments: Vec<Arc<Segment>> = self
            .segments
            .read()
            .expect("kilit")
            .iter()
            .cloned()
            .collect();
        let mut refs = Vec::with_capacity(segments.len());
        for (i, seg) in segments.iter().enumerate() {
            let existing = seg.stored.read().expect("kilit").clone();
            let stored = match existing {
                // Immutable: a segment written once is never written again.
                Some(s) => s,
                None => {
                    let bytes = seg.index.to_bytes()?;
                    let name = Manifest::segment_file_name(generation, i);
                    storage::write_file_durable(&dir.join(&name), &bytes)?;
                    let s = StoredFile {
                        name,
                        crc32: storage::crc32(&bytes),
                    };
                    *seg.stored.write().expect("kilit") = Some(s.clone());
                    s
                }
            };
            refs.push(SegmentRef {
                file: stored.name,
                crc32: stored.crc32,
                records: seg.index.len() as u64,
                tombstones: seg
                    .tombstones
                    .read()
                    .expect("kilit")
                    .iter()
                    .map(|id| id.0)
                    .collect(),
            });
        }

        let entries: Vec<(VectorId, Metadata)> = self
            .metadata
            .read()
            .expect("kilit")
            .iter()
            .map(|(id, m)| (id, m.to_metadata()))
            .collect();
        let (metadata_file, metadata_crc) = if entries.is_empty() {
            (None, 0)
        } else {
            let bytes = storage::encode_metadata(&entries)?;
            let name = Manifest::metadata_file_name(generation);
            storage::write_file_durable(&dir.join(&name), &bytes)?;
            (Some(name), storage::crc32(&bytes))
        };

        // WAL rotation: because the buffer was sealed, ALL records of the old
        // WAL now live in segments. The new (empty) file is opened BEFORE the
        // manifest is written; the manifest points at the new name and GC deletes
        // the old one. On an interruption the old manifest still points at the
        // old WAL — consistent.
        let wal_file = {
            let mut guard = self.wal.write().expect("kilit");
            match guard.as_ref() {
                Some(old) => {
                    let policy = old.policy();
                    let name = Manifest::wal_file_name(generation);
                    *guard = Some(Wal::open_append(dir.join(&name), policy)?);
                    Some(name)
                }
                None => None,
            }
        };

        let now = storage::now_unix_secs();
        let manifest = Manifest {
            generation,
            dim: self.dim as u64,
            metric: self.metric,
            hnsw_params: self.hnsw_params.clone(),
            seal_threshold: self.seal_threshold.load(Ordering::Relaxed) as u64,
            max_segments: self.max_segments as u64,
            segments: refs,
            metadata_file,
            metadata_crc,
            wal_file,
            created_unix_secs: now,
        };
        manifest.write(&dir)?;
        storage::gc_unreferenced(&dir, &manifest)?;
        self.last_checkpoint.store(now, Ordering::Relaxed);
        Ok(generation)
    }

    /// Forces an fsync of the WAL (closing the group window). Called by graceful
    /// shutdown and at the end of the writer task's batch.
    pub fn flush_wal(&self) -> Result<(), IndexError> {
        if let Some(w) = self.wal.write().expect("kilit").as_mut() {
            w.sync().map_err(|e| IndexError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    /// fsyncs if the group window has elapsed; a cheap call at the end of a batch.
    pub fn sync_wal_if_due(&self) -> Result<bool, IndexError> {
        if let Some(w) = self.wal.write().expect("kilit").as_mut() {
            return w
                .sync_if_due()
                .map_err(|e| IndexError::Storage(e.to_string()));
        }
        Ok(false)
    }

    /// End-of-batch commit: provides the durability the policy promises (None →
    /// hand to the OS only, the others → fsync). The writer task calls it BEFORE
    /// sending HTTP responses; group commit's "200 = fsynced" contract rests on
    /// it.
    pub fn commit_wal(&self) -> Result<(), IndexError> {
        if let Some(w) = self.wal.write().expect("kilit").as_mut() {
            w.commit().map_err(|e| IndexError::Storage(e.to_string()))?;
        }
        Ok(())
    }

    /// Size of the active WAL in bytes (0 = no WAL).
    pub fn wal_len_bytes(&self) -> u64 {
        self.wal
            .read()
            .expect("kilit")
            .as_ref()
            .map(|w| w.len_bytes())
            .unwrap_or(0)
    }

    pub fn wal_policy_label(&self) -> Option<String> {
        self.wal
            .read()
            .expect("kilit")
            .as_ref()
            .map(|w| w.policy().label())
    }

    /// Report of the WAL replay performed at startup.
    pub fn replay_report(&self) -> ReplayReport {
        self.replay_report.read().expect("kilit").clone()
    }

    /// Builds from the manifest if the directory has one; otherwise opens an
    /// empty index with the given parameters. Either way the directory is
    /// attached.
    ///
    /// When a manifest exists, dim/metric/params/thresholds come FROM IT: the
    /// truth on disk overrides the caller's assumption (rather than opening with
    /// the wrong dim and corrupting the data).
    pub fn open_or_create(
        dir: PathBuf,
        dim: usize,
        metric: Metric,
        hnsw_params: HnswParams,
        seal_threshold: usize,
    ) -> Result<Self, StorageError> {
        Self::open_inner(dir, dim, metric, hnsw_params, seal_threshold, None)
    }

    /// Opening with a WAL: the manifest and segments are loaded, then the WAL is
    /// replayed and attached in append mode. Recovery is always completed with
    /// the WAL — the manifest alone is not considered sufficient (the Windows
    /// directory-fsync gap in DECISIONS #33).
    pub fn open_durable(
        dir: PathBuf,
        dim: usize,
        metric: Metric,
        hnsw_params: HnswParams,
        seal_threshold: usize,
        policy: SyncPolicy,
    ) -> Result<Self, StorageError> {
        Self::open_inner(dir, dim, metric, hnsw_params, seal_threshold, Some(policy))
    }

    fn open_inner(
        dir: PathBuf,
        dim: usize,
        metric: Metric,
        hnsw_params: HnswParams,
        seal_threshold: usize,
        wal_policy: Option<SyncPolicy>,
    ) -> Result<Self, StorageError> {
        std::fs::create_dir_all(&dir)?;
        let Some(manifest) = Manifest::read(&dir)? else {
            let idx = Self::new(dim, metric, hnsw_params, seal_threshold);
            idx.attach_storage(dir.clone());
            if let Some(policy) = wal_policy {
                // There is no manifest but there may be a WAL: if the previous
                // run crashed before reaching a checkpoint, all the data is
                // there.
                let name = Manifest::wal_file_name(0);
                idx.recover_wal(&dir, &name, policy)?;
            }
            return Ok(idx);
        };
        let mut idx = Self::new(
            manifest.dim as usize,
            manifest.metric,
            manifest.hnsw_params.clone(),
            manifest.seal_threshold as usize,
        );
        idx.set_max_segments(manifest.max_segments as usize);

        let mut segments = Vec::with_capacity(manifest.segments.len());
        for sref in &manifest.segments {
            let bytes = storage::read_verified(&dir, &sref.file, sref.crc32)?;
            let index = HnswIndex::load_from_bytes(&bytes)?;
            let tombstones: HashSet<VectorId> =
                sref.tombstones.iter().map(|&i| VectorId(i)).collect();
            segments.push(Arc::new(Segment {
                index,
                tombstones: RwLock::new(tombstones),
                stored: RwLock::new(Some(StoredFile {
                    name: sref.file.clone(),
                    crc32: sref.crc32,
                })),
            }));
        }
        *idx.segments.write().expect("kilit") = segments;

        if let Some(file) = &manifest.metadata_file {
            let bytes = storage::read_verified(&dir, file, manifest.metadata_crc)?;
            // The derived structures (posting lists, numeric indexes) are
            // rebuilt here — they are never written to disk, and metadata is
            // their only source.
            for (id, meta) in storage::decode_metadata(&bytes, &dir.join(file))? {
                idx.index_metadata(id, meta);
            }
        }
        idx.generation.store(manifest.generation, Ordering::Relaxed);
        idx.last_checkpoint
            .store(manifest.created_unix_secs, Ordering::Relaxed);
        idx.attach_storage(dir.clone());
        if let Some(policy) = wal_policy {
            let name = manifest
                .wal_file
                .clone()
                .unwrap_or_else(|| Manifest::wal_file_name(manifest.generation));
            idx.recover_wal(&dir, &name, policy)?;
        }
        Ok(idx)
    }

    /// WAL replay plus attaching the log in append mode.
    ///
    /// During replay `self.wal` is STILL None, so records are not written again;
    /// that makes the "log what I just replayed" bug structurally impossible.
    fn recover_wal(
        &self,
        dir: &std::path::Path,
        name: &str,
        policy: SyncPolicy,
    ) -> Result<(), StorageError> {
        let path = dir.join(name);
        let (records, report) = wal::replay(&path)?;
        for rec in records {
            match rec {
                WalRecord::Insert { id, vector, meta } => {
                    self.apply_insert(VectorId(id), &vector, wal::record_meta(meta))?;
                }
                WalRecord::Delete { id } => {
                    // During replay a "not found" can be legitimate at a rotation
                    // boundary: skip it silently, never synthesize a phantom op.
                    let _ = self.apply_delete(VectorId(id));
                }
            }
        }
        *self.replay_report.write().expect("kilit") = report;
        *self.wal.write().expect("kilit") = Some(Wal::open_append(path, policy)?);
        Ok(())
    }

    /// Total index memory (vectors + graph, across all segments; bytes).
    pub fn memory_bytes(&self) -> usize {
        self.segments
            .read()
            .expect("kilit")
            .iter()
            .map(|s| {
                let (v, l) = s.index.memory_bytes();
                v + l
            })
            .sum::<usize>()
            + self.buffer.read().expect("kilit").memory_bytes()
    }

    /// For measurement: produces an int8 (quantized) copy of every sealed
    /// segment. The conversion happens here so the `Segment` type is not leaked
    /// outside.
    ///
    /// NOTE: this is a MEASUREMENT path, not a production one — quantized
    /// segments are NOT integrated into `SegmentedIndex` (tombstones, the buffer
    /// and the planner stay on the f32 side). The integration decision depends on
    /// the 8a measurement.
    pub fn quantize_segments(&self) -> Vec<crate::index::quant::QuantizedHnsw> {
        self.segments
            .read()
            .expect("kilit")
            .iter()
            .map(|s| crate::index::quant::QuantizedHnsw::from_hnsw(&s.index))
            .collect()
    }

    /// Observability: (segment count, buffer occupancy).
    /// Observability: the number of buffers being sealed (the 9a-2 accumulation
    /// indicator).
    pub fn sealing_count(&self) -> usize {
        self.sealing.read().expect("kilit").len()
    }

    pub fn shape(&self) -> (usize, usize) {
        (
            self.segments.read().expect("kilit").len(),
            self.buffer.read().expect("kilit").len(),
        )
    }
}

/// Trait conformance: for single-threaded use, the &mut signatures delegate to
/// the shared
/// implementasyona delege eder.
impl VectorIndex for SegmentedIndex {
    fn insert(&mut self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        self.insert_shared(id, vector)
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.search_shared(query, k)
    }

    fn delete(&mut self, id: VectorId) -> Result<(), IndexError> {
        self.delete_shared(id)
    }

    fn len(&self) -> usize {
        self.len_shared()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::random_vectors;
    use crate::eval::exact_top_k;
    use std::sync::atomic::{AtomicBool, Ordering};

    fn build(vecs: &[Vec<f32>], seal_threshold: usize) -> SegmentedIndex {
        let idx = SegmentedIndex::new(
            vecs[0].len(),
            Metric::L2,
            HnswParams::default(),
            seal_threshold,
        );
        for (i, v) in vecs.iter().enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        idx
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 100);
        assert!(idx.is_empty());
        assert!(idx.search(&[0.0; 4], 5).is_empty());
    }

    #[test]
    fn single_element_and_k_larger_than_len() {
        let idx = build(&[vec![1.0, 2.0]], 100);
        let res = idx.search(&[0.0, 0.0], 10);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId(0));
    }

    #[test]
    fn spans_multiple_segments_and_buffer() {
        let vecs = random_vectors(1_050, 8, 42);
        let idx = build(&vecs, 400); // 2 segment (400+400) + 250 buffer
        idx.wait_for_background(); // 9a-2: sealing runs in the background
        let (n_seg, n_buf) = idx.shape();
        assert_eq!(n_seg, 2);
        assert_eq!(n_buf, 250);
        assert_eq!(idx.len(), 1_050);
        // correctness: high agreement with the exact reference
        let queries = random_vectors(20, 8, 43);
        let mut hits = 0;
        for q in &queries {
            let truth: Vec<_> = exact_top_k(&vecs, q, 10, Metric::L2)
                .iter()
                .map(|r| r.id)
                .collect();
            hits += idx
                .search(q, 10)
                .iter()
                .filter(|r| truth.contains(&r.id))
                .count();
        }
        let recall = hits as f64 / (20 * 10) as f64;
        assert!(recall >= 0.95, "segmentli recall {recall}");
    }

    #[test]
    fn duplicate_id_rejected_across_sources() {
        let vecs = random_vectors(500, 4, 42);
        let idx = build(&vecs, 300); // id 0..299 segmentte, 300.. buffer'da
                                     // segmentteki id
        assert!(matches!(
            idx.insert_shared(VectorId(5), &[0.0; 4]),
            Err(IndexError::DuplicateId(_))
        ));
        // buffer'daki id
        assert!(matches!(
            idx.insert_shared(VectorId(450), &[0.0; 4]),
            Err(IndexError::DuplicateId(_))
        ));
    }

    #[test]
    fn delete_from_buffer_and_segment() {
        let vecs = random_vectors(500, 4, 42);
        let idx = build(&vecs, 300);
        // a real deletion from the buffer
        idx.delete_shared(VectorId(400)).unwrap();
        // segmentten tombstone silme
        idx.delete_shared(VectorId(5)).unwrap();
        assert_eq!(idx.len(), 498);
        for q in random_vectors(10, 4, 43) {
            let ids: Vec<_> = idx.search(&q, 498).iter().map(|r| r.id).collect();
            assert!(!ids.contains(&VectorId(400)) && !ids.contains(&VectorId(5)));
        }
        assert!(matches!(
            idx.delete_shared(VectorId(5)),
            Err(IndexError::NotFound(_))
        ));
    }

    #[test]
    fn reinsert_after_segment_delete_returns_new_vector() {
        let vecs = random_vectors(300, 4, 42);
        let idx = build(&vecs, 200); // id 5 segmentte
        idx.delete_shared(VectorId(5)).unwrap();
        idx.insert_shared(VectorId(5), &[9.0; 4]).unwrap();
        assert_eq!(idx.len(), 300);
        // the new vector is found, the old copy stays shadowed
        let res = idx.search(&[9.0; 4], 1);
        assert_eq!(res[0].id, VectorId(5));
        let old = idx.search(&vecs[5].clone(), 3);
        // if id 5 comes back at the old position it is a resurrected old copy —
        // it must not (the new vector [9;4] is far from the old position)
        assert!(old.iter().all(|r| r.id != VectorId(5)));
    }

    #[test]
    fn zero_vector_and_duplicate_vectors() {
        let idx = SegmentedIndex::new(3, Metric::Cosine, HnswParams::default(), 10);
        idx.insert_shared(VectorId(0), &[0.0; 3]).unwrap();
        idx.insert_shared(VectorId(1), &[1.0, 0.0, 0.0]).unwrap();
        idx.insert_shared(VectorId(2), &[1.0, 0.0, 0.0]).unwrap();
        let res = idx.search(&[1.0, 0.0, 0.0], 3);
        assert_eq!(res.len(), 3);
        assert!(res.iter().all(|r| !r.distance.is_nan()));
        assert!(res[..2].iter().all(|r| r.id != VectorId(0)));
    }

    // ---- Metadata filtreleme testleri ----

    use crate::meta::{MetaValue, Predicate};

    /// Records are split into two categories by id parity; a filtered search must
    /// return only the requested category and agree with the filtered brute-force
    /// reference.
    #[test]
    fn filtered_search_matches_reference() {
        let vecs = random_vectors(1_000, 8, 42);
        let idx = SegmentedIndex::new(8, Metric::L2, HnswParams::default(), 400);
        let mut bf = BruteForceIndex::new(8, Metric::L2);
        for (i, v) in vecs.iter().enumerate() {
            let meta: Metadata = [(
                "grup".to_string(),
                MetaValue::Str(if i % 2 == 0 { "even" } else { "odd" }.into()),
            )]
            .into();
            idx.insert_with_meta(VectorId(i as u64), v, meta).unwrap();
            bf.insert(VectorId(i as u64), v).unwrap();
        }
        let filter = Filter {
            must: vec![Predicate::Eq {
                key: "grup".into(),
                value: MetaValue::Str("even".into()),
            }],
        };
        let allow = |id: VectorId| id.0.is_multiple_of(2);
        let queries = random_vectors(20, 8, 43);
        let mut hits = 0;
        for q in &queries {
            let res = idx.search_filtered(q, 10, &filter);
            assert_eq!(res.len(), 10);
            assert!(res.iter().all(|r| r.id.0.is_multiple_of(2)), "filter leak");
            let truth: Vec<_> = bf
                .search_filtered(q, 10, &allow)
                .iter()
                .map(|r| r.id)
                .collect();
            hits += res.iter().filter(|r| truth.contains(&r.id)).count();
        }
        assert!(
            hits as f64 / 200.0 >= 0.95,
            "filtered recall too low: {hits}/200"
        );
    }

    /// #53: sealing runs on a SINGLE worker and the queue is drained.
    ///
    /// Measures two things at once: (a) at no point is more than one sealing
    /// running — in the old design 35 ran concurrently (#52); (b) a queue really
    /// did form (otherwise the test proves nothing) and was eventually drained,
    /// with no data loss.
    #[test]
    fn sealing_worker_is_single_and_queue_drains() {
        let n = 8_000;
        let vecs = random_vectors(n, 16, 42);
        let mut idx = SegmentedIndex::new(16, Metric::L2, HnswParams::default(), 200);
        idx.set_max_segments(2);
        let mut max_in_flight = 0;
        let mut max_queue = 0;
        for (i, v) in vecs.iter().enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
            max_in_flight = max_in_flight.max(idx.seal_in_flight());
            max_queue = max_queue.max(idx.sealing_count());
        }
        assert!(
            max_in_flight <= 1,
            "more than one sealing worker: {max_in_flight}"
        );
        assert!(
            max_queue > 0,
            "no sealing queue ever formed: the test is weak"
        );
        idx.wait_for_background();
        assert_eq!(idx.sealing_count(), 0, "the queue was not drained");
        assert_eq!(idx.len_shared(), n, "data was lost during sealing");
    }

    /// #53: backpressure bounds the queue — writes are slowed, not rejected.
    ///
    /// The test verifies both that the bound holds and that the stall was
    /// ACTUALLY triggered; if it never triggers, the bound holds by itself and
    /// the test silently measures nothing.
    #[test]
    fn backpressure_bounds_the_sealing_queue() {
        let n = 8_000;
        let vecs = random_vectors(n, 16, 7);
        let mut idx = SegmentedIndex::new(16, Metric::L2, HnswParams::default(), 100);
        idx.set_max_segments(2);
        let limit = 2; // the queue threshold (#56: the signal is NOT the segment count)
        let mut max_queue = 0;
        for (i, v) in vecs.iter().enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
            max_queue = max_queue.max(idx.sealing_count());
        }
        let (stalls, stall_us) = idx.stall_stats();
        assert!(stalls > 0, "backpressure never kicked in: the test is weak");
        assert!(stall_us > 0, "stall duration was not measured");
        // Since the wait happens AFTER sealing, the bound can be exceeded by one.
        assert!(
            max_queue <= limit + 1,
            "queue bound exceeded: {max_queue} > {}",
            limit + 1
        );
        idx.wait_for_background();
        assert_eq!(idx.len_shared(), n, "data was lost under backpressure");
    }

    /// A #61 regression: pre-allocating the buffer up to the threshold panicked
    /// with a capacity overflow wherever the threshold is given as `usize::MAX`
    /// to mean "no sealing in practice". The allocation is now bounded in bytes.
    #[test]
    fn huge_seal_threshold_does_not_panic() {
        let idx = SegmentedIndex::new(128, Metric::L2, HnswParams::default(), usize::MAX);
        let v = vec![0.1f32; 128];
        idx.insert_shared(VectorId(1), &v).unwrap();
        assert_eq!(idx.len_shared(), 1);
    }

    /// A highly selective filter (a single match): the fallback linear scan must
    /// kick in and find that one record.
    #[test]
    fn highly_selective_filter_falls_back_to_exact() {
        let vecs = random_vectors(2_000, 8, 42);
        let idx = SegmentedIndex::new(8, Metric::L2, HnswParams::default(), 500);
        for (i, v) in vecs.iter().enumerate() {
            let meta: Metadata = [("nadir".to_string(), MetaValue::Bool(i == 1_234))].into();
            idx.insert_with_meta(VectorId(i as u64), v, meta).unwrap();
        }
        let filter = Filter {
            must: vec![Predicate::Eq {
                key: "nadir".into(),
                value: MetaValue::Bool(true),
            }],
        };
        let res = idx.search_filtered(&random_vectors(1, 8, 43)[0], 5, &filter);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId(1_234));
        // must return empty when there is no match at all
        let none = Filter {
            must: vec![Predicate::Eq {
                key: "yok".into(),
                value: MetaValue::Bool(true),
            }],
        };
        assert!(idx.search_filtered(&vecs[0].clone(), 5, &none).is_empty());
    }

    /// A deleted record's metadata is dropped; the same id can come back with
    /// new metadata.
    #[test]
    fn delete_drops_metadata_reinsert_gets_fresh() {
        let vecs = random_vectors(100, 4, 42);
        let idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 1_000);
        for (i, v) in vecs.iter().enumerate() {
            let meta: Metadata = [("v".to_string(), MetaValue::Int(1))].into();
            idx.insert_with_meta(VectorId(i as u64), v, meta).unwrap();
        }
        idx.delete_shared(VectorId(7)).unwrap();
        idx.insert_with_meta(
            VectorId(7),
            &vecs[7],
            [("v".to_string(), MetaValue::Int(2))].into(),
        )
        .unwrap();
        let f_v2 = Filter {
            must: vec![Predicate::Eq {
                key: "v".into(),
                value: MetaValue::Int(2),
            }],
        };
        let res = idx.search_filtered(&vecs[7].clone(), 1, &f_v2);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId(7));
    }

    // ---- 9a-2: background sealing + the "two buffers" state ----

    /// The structural risk of 9a-2 (DECISIONS #50): once sealing moves to the
    /// background, the buffer being sealed and the new buffer coexist. ALL THREE
    /// paths must walk this source: search, duplicate-id and delete.
    ///
    /// The test follows the 9a-1 pattern: "did the two-buffer state actually
    /// occur" is asserted inside (`seal_in_flight() > 0`), otherwise the test
    /// silently weakens.
    #[test]
    fn sealing_window_covers_search_duplicate_and_delete() {
        let vecs = random_vectors(6_000, 16, 42);
        let idx = SegmentedIndex::new(16, Metric::L2, HnswParams::default(), 5_000);
        for (i, v) in vecs.iter().take(5_000).enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        // The 5,000th insert triggered sealing; the build is running in the
        // background.
        assert!(
            idx.seal_in_flight() > 0,
            "the two-buffer state did NOT occur — the test has silently weakened"
        );
        assert_eq!(
            idx.sealing_count(),
            1,
            "the buffer being sealed is not visible"
        );

        // (a) SEARCH: the data being sealed must stay visible.
        let res = idx.search_shared(&vecs[42], 1);
        assert_eq!(
            res[0].id,
            VectorId(42),
            "data became invisible during the sealing window"
        );
        assert_eq!(
            idx.len(),
            5_000,
            "records being sealed dropped out of the count"
        );

        // (b) DUPLICATE-ID (the sneakiest): an id in the buffer being sealed
        // yeni buffer'a ikinci kez eklenememeli.
        assert!(
            matches!(
                idx.insert_shared(VectorId(42), &vecs[42]),
                Err(IndexError::DuplicateId(_))
            ),
            "an id from the buffer being sealed was re-inserted — the collision \
             would only surface once sealing finished, and both copies would be \
             permanent"
        );

        // (c) DELETE: a record being sealed must be deletable, and the deletion
        // must be carried into the merged segment by diff-replay once the build
        // finishes.
        idx.delete_shared(VectorId(100)).unwrap();
        assert_eq!(idx.len(), 4_999);
        // Yeni buffer'a yazmaya devam edilebilmeli.
        idx.insert_shared(VectorId(9_000), &vecs[5_500]).unwrap();

        idx.wait_for_background();
        assert_eq!(idx.sealing_count(), 0, "the sealing list was not drained");
        let all: HashSet<VectorId> = idx
            .search_shared(&vecs[100], 6_000)
            .iter()
            .map(|r| r.id)
            .collect();
        assert!(
            !all.contains(&VectorId(100)),
            "a deletion made during sealing was LOST in diff-replay"
        );
        assert!(
            all.contains(&VectorId(9_000)),
            "a record from the new buffer was lost"
        );
        assert_eq!(idx.len(), 5_000); // 5000 - 1 silme + 1 yeni
    }

    /// Continuous searching while sealing is in progress: results must never
    /// disappear at any instant (a reader sees either the buffer being sealed or
    /// the segment it becomes).
    #[test]
    fn searches_never_lose_data_during_sealing() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let vecs = random_vectors(4_000, 8, 7);
        let idx = Arc::new(SegmentedIndex::new(
            8,
            Metric::L2,
            HnswParams::default(),
            3_000,
        ));
        for (i, v) in vecs.iter().take(3_000).enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        assert!(
            idx.seal_in_flight() > 0,
            "the sealing window was not caught"
        );
        let stop = AtomicBool::new(false);
        let missing = AtomicUsize::new(0);
        let saw_sealing = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..3 {
                let (idx, stop, missing, saw) = (&idx, &stop, &missing, &saw_sealing);
                let vecs = &vecs;
                sc.spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        if idx.sealing_count() > 0 {
                            saw.fetch_add(1, Ordering::Relaxed);
                        }
                        for probe in [3usize, 1_500, 2_999] {
                            let r = idx.search_shared(&vecs[probe], 1);
                            if r.is_empty() || r[0].id != VectorId(probe as u64) {
                                missing.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
            idx.wait_for_background();
            stop.store(true, Ordering::Relaxed);
        });
        assert!(
            saw_sealing.load(Ordering::Relaxed) > 0,
            "the readers never saw the sealing window — the test is weak"
        );
        assert_eq!(
            missing.load(Ordering::Relaxed),
            0,
            "a record temporarily DISAPPEARED during sealing"
        );
    }

    // ---- 9a-1: background merge + the tombstone race ----

    /// The REAL acceptance criterion of 9a-1: tombstones landing on the source
    /// segments WHILE a merge runs must not be lost. The merge rebuild copies the
    /// sources' live records as of that moment; records deleted during the build
    /// pass into the merged segment as live, and unless diff-replay tombstones
    /// them they **silently come back**. A latency measurement would never catch
    /// this.
    #[test]
    fn merge_carries_tombstones_created_during_build() {
        // Relatively large segments so the merge takes a while; to land the
        // deletions inside the merge window, the writer thread
        // sonra siler.
        let vecs = random_vectors(3_000, 16, 42);
        let mut idx = SegmentedIndex::new(16, Metric::L2, HnswParams::default(), 500);
        idx.set_max_segments(4);
        let idx = Arc::new(idx);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        // 9a-2: sealing runs in the background too. A merge is only triggered
        // once segments exist, so we first wait for the sealings to finish.
        idx.wait_for_seal();
        // At this point 6 segments were sealed and the ceiling is 4 → a merge has
        // started in the background. Delete while it runs: the victims are the two
        // smallest segments, and since we do not know which ids landed in them we
        // delete over a wide range.
        let mut deleted: Vec<VectorId> = Vec::new();
        let mut i = 0u64;
        let mut during_merge = 0usize; // proves the race was actually triggered
        while idx.merge_in_flight() && i < 400 {
            during_merge += 1;
            let id = VectorId(i);
            if idx.delete_shared(id).is_ok() {
                deleted.push(id);
            }
            i += 1;
        }
        // Do at least a few deletions even if no merge is running (the race may
        // not trigger on every run; the test still checks correctness).
        for extra in i..(i + 50).min(3_000) {
            let id = VectorId(extra);
            if idx.delete_shared(id).is_ok() {
                deleted.push(id);
            }
        }
        idx.wait_for_merge();

        assert!(!deleted.is_empty(), "no deletion could be performed");
        assert!(
            during_merge > 0,
            "the merge window was not caught — the race was NOT exercised, the test has silently weakened"
        );
        // 1) NONE of the deleted ids may come back in a search.
        let all: HashSet<VectorId> = idx
            .search_shared(&vecs[0], 3_000)
            .iter()
            .map(|r| r.id)
            .collect();
        for id in &deleted {
            assert!(
                !all.contains(id),
                "deleted {id:?} came back after the merge (a diff-replay leak)"
            );
        }
        // 2) len must be consistent.
        assert_eq!(
            idx.len(),
            3_000 - deleted.len(),
            "the live count is inconsistent"
        );
        // 3) The non-deleted ones must still be findable.
        let deleted_set: HashSet<VectorId> = deleted.iter().copied().collect();
        let survivor = (0..3_000u64)
            .map(VectorId)
            .find(|id| !deleted_set.contains(id))
            .expect("hayatta kalan yok");
        let res = idx.search_shared(&vecs[survivor.0 as usize], 1);
        assert_eq!(res[0].id, survivor, "a surviving record disappeared");
    }

    /// While a merge runs in the background, searches must not stop and must
    /// stay consistent (readers see either the old pair or the merged segment).
    #[test]
    fn searches_stay_consistent_during_background_merge() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let vecs = random_vectors(2_500, 8, 42);
        let mut idx = SegmentedIndex::new(8, Metric::L2, HnswParams::default(), 400);
        idx.set_max_segments(3);
        let idx = Arc::new(idx);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        let stop = AtomicBool::new(false);
        let mismatch = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..3 {
                let (idx, stop, mismatch) = (&idx, &stop, &mismatch);
                let vecs = &vecs;
                sc.spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        // each query's own vector must come back first
                        for probe in [7usize, 500, 1200, 2400] {
                            let r = idx.search_shared(&vecs[probe], 1);
                            if r.is_empty() || r[0].id != VectorId(probe as u64) {
                                mismatch.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                });
            }
            idx.wait_for_merge();
            stop.store(true, Ordering::Relaxed);
        });
        assert_eq!(
            mismatch.load(Ordering::Relaxed),
            0,
            "a search returned an inconsistent result during the merge"
        );
        assert_eq!(idx.len(), 2_500);
    }

    // ---- Persistence: cold-path tests (phase 7a) ----

    fn meta_of(i: usize) -> Metadata {
        [
            ("grup".to_string(), MetaValue::Int((i % 4) as i64)),
            ("v".to_string(), MetaValue::Int(i as i64)),
        ]
        .into()
    }

    #[test]
    fn checkpoint_reopen_gives_identical_results() {
        let dir = crate::storage::temp_dir("ckpt-roundtrip");
        let vecs = random_vectors(800, 8, 42);
        let queries = random_vectors(15, 8, 43);
        let gen = {
            let idx = SegmentedIndex::open_or_create(
                dir.clone(),
                8,
                Metric::L2,
                HnswParams::default(),
                200,
            )
            .unwrap();
            for (i, v) in vecs.iter().enumerate() {
                idx.insert_with_meta(VectorId(i as u64), v, meta_of(i))
                    .unwrap();
            }
            let before: Vec<Vec<SearchResult>> =
                queries.iter().map(|q| idx.search_shared(q, 10)).collect();
            let gen = idx.checkpoint().unwrap();
            // the sealing done by a checkpoint must not change results
            for (q, b) in queries.iter().zip(&before) {
                assert_eq!(
                    &idx.search_shared(q, 10),
                    b,
                    "the checkpoint corrupted the results"
                );
            }
            gen
        };
        // reopen
        let idx = SegmentedIndex::open_or_create(
            dir.clone(),
            999, // a wrong dim: the manifest's value must win
            Metric::Dot,
            HnswParams::default(),
            1,
        )
        .unwrap();
        assert_eq!(idx.generation(), gen);
        assert_eq!(idx.len(), 800);
        assert_eq!(
            idx.shape().1,
            0,
            "the buffer must be empty after a checkpoint"
        );
        // The same queries must give exactly the same results
        let fresh = SegmentedIndex::new(8, Metric::L2, HnswParams::default(), 200);
        for (i, v) in vecs.iter().enumerate() {
            fresh
                .insert_with_meta(VectorId(i as u64), v, meta_of(i))
                .unwrap();
        }
        fresh.seal();
        for q in &queries {
            assert_eq!(idx.search_shared(q, 10), fresh.search_shared(q, 10));
        }
        // the derived indexes must have been rebuilt: Eq + Range filters
        let f_eq = Filter {
            must: vec![Predicate::Eq {
                key: "grup".into(),
                value: MetaValue::Int(2),
            }],
        };
        let res = idx.search_filtered(&queries[0], 10, &f_eq);
        assert_eq!(res.len(), 10);
        assert!(
            res.iter().all(|r| r.id.0 % 4 == 2),
            "the Eq posting list was not rebuilt"
        );
        let f_range = Filter {
            must: vec![Predicate::Range {
                key: "v".into(),
                min: 0.0,
                max: 49.0,
            }],
        };
        let res = idx.search_filtered(&queries[0], 10, &f_range);
        assert_eq!(
            idx.debug_plan_arm(&f_range, 10),
            "scan",
            "numeric indeks yok"
        );
        assert!(res.iter().all(|r| r.id.0 < 50));
    }

    #[test]
    fn segments_written_once_across_checkpoints() {
        let dir = crate::storage::temp_dir("ckpt-once");
        let vecs = random_vectors(500, 4, 42);
        let idx =
            SegmentedIndex::open_or_create(dir.clone(), 4, Metric::L2, HnswParams::default(), 200)
                .unwrap();
        for (i, v) in vecs.iter().take(400).enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        let g1 = idx.checkpoint().unwrap();
        let first: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let n = e.unwrap().file_name().to_string_lossy().to_string();
                n.starts_with("segment-").then_some(n)
            })
            .collect();
        // second round: new data → a new segment, the old ones must stay in the
        // SAME files
        for (i, v) in vecs.iter().enumerate().skip(400) {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        let g2 = idx.checkpoint().unwrap();
        assert!(g2 > g1);
        let second: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let n = e.unwrap().file_name().to_string_lossy().to_string();
                n.starts_with("segment-").then_some(n)
            })
            .collect();
        for f in &first {
            assert!(
                second.contains(f),
                "an old segment was rewritten/deleted: {f} (immutability violation)"
            );
        }
        assert!(second.len() > first.len(), "no new segment was written");
    }

    #[test]
    fn reopen_restores_tombstones() {
        let dir = crate::storage::temp_dir("ckpt-tombstone");
        let vecs = random_vectors(400, 4, 42);
        {
            let idx = SegmentedIndex::open_or_create(
                dir.clone(),
                4,
                Metric::L2,
                HnswParams::default(),
                150,
            )
            .unwrap();
            for (i, v) in vecs.iter().enumerate() {
                idx.insert_with_meta(VectorId(i as u64), v, meta_of(i))
                    .unwrap();
            }
            idx.checkpoint().unwrap(); // hepsi segmentlerde
            idx.delete_shared(VectorId(7)).unwrap();
            idx.delete_shared(VectorId(200)).unwrap();
            idx.checkpoint().unwrap(); // tombstone'lar manifest'e
        }
        let idx =
            SegmentedIndex::open_or_create(dir, 4, Metric::L2, HnswParams::default(), 150).unwrap();
        assert_eq!(idx.len(), 398);
        let all: Vec<_> = idx
            .search_shared(&vecs[7].clone(), 400)
            .iter()
            .map(|r| r.id)
            .collect();
        assert!(
            !all.contains(&VectorId(7)),
            "the tombstone was not recovered"
        );
        assert!(!all.contains(&VectorId(200)));
        // silinen metadata da geri gelmemeli
        let f = Filter {
            must: vec![Predicate::Eq {
                key: "v".into(),
                value: MetaValue::Int(7),
            }],
        };
        assert!(idx.search_filtered(&vecs[7].clone(), 5, &f).is_empty());
    }

    #[test]
    fn corrupt_segment_file_is_error_not_panic() {
        let dir = crate::storage::temp_dir("ckpt-corrupt");
        let vecs = random_vectors(300, 4, 42);
        {
            let idx = SegmentedIndex::open_or_create(
                dir.clone(),
                4,
                Metric::L2,
                HnswParams::default(),
                100,
            )
            .unwrap();
            for (i, v) in vecs.iter().enumerate() {
                idx.insert_shared(VectorId(i as u64), v).unwrap();
            }
            idx.checkpoint().unwrap();
        }
        let seg_file = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .find(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("segment-"))
            })
            .expect("segment file");
        let mut bytes = std::fs::read(&seg_file).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xff;
        std::fs::write(&seg_file, &bytes).unwrap();
        let err =
            SegmentedIndex::open_or_create(dir.clone(), 4, Metric::L2, HnswParams::default(), 100);
        assert!(err.is_err(), "a corrupt segment should have been caught");
        // a truncated file must not panic either
        std::fs::write(&seg_file, &bytes[..bytes.len() / 3]).unwrap();
        assert!(
            SegmentedIndex::open_or_create(dir, 4, Metric::L2, HnswParams::default(), 100).is_err()
        );
    }

    #[test]
    fn merge_output_persists_and_gc_cleans_sources() {
        let dir = crate::storage::temp_dir("ckpt-merge");
        let vecs = random_vectors(1_200, 4, 42);
        let (before_len, gen) = {
            let mut idx = SegmentedIndex::open_or_create(
                dir.clone(),
                4,
                Metric::L2,
                HnswParams::default(),
                100,
            )
            .unwrap();
            idx.set_max_segments(4);
            for (i, v) in vecs.iter().enumerate() {
                idx.insert_shared(VectorId(i as u64), v).unwrap();
            }
            idx.checkpoint().unwrap();
            // the merges must have kept the ceiling
            idx.wait_for_merge(); // 9a-1: merge arka planda, bekle
            assert!(idx.shape().0 <= 4);
            let g = idx.checkpoint().unwrap();
            (idx.len(), g)
        };
        // GC: no segment file absent from the manifest may remain
        let manifest = Manifest::read(&dir).unwrap().unwrap();
        assert_eq!(manifest.generation, gen);
        let on_disk: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| {
                let n = e.unwrap().file_name().to_string_lossy().to_string();
                n.starts_with("segment-").then_some(n)
            })
            .collect();
        assert_eq!(
            on_disk.len(),
            manifest.segments.len(),
            "an orphaned segment file remained: {on_disk:?}"
        );
        let idx =
            SegmentedIndex::open_or_create(dir, 4, Metric::L2, HnswParams::default(), 100).unwrap();
        assert_eq!(idx.len(), before_len);
    }

    #[test]
    fn checkpoint_without_storage_dir_is_error() {
        let idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 100);
        assert!(idx.checkpoint().is_err());
    }

    // ---- Segment ceiling / merge tests ----

    #[test]
    fn merge_guard_enforces_ceiling() {
        let vecs = random_vectors(1_200, 8, 42);
        let mut idx = SegmentedIndex::new(8, Metric::L2, HnswParams::default(), 100);
        idx.set_max_segments(4);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        // 9a-1: merging now runs in the background. Because sealing is faster
        // than merging, the segment count can TEMPORARILY exceed the ceiling; once
        // writing stops the worker brings it back down. So the ceiling check waits
        // first.
        idx.wait_for_merge();
        let (n_seg, _) = idx.shape();
        assert!(
            n_seg <= 4,
            "the ceiling must hold once merging finishes: {n_seg}"
        );
        assert_eq!(idx.len(), 1_200, "the merge lost records");
        // correctness: agreement with the exact reference
        let queries = random_vectors(20, 8, 43);
        let mut hits = 0;
        for q in &queries {
            let truth: Vec<_> = exact_top_k(&vecs, q, 10, Metric::L2)
                .iter()
                .map(|r| r.id)
                .collect();
            hits += idx
                .search_shared(q, 10)
                .iter()
                .filter(|r| truth.contains(&r.id))
                .count();
        }
        assert!(
            hits as f64 / 200.0 >= 0.95,
            "recall after merge: {hits}/200"
        );
    }

    #[test]
    fn merge_drops_tombstones_and_preserves_reinserts() {
        let vecs = random_vectors(600, 4, 42);
        let mut idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 100);
        idx.set_max_segments(3);
        for (i, v) in vecs.iter().take(300).enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        // delete records that landed in segments + re-insert one with a new vector
        idx.delete_shared(VectorId(5)).unwrap();
        idx.delete_shared(VectorId(50)).unwrap();
        idx.insert_shared(VectorId(5), &[9.0; 4]).unwrap();
        // insert enough to press against the ceiling → merges are triggered
        for (i, v) in vecs.iter().enumerate().skip(300) {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        idx.wait_for_merge(); // merge arka planda (9a-1)
        let (n_seg, _) = idx.shape();
        assert!(n_seg <= 3);
        assert_eq!(idx.len(), 599); // 600 - 1 permanent deletion
                                    // the deleted id does not come back; the
                                    // re-inserted one returns with its new vector
        let all: Vec<_> = idx
            .search_shared(&[9.0; 4], 599)
            .iter()
            .map(|r| r.id)
            .collect();
        assert!(!all.contains(&VectorId(50)));
        assert_eq!(idx.search_shared(&[9.0; 4], 1)[0].id, VectorId(5));
    }

    /// Acceptance criterion (DECISIONS #31): for queries with a Range, the arm
    /// chosen must agree with the arm that the true cardinality would select. The
    /// bounded-counting design guarantees this structurally at the scan boundary;
    /// the test documents it anyway.
    #[test]
    fn range_planner_arm_matches_oracle() {
        let vecs = random_vectors(2_000, 4, 42);
        let idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 500);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert_with_meta(
                VectorId(i as u64),
                v,
                [("v".to_string(), MetaValue::Int(i as i64))].into(),
            )
            .unwrap();
        }
        let k = 10;
        let n = 2_000usize;
        let scan_limit =
            (idx.planner.scan_factor * k).max((idx.planner.scan_fraction * n as f64) as usize);
        for m in [1usize, 50, 160, 161, 300, 800, 1500] {
            let f = Filter {
                must: vec![Predicate::Range {
                    key: "v".into(),
                    min: 0.0,
                    max: (m - 1) as f64,
                }],
            };
            let oracle = if m <= scan_limit { "scan" } else { "post" };
            assert_eq!(idx.debug_plan_arm(&f, k), oracle, "m={m}");
            // and the results are correct against the exact reference
            let allow = |id: VectorId| (id.0 as usize) < m;
            let q = &vecs[m / 2];
            let truth: Vec<_> = {
                let mut bf = BruteForceIndex::new(4, Metric::L2);
                for (i, v) in vecs.iter().enumerate() {
                    bf.insert(VectorId(i as u64), v).unwrap();
                }
                bf.search_filtered(q, k, &allow)
                    .iter()
                    .map(|r| r.id)
                    .collect()
            };
            let got: Vec<_> = idx.search_filtered(q, k, &f).iter().map(|r| r.id).collect();
            let hit = got.iter().filter(|id| truth.contains(id)).count();
            assert!(hit * 10 >= truth.len() * 9, "m={m}: {hit}/{}", truth.len());
        }
        // zero matches in the Range → empty
        let f_empty = Filter {
            must: vec![Predicate::Range {
                key: "v".into(),
                min: 1e9,
                max: 2e9,
            }],
        };
        assert!(idx
            .search_filtered(&vecs[0].clone(), k, &f_empty)
            .is_empty());
    }

    /// Posting-list'ler her mutasyon dizisinden sonra metadata deposuyla
    /// must stay exactly consistent (the planner's estimate rests on it).
    #[test]
    fn postings_consistent_after_insert_delete_reinsert() {
        let vecs = random_vectors(200, 4, 42);
        let idx = SegmentedIndex::new(4, Metric::L2, HnswParams::default(), 80);
        for (i, v) in vecs.iter().enumerate() {
            let meta: Metadata = [("g".to_string(), MetaValue::Int((i % 5) as i64))].into();
            idx.insert_with_meta(VectorId(i as u64), v, meta).unwrap();
        }
        for i in (0..200).step_by(3) {
            idx.delete_shared(VectorId(i)).unwrap();
        }
        // a few re-insertions, with a different group
        for i in (0..30).step_by(3) {
            idx.insert_with_meta(
                VectorId(i),
                &vecs[i as usize],
                [("g".to_string(), MetaValue::Int(99))].into(),
            )
            .unwrap();
        }
        // recount and compare
        let meta_store = idx.metadata.read().unwrap();
        let postings = idx.postings.read().unwrap();
        for ((key, mk), list) in postings.iter() {
            let mut recount: Vec<VectorId> = meta_store
                .iter()
                .filter(|(_, m)| m.get(key).is_some_and(|v| v.key() == *mk))
                .map(|(id, _)| id)
                .collect();
            recount.sort();
            // Sortedness is verified too: binary search depends on it.
            assert_eq!(*list, recount, "posting list inconsistent: {key}/{mk:?}");
        }
        // the estimate equals the true match count (exact for a single Eq)
        let f = Filter {
            must: vec![Predicate::Eq {
                key: "g".into(),
                value: MetaValue::Int(99),
            }],
        };
        let (est, cands) = idx.estimate(&f).unwrap();
        assert_eq!(est, 10);
        assert_eq!(cands.len(), 10);
    }

    /// Stress test: many readers + a single writer. While the writer inserts and
    /// deletes, the readers search continuously; nothing may panic and the results
    /// must obey the basic consistency rules (no duplicate ids, no NaN, never more
    /// than k).
    #[test]
    fn stress_concurrent_readers_single_writer() {
        let dim = 16;
        let idx = Arc::new(SegmentedIndex::new(
            dim,
            Metric::L2,
            HnswParams {
                ef_construction: 40, // in a stress test build speed matters more
                // than graph quality
                ..Default::default()
            },
            500,
        ));
        let vecs = random_vectors(4_000, dim, 42);
        // initial load
        for (i, v) in vecs.iter().take(1_000).enumerate() {
            idx.insert_shared(VectorId(i as u64), v).unwrap();
        }
        let stop = AtomicBool::new(false);
        let queries = random_vectors(50, dim, 43);

        std::thread::scope(|s| {
            // 4 okuyucu
            for t in 0..4 {
                let idx = &idx;
                let stop = &stop;
                let queries = &queries;
                s.spawn(move || {
                    let mut iters = 0usize;
                    while !stop.load(Ordering::Relaxed) {
                        let q = &queries[(iters + t) % queries.len()];
                        let res = idx.search_shared(q, 10);
                        assert!(res.len() <= 10);
                        let mut seen = HashSet::new();
                        for r in &res {
                            assert!(!r.distance.is_nan());
                            assert!(seen.insert(r.id), "duplicate id in the results");
                        }
                        // results must be in ascending distance order
                        for w in res.windows(2) {
                            assert!(w[0].distance <= w[1].distance);
                        }
                        iters += 1;
                    }
                    assert!(iters > 0);
                });
            }
            // a single writer: 3,000 inserts (triggering 5+ sealings) plus
            // intermittent deletions
            for (i, v) in vecs.iter().enumerate().skip(1_000) {
                idx.insert_shared(VectorId(i as u64), v).unwrap();
                if i % 7 == 0 {
                    // delete a previously inserted id
                    let victim = VectorId((i / 2) as u64);
                    let _ = idx.delete_shared(victim); // NotFound if already gone: fine
                }
            }
            // 9a-2: since the write path no longer waits for sealing, 2,000
            // inserts finish in milliseconds and the test used to end before the
            // readers did any work at all (iters == 0). Leave them a window.
            std::thread::sleep(std::time::Duration::from_millis(50));
            stop.store(true, Ordering::Relaxed);
        });

        idx.wait_for_background();
        let (n_seg, _) = idx.shape();
        assert!(n_seg >= 3, "sealing was never triggered: {n_seg}");
        // after the writer finishes, searching is deterministic and healthy
        let res = idx.search_shared(&queries[0], 10);
        assert_eq!(res.len(), 10);
    }
}
