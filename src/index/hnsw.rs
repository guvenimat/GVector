//! HNSW indeksi — Malkov & Yashunin (2016), "Efficient and robust approximate
//! nearest neighbor search using Hierarchical Navigable Small World graphs".
//!
//! Representation: no Rc/RefCell; every node is a `usize` slot and adjacency is
//! `links[slot][level] = Vec<usize>`. This removes all borrow-checker friction
//! and makes serialization (phase 3) trivial.
//!
//! Algorithm map (using the numbering from the paper):
//! - Algorithm 1 (INSERT): `insert`
//! - Algorithm 2 (SEARCH-LAYER): `search_layer`
//! - Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC): `select_neighbors_heuristic`
//! - Algorithm 5 (K-NN-SEARCH): `search`

use crate::distance::{normalize, Metric};
use crate::index::{IndexError, VectorIndex};
use crate::types::{SearchResult, VectorId};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// A (distance, slot) pair ordered by distance via total_cmp.
/// The `quant` module uses the same candidate structure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Cand {
    pub(crate) dist: f32,
    pub(crate) slot: usize,
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .total_cmp(&other.dist)
            .then_with(|| self.slot.cmp(&other.slot))
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HnswParams {
    /// Target neighbour count on the upper layers (M in the paper).
    pub m: usize,
    /// Neighbour limit for the base layer (0); the paper suggests 2M.
    pub m_max0: usize,
    /// Search width during construction (efConstruction).
    pub ef_construction: usize,
    /// Base-layer width at query time (ef).
    pub ef_search: usize,
    /// Seed for level-assignment randomness (reproducibility).
    pub seed: u64,
    /// When the tombstone ratio exceeds this threshold, delete triggers an
    /// automatic compaction.
    pub tombstone_threshold: f64,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max0: 32,
            ef_construction: 200,
            ef_search: 50,
            seed: crate::dataset::DEFAULT_SEED,
            tombstone_threshold: 0.3,
        }
    }
}

/// Traversal statistics of a filtered search (for measurement, and later as a
/// planner signal). A collapsing `admitted/visited` ratio is the signature of
/// the "silent recall decline" that occurs without the fallback ever firing.
#[derive(Debug, Clone, Copy, Default)]
pub struct FilterSearchStats {
    /// Number of nodes visited on the base layer.
    pub visited: usize,
    /// Number of candidates admitted into the result set (before eviction).
    pub admitted: usize,
    /// Did the graph search find fewer than k results and fall back to a
    /// linear scan?
    pub fallback_used: bool,
    /// Was the search cut short because the visit budget ran out? (A live
    /// detector for the pathological case where the admit/visit ratio collapses
    /// — the "clustered × distant query" cell in the measurements. When cut
    /// short, the fallback scan takes over.)
    pub budget_exhausted: bool,
}

/// Where the vector data lives: an owned in-memory block, or a region lazily
/// mapped from disk. The mmap path is not writable; on the first insert the
/// data is copied into an owned Vec (copy-on-write).
enum VectorStorage {
    Owned(Vec<f32>),
    /// Not constructed for now: opening with memmap2 requires unsafe and the
    /// crate is compiled with deny(unsafe_code); if that is ever lifted, lazy
    /// loading will use this.
    #[allow(dead_code)]
    Mmap {
        map: memmap2::Mmap,
        /// Byte offset of the f32 data within the file (guaranteed 4-aligned).
        offset: usize,
        /// Number of f32 elements.
        len: usize,
    },
}

impl VectorStorage {
    #[inline]
    fn as_slice(&self) -> &[f32] {
        match self {
            VectorStorage::Owned(v) => v,
            // cast_slice verifies alignment at runtime; this is safe because
            // we write the offset 4-aligned and the mmap base is page-aligned.
            VectorStorage::Mmap { map, offset, len } => {
                bytemuck::cast_slice(&map[*offset..*offset + *len * 4])
            }
        }
    }

    /// Write access: if backed by mmap, first convert to an owned copy.
    fn to_owned_mut(&mut self) -> &mut Vec<f32> {
        if let VectorStorage::Mmap { .. } = self {
            *self = VectorStorage::Owned(self.as_slice().to_vec());
        }
        match self {
            VectorStorage::Owned(v) => v,
            VectorStorage::Mmap { .. } => unreachable!("converted above"),
        }
    }
}

pub struct HnswIndex {
    params: HnswParams,
    metric: Metric,
    dim: usize,
    /// mL = 1/ln(M): the level distribution factor (the optimum from §4.1 of
    /// the paper).
    ml: f64,
    /// Vectors in one contiguous block, slot-major.
    storage: VectorStorage,
    ids: Vec<VectorId>,
    slot_of: HashMap<VectorId, usize>,
    /// links[slot][level] = list of neighbour slots. `links[slot].len()-1` is
    /// the node's top level.
    links: Vec<Vec<Vec<usize>>>,
    /// The graph entry point (the node with the highest level).
    entry: Option<usize>,
    /// Tombstone flags: a deleted node is still TRAVERSED in the graph (it goes
    /// on serving as a bridge for connectivity) but never enters the results.
    /// Actual cleanup happens during compaction.
    deleted: Vec<bool>,
    deleted_count: usize,
    rng: StdRng,
}

impl HnswIndex {
    pub fn new(dim: usize, metric: Metric, params: HnswParams) -> Self {
        let ml = 1.0 / (params.m as f64).ln();
        Self {
            rng: StdRng::seed_from_u64(params.seed),
            params,
            metric,
            dim,
            ml,
            storage: VectorStorage::Owned(Vec::new()),
            ids: Vec::new(),
            slot_of: HashMap::new(),
            links: Vec::new(),
            entry: None,
            deleted: Vec::new(),
            deleted_count: 0,
        }
    }

    pub fn params(&self) -> &HnswParams {
        &self.params
    }

    /// Adjusting ef_search after the fact is needed for parameter sweeps; it
    /// does not affect the graph, only the query width.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.params.ef_search = ef;
    }

    #[inline]
    fn vector_at(&self, slot: usize) -> &[f32] {
        &self.storage.as_slice()[slot * self.dim..(slot + 1) * self.dim]
    }

    #[inline]
    fn dist_to(&self, query: &[f32], slot: usize) -> f32 {
        self.metric.distance(query, self.vector_at(slot))
    }

    /// Exponential level assignment: floor(-ln(U) * mL). The U=0 lower bound is
    /// clamped.
    fn random_level(&mut self) -> usize {
        let u: f64 = self.rng.gen_range(f64::MIN_POSITIVE..1.0);
        (-u.ln() * self.ml).floor() as usize
    }

    /// Algorithm 2 — SEARCH-LAYER: a greedy best-first search of width ef on
    /// `level`, starting from `entry_points`. Returns ef results in ascending
    /// distance order.
    ///
    /// `exclude_deleted`: when true, tombstoned nodes are still TRAVERSED (their
    /// neighbours are explored — they are needed as connectivity bridges) but
    /// are not admitted into the result set. False during construction: a new
    /// node may link to tombstones too, and compaction will clear them wholesale
    /// anyway.
    ///
    /// `filter`: the metadata filter works on the same principle — a
    /// non-matching node is traversed (connectivity) but never enters the
    /// results. None = no filter.
    ///
    /// Returns: (results, number of nodes visited, number of candidates admitted
    /// into the result set). The counters are just two increments — the
    /// production path is instrumented for free; a collapsing admit/visit ratio
    /// in a filtered search is the signature of the "silent recall decline".
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn search_layer(
        &self,
        query: &[f32],
        entry_points: &[usize],
        ef: usize,
        level: usize,
        exclude_deleted: bool,
        filter: Option<&dyn Fn(usize) -> bool>,
        visited_budget: Option<usize>,
    ) -> (Vec<Cand>, usize, usize) {
        // visited: one flag per slot. A Vec<bool> rather than a HashSet: even at
        // n=100K that is a single 100KB allocation, with no per-branch hashing
        // cost.
        let mut visited = vec![false; self.links.len()];
        let mut visited_count = 0usize;
        let mut admitted_count = 0usize;
        // candidates: en YAKIN tepede (min-heap, Reverse ile).
        let mut candidates: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        // results: the FARTHEST on top (a max-heap) — so the worst can be evicted.
        let mut results: BinaryHeap<Cand> = BinaryHeap::new();

        for &ep in entry_points {
            if visited[ep] {
                continue;
            }
            visited[ep] = true;
            visited_count += 1;
            let c = Cand {
                dist: self.dist_to(query, ep),
                slot: ep,
            };
            candidates.push(Reverse(c));
            let admissible = !(exclude_deleted && self.deleted[ep]) && filter.is_none_or(|f| f(ep));
            if admissible {
                results.push(c);
                admitted_count += 1;
            }
        }

        while let Some(Reverse(cur)) = candidates.pop() {
            // Budget: when the admit ratio collapses in a filtered search, the
            // traversal can spread across the whole graph; the budget cuts that
            // off and the caller switches to a scan.
            if visited_budget.is_some_and(|b| visited_count >= b) {
                break;
            }
            // Early exit: if even the nearest candidate is farther than the
            // worst of the result set, nothing better can be found on this layer
            // (the stopping condition from the paper).
            if let Some(worst) = results.peek() {
                if cur.dist > worst.dist && results.len() >= ef {
                    break;
                }
            }
            for &nb in &self.links[cur.slot][level] {
                if visited[nb] {
                    continue;
                }
                visited[nb] = true;
                visited_count += 1;
                let d = self.dist_to(query, nb);
                let within =
                    results.len() < ef || results.peek().is_none_or(|worst| d < worst.dist);
                if within {
                    let c = Cand { dist: d, slot: nb };
                    candidates.push(Reverse(c));
                    // A tombstoned / filtered-out node is traversed but never
                    // enters the results.
                    let admissible =
                        !(exclude_deleted && self.deleted[nb]) && filter.is_none_or(|f| f(nb));
                    if admissible {
                        results.push(c);
                        admitted_count += 1;
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }
        let mut out = results.into_vec();
        out.sort();
        (out, visited_count, admitted_count)
    }

    /// Algorithm 4 — SELECT-NEIGHBORS-HEURISTIC.
    ///
    /// Instead of the naive "nearest M": a candidate is DISCARDED if it is
    /// closer to any already-selected neighbour than it is to the query. This
    /// prunes redundant edges within a cluster and preserves bridge edges
    /// BETWEEN clusters — the graph's connectivity (and therefore its recall)
    /// depends on it.
    ///
    /// keepPrunedConnections=true behaviour: top up to M with the nearest of the
    /// discarded ones (the optional step in the paper, so no node is left with a
    /// low degree).
    fn select_neighbors_heuristic(&self, candidates: &[Cand], m: usize) -> Vec<usize> {
        let mut selected: Vec<Cand> = Vec::with_capacity(m);
        let mut pruned: Vec<Cand> = Vec::new();
        for &c in candidates {
            if selected.len() >= m {
                break;
            }
            let c_vec = self.vector_at(c.slot);
            // is c closer to one of the selected than it is to the query?
            let dominated = selected
                .iter()
                .any(|s| self.metric.distance(c_vec, self.vector_at(s.slot)) < c.dist);
            if dominated {
                pruned.push(c);
            } else {
                selected.push(c);
            }
        }
        // keepPrunedConnections: fill the remaining quota with the nearest of
        // the discarded
        for c in pruned {
            if selected.len() >= m {
                break;
            }
            selected.push(c);
        }
        selected.into_iter().map(|c| c.slot).collect()
    }

    /// The neighbour limit at a level: the base layer is denser (paper:
    /// M_max0 = 2M).
    #[inline]
    fn max_links(&self, level: usize) -> usize {
        if level == 0 {
            self.params.m_max0
        } else {
            self.params.m
        }
    }

    /// If `node`'s neighbour list at `level` exceeds the limit, prune it with
    /// the heuristic.
    fn shrink_links(&mut self, node: usize, level: usize) {
        let limit = self.max_links(level);
        if self.links[node][level].len() <= limit {
            return;
        }
        // Candidate list: the current neighbours with their distance to the
        // node, in ascending order.
        let mut cands: Vec<Cand> = self.links[node][level]
            .iter()
            .map(|&nb| Cand {
                dist: self
                    .metric
                    .distance(self.vector_at(node), self.vector_at(nb)),
                slot: nb,
            })
            .collect();
        cands.sort();
        self.links[node][level] = self.select_neighbors_heuristic(&cands, limit);
    }

    /// Runs the query at the given ef width and returns a list of
    /// SearchResults (for use in parameter sweeps without set_ef_search).
    pub fn search_with_ef(&self, query: &[f32], k: usize, ef: usize) -> Vec<SearchResult> {
        let Some(entry) = self.entry else {
            return Vec::new();
        };
        if k == 0 {
            return Vec::new();
        }
        let normalized_query;
        let query: &[f32] = if self.metric.requires_normalization() {
            normalized_query = crate::distance::normalized(query);
            &normalized_query
        } else {
            query
        };
        // Greedy descent through the upper layers (ef=1): jump to the nearest
        // node on each layer.
        let top = self.links[entry].len() - 1;
        let mut ep = entry;
        for level in (1..=top).rev() {
            // While descending, a tombstone is a valid stop: it is only guiding
            // the way.
            ep = self
                .search_layer(query, &[ep], 1, level, false, None, None)
                .0[0]
                .slot;
        }
        // A wide search on the base layer; ef must be at least k or fewer than k
        // results come out. Tombstones are excluded from the results here.
        let ef = ef.max(k);
        let (found, _, _) = self.search_layer(query, &[ep], ef, 0, true, None, None);
        found
            .into_iter()
            .take(k)
            .map(|c| SearchResult::new(self.ids[c.slot], c.dist))
            .collect()
    }

    /// Filtered search: only records for which `allow(id)` returns true can be
    /// candidates. Non-matching nodes are still used as traversal bridges (see
    /// the meta module).
    ///
    /// Correctness guarantee: if the graph search finds fewer than k results
    /// (because a highly selective filter left few matches in the traversed
    /// region), it falls back to a filtered linear scan over all live records —
    /// slow but complete.
    pub fn search_filtered_with_ef(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allow: &dyn Fn(VectorId) -> bool,
    ) -> Vec<SearchResult> {
        self.search_filtered_stats(query, k, ef, allow, None).0
    }

    /// The instrumented form of `search_filtered_with_ef` — the production path
    /// wraps this function, so when extra output is needed the signature stays
    /// stable and a field is simply added to `FilterSearchStats`.
    pub fn search_filtered_stats(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        allow: &dyn Fn(VectorId) -> bool,
        visited_budget: Option<usize>,
    ) -> (Vec<SearchResult>, FilterSearchStats) {
        let mut stats = FilterSearchStats::default();
        let Some(entry) = self.entry else {
            return (Vec::new(), stats);
        };
        if k == 0 {
            return (Vec::new(), stats);
        }
        let normalized_query;
        let query: &[f32] = if self.metric.requires_normalization() {
            normalized_query = crate::distance::normalized(query);
            &normalized_query
        } else {
            query
        };
        let top = self.links[entry].len() - 1;
        let mut ep = entry;
        for level in (1..=top).rev() {
            ep = self
                .search_layer(query, &[ep], 1, level, false, None, None)
                .0[0]
                .slot;
        }
        let ef = ef.max(k);
        let slot_allow = |slot: usize| allow(self.ids[slot]);
        let (found, visited, admitted) =
            self.search_layer(query, &[ep], ef, 0, true, Some(&slot_allow), visited_budget);
        stats.visited = visited;
        stats.admitted = admitted;
        stats.budget_exhausted = visited_budget.is_some_and(|b| visited >= b);
        // If the budget ran out, return the partial results AS IS — the caller
        // decides what to do next (a posting-list scan, say); running the O(n)
        // fallback here would defeat the purpose of the budget.
        if stats.budget_exhausted {
            let out = found
                .into_iter()
                .take(k)
                .map(|c| SearchResult::new(self.ids[c.slot], c.dist))
                .collect();
            return (out, stats);
        }
        if found.len() >= k {
            let out = found
                .into_iter()
                .take(k)
                .map(|c| SearchResult::new(self.ids[c.slot], c.dist))
                .collect();
            return (out, stats);
        }
        // Fallback: the selectivity outran the traversed region — linear scan.
        stats.fallback_used = true;
        let mut all: Vec<SearchResult> = (0..self.ids.len())
            .filter(|&s| !self.deleted[s] && slot_allow(s))
            .map(|s| SearchResult::new(self.ids[s], self.dist_to(query, s)))
            .collect();
        all.sort();
        all.truncate(k);
        (all, stats)
    }

    /// Total index memory including graph edges (bytes).
    pub fn memory_bytes(&self) -> (usize, usize) {
        let vec_bytes = self.storage.as_slice().len() * 4 + self.ids.capacity() * 8;
        let link_bytes: usize = self
            .links
            .iter()
            .map(|levels| {
                std::mem::size_of::<Vec<usize>>()
                    + levels
                        .iter()
                        .map(|l| std::mem::size_of::<Vec<usize>>() + l.capacity() * 8)
                        .sum::<usize>()
            })
            .sum();
        (vec_bytes, link_bytes)
    }
}

impl VectorIndex for HnswIndex {
    /// Algorithm 1 — INSERT.
    fn insert(&mut self, id: VectorId, vector: &[f32]) -> Result<(), IndexError> {
        if vector.len() != self.dim {
            return Err(IndexError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        if self.slot_of.contains_key(&id) {
            return Err(IndexError::DuplicateId(id));
        }
        let slot = self.ids.len();
        let data = self.storage.to_owned_mut();
        data.extend_from_slice(vector);
        if self.metric.requires_normalization() {
            let start = slot * self.dim;
            normalize(&mut data[start..start + self.dim]);
        }
        self.ids.push(id);
        self.slot_of.insert(id, slot);
        self.deleted.push(false);

        let level = self.random_level();
        self.links.push(vec![Vec::new(); level + 1]);

        let Some(entry) = self.entry else {
            // The first element: it becomes the entry point directly.
            self.entry = Some(slot);
            return Ok(());
        };

        let query = self.vector_at(slot).to_vec(); // a copy, to separate borrows
        let top = self.links[entry].len() - 1;
        let mut ep = entry;

        // Phase 1: on the layers ABOVE the new node's level, only greedy
        // descent — no edges will be added there, we are merely getting closer.
        for lc in ((level + 1)..=top).rev() {
            ep = self.search_layer(&query, &[ep], 1, lc, false, None, None).0[0].slot;
        }

        // Phase 2: on every layer from level down to 0, search at
        // ef_construction width, pick neighbours with the heuristic, link both
        // ways, and prune neighbours that exceed the limit.
        let mut eps = vec![ep];
        for lc in (0..=level.min(top)).rev() {
            let (found, _, _) = self.search_layer(
                &query,
                &eps,
                self.params.ef_construction,
                lc,
                false,
                None,
                None,
            );
            let neighbors = self.select_neighbors_heuristic(&found, self.params.m);
            for &nb in &neighbors {
                self.links[slot][lc].push(nb);
                self.links[nb][lc].push(slot);
                self.shrink_links(nb, lc);
            }
            // Descend to the next layer from everything found on this one (the
            // paper carries W down).
            eps = found.into_iter().map(|c| c.slot).collect();
        }

        // If the new node is higher than everyone else, the entry point changes
        // hands.
        if level > top {
            self.entry = Some(slot);
        }
        Ok(())
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.search_with_ef(query, k, self.params.ef_search)
    }

    /// Tombstone-based deletion: the node stays in the graph (serving as a
    /// bridge) but drops out of the results. Since it is removed from
    /// `slot_of`, the same id can be inserted again.
    fn delete(&mut self, id: VectorId) -> Result<(), IndexError> {
        let slot = self.slot_of.remove(&id).ok_or(IndexError::NotFound(id))?;
        self.deleted[slot] = true;
        self.deleted_count += 1;
        // Critical case: the entry point was deleted. A tombstone could keep
        // working as a waypoint, but having every search start from a dead node
        // is both confusing and awkward for compaction — make the highest-level
        // live node the new entry.
        if self.entry == Some(slot) {
            self.pick_new_entry();
        }
        let ratio = self.deleted_count as f64 / self.ids.len() as f64;
        if ratio >= self.params.tombstone_threshold {
            self.compact();
        }
        Ok(())
    }

    fn len(&self) -> usize {
        self.ids.len() - self.deleted_count
    }
}

impl HnswIndex {
    /// Makes the highest-level live node the entry point. If all of them are
    /// deleted, entry becomes None (search returns empty and insert rebuilds
    /// from scratch).
    fn pick_new_entry(&mut self) {
        self.entry = (0..self.ids.len())
            .filter(|&s| !self.deleted[s])
            .max_by_key(|&s| self.links[s].len());
    }

    /// Rebuilds the index from its live elements regardless of the tombstone
    /// ratio: vector data, graph edges and tombstone memory are genuinely
    /// released. It costs O(n · insert) — which is why it is triggered by a
    /// threshold.
    pub fn compact(&mut self) {
        let mut fresh = HnswIndex::new(self.dim, self.metric, self.params.clone());
        for slot in 0..self.ids.len() {
            if !self.deleted[slot] {
                // Cosine: the stored vector is already normalized; normalizing
                // again is idempotent.
                fresh
                    .insert(self.ids[slot], self.vector_at(slot))
                    .expect("a compaction insert cannot fail");
            }
        }
        *self = fresh;
    }

    // Quantization (phase 6) reuses the graph itself; these accessors are
    // crate-visible so the `quant` module can produce a frozen copy.
    pub(crate) fn graph_links(&self) -> &[Vec<Vec<usize>>] {
        &self.links
    }
    pub(crate) fn graph_ids(&self) -> &[VectorId] {
        &self.ids
    }
    pub(crate) fn graph_entry(&self) -> Option<usize> {
        self.entry
    }
    pub(crate) fn graph_deleted(&self) -> &[bool] {
        &self.deleted
    }
    pub(crate) fn raw_vectors(&self) -> &[f32] {
        self.storage.as_slice()
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Is this id live in this index (excluding tombstoned ones)?
    pub fn contains(&self, id: VectorId) -> bool {
        self.slot_of.contains_key(&id)
    }

    /// Live (id, vector) pairs — read by the segment merge rebuild; tombstoned
    /// ones are skipped (a merge is a natural compaction).
    pub fn live_entries(&self) -> impl Iterator<Item = (VectorId, &[f32])> {
        self.ids
            .iter()
            .enumerate()
            .filter(|(slot, _)| !self.deleted[*slot])
            .map(|(slot, &id)| (id, self.vector_at(slot)))
    }

    /// The (possibly normalized) vector of a live record.
    /// The planner's scan arm computes distances directly from an id list.
    pub fn vector_of(&self, id: VectorId) -> Option<&[f32]> {
        self.slot_of.get(&id).map(|&s| self.vector_at(s))
    }

    /// The tombstone ratio (for tests and observability).
    pub fn tombstone_ratio(&self) -> f64 {
        if self.ids.is_empty() {
            0.0
        } else {
            self.deleted_count as f64 / self.ids.len() as f64
        }
    }
}

// ---------------------------------------------------------------------------
// Persistence (phase 3)
//
// File layout (all numbers little-endian):
//   [0..4)   magic  b"GVDB"
//   [4..8)   format version (u32) = 1
//   [8..16)  meta length (u64)
//   [16..16+meta_len)  bincode(Meta)
//   ...pad (aligns the end of meta to 4 bytes so the f32 section can be cast)
//   [data_off..data_off+n*dim*4)  raw f32 vector data
//   [last 4 bytes)  crc32 (of everything before it)
//
// The vector section is kept OUTSIDE meta so that the file can be opened with
// memmap2 and that region used without copying (lazy load). Meta (the graph and
// ids) is deserialized into memory in every case — graph traversal is
// random-access and small anyway; the vector data is what takes up the space.
// ---------------------------------------------------------------------------

const MAGIC: [u8; 4] = *b"GVDB";
const FORMAT_VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum PersistError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("bozuk dosya: {0}")]
    Corrupt(String),
    #[error("unsupported format version: {0} (this build reads {FORMAT_VERSION})")]
    UnsupportedVersion(u32),
    #[error("serialization error: {0}")]
    Encode(#[from] bincode::Error),
}

/// The graph metadata written to disk. The vector data is deliberately not here.
#[derive(serde::Serialize, serde::Deserialize)]
struct Meta {
    params: HnswParams,
    metric: Metric,
    dim: u64,
    n: u64,
    ids: Vec<VectorId>,
    links: Vec<Vec<Vec<u64>>>,
    entry: Option<u64>,
    /// Tombstone flags (phase 4). Because of ids that were deleted and
    /// re-inserted, the same id can appear in two slots within `ids`; the live
    /// one is unique.
    deleted: Vec<bool>,
}

fn corrupt(msg: impl Into<String>) -> PersistError {
    PersistError::Corrupt(msg.into())
}

impl HnswIndex {
    pub fn save(&self, path: &std::path::Path) -> Result<(), PersistError> {
        let buf = self.to_bytes()?;
        crate::storage::write_file_durable(path, &buf)?;
        Ok(())
    }

    /// The serialized body (magic + version + meta + f32 section + CRC).
    /// The segment snapshot uses this: it needs the bytes in order to compute
    /// the CRC in memory and record it in the manifest (rather than reading the
    /// file back).
    pub fn to_bytes(&self) -> Result<Vec<u8>, PersistError> {
        let meta = Meta {
            params: self.params.clone(),
            metric: self.metric,
            dim: self.dim as u64,
            n: self.ids.len() as u64,
            ids: self.ids.clone(),
            links: self
                .links
                .iter()
                .map(|ls| {
                    ls.iter()
                        .map(|l| l.iter().map(|&s| s as u64).collect())
                        .collect()
                })
                .collect(),
            entry: self.entry.map(|e| e as u64),
            deleted: self.deleted.clone(),
        };
        let meta_bytes = bincode::serialize(&meta)?;

        let mut buf: Vec<u8> = Vec::new();
        buf.extend(MAGIC);
        buf.extend(FORMAT_VERSION.to_le_bytes());
        buf.extend((meta_bytes.len() as u64).to_le_bytes());
        buf.extend(&meta_bytes);
        while !buf.len().is_multiple_of(4) {
            buf.push(0); // alignment of the f32 section
        }
        buf.extend_from_slice(bytemuck::cast_slice::<f32, u8>(self.storage.as_slice()));

        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&buf);
        buf.extend(hasher.finalize().to_le_bytes());
        Ok(buf)
    }

    /// Loading from bytes — the fuzz target and the tests share this path.
    /// The vector data is copied (Owned). The returned index is ready to search.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<HnswIndex, PersistError> {
        let (meta, data_range) = Self::parse(bytes)?;
        let data: Vec<f32> = bytes[data_range]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 byte")))
            .collect();
        Self::rebuild(meta, VectorStorage::Owned(data))
    }

    /// Loading from a file.
    ///
    /// The intent of `lazy=true` is to use the vector section without copying,
    /// via memmap2 (VectorStorage::Mmap). But `memmap2::Mmap::map` is an
    /// `unsafe fn` (it is the caller's responsibility that the file does not
    /// change while the mapping is alive) and the crate is compiled with
    /// `#![deny(unsafe_code)]`. Until that is permitted the lazy parameter is
    /// accepted but both paths run the safe full read — the behaviour is
    /// identical, only the saved memory copy is deferred (see DECISIONS.md,
    /// phase 3).
    pub fn load(path: &std::path::Path, _lazy: bool) -> Result<HnswIndex, PersistError> {
        let bytes = std::fs::read(path)?;
        Self::load_from_bytes(&bytes)
    }

    /// Header + crc + bounds validation. On success: (Meta, the range of the
    /// f32 section).
    fn parse(bytes: &[u8]) -> Result<(Meta, std::ops::Range<usize>), PersistError> {
        if bytes.len() < 20 {
            return Err(corrupt("file too short even for the header"));
        }
        if bytes[0..4] != MAGIC {
            return Err(corrupt("magic mismatch (this is not a GVDB file)"));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().expect("4 byte"));
        if version != FORMAT_VERSION {
            return Err(PersistError::UnsupportedVersion(version));
        }
        // Checksum first: if the body is not intact, do not try to interpret
        // the rest.
        let body = &bytes[..bytes.len() - 4];
        let stored_crc = u32::from_le_bytes(bytes[bytes.len() - 4..].try_into().expect("4 byte"));
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(body);
        if hasher.finalize() != stored_crc {
            return Err(corrupt("crc32 mismatch (file corrupted/truncated)"));
        }
        let meta_len = u64::from_le_bytes(bytes[8..16].try_into().expect("8 byte")) as usize;
        let meta_end = 16usize
            .checked_add(meta_len)
            .ok_or_else(|| corrupt("meta_len overflows"))?;
        if meta_end > body.len() {
            return Err(corrupt("meta_len exceeds the file size"));
        }
        let meta: Meta = bincode::deserialize(&bytes[16..meta_end])?;
        let data_off = meta_end.div_ceil(4) * 4;
        let expected = (meta.n as usize)
            .checked_mul(meta.dim as usize)
            .and_then(|x| x.checked_mul(4))
            .ok_or_else(|| corrupt("n*dim overflows"))?;
        if body.len() < data_off || body.len() - data_off != expected {
            return Err(corrupt(format!(
                "the vector section should have been {} bytes, found {}",
                expected,
                body.len().saturating_sub(data_off)
            )));
        }
        Ok((meta, data_off..data_off + expected))
    }

    /// Builds a working index from meta + storage, validating internal
    /// consistency (every slot reference is bounds-checked so fuzzing cannot
    /// crash it).
    fn rebuild(meta: Meta, storage: VectorStorage) -> Result<HnswIndex, PersistError> {
        let n = meta.n as usize;
        let dim = meta.dim as usize;
        if dim == 0 || dim > 1 << 20 {
            return Err(corrupt("implausible dim"));
        }
        if meta.ids.len() != n || meta.links.len() != n {
            return Err(corrupt("ids/links length does not match n"));
        }
        let mut links = Vec::with_capacity(n);
        for ls in &meta.links {
            if ls.is_empty() {
                return Err(corrupt("node has no levels at all"));
            }
            let mut node_levels = Vec::with_capacity(ls.len());
            for level in ls {
                let l: Vec<usize> = level.iter().map(|&s| s as usize).collect();
                if l.iter().any(|&s| s >= n) {
                    return Err(corrupt("neighbour slot out of bounds"));
                }
                node_levels.push(l);
            }
            links.push(node_levels);
        }
        if meta.deleted.len() != n {
            return Err(corrupt("deleted length does not match n"));
        }
        let deleted_count = meta.deleted.iter().filter(|&&d| d).count();
        let entry = match meta.entry {
            Some(e) if (e as usize) < n => Some(e as usize),
            Some(_) => return Err(corrupt("entry point out of bounds")),
            // If every element is a tombstone, entry may legitimately be None.
            None if deleted_count == n => None,
            None => return Err(corrupt("there are live elements but no entry")),
        };
        // Only live slots go into the id map; the id in a tombstoned slot may
        // have been re-inserted, and the live copy is the one that belongs in
        // the map.
        let mut slot_of = HashMap::with_capacity(n - deleted_count);
        for (slot, &id) in meta.ids.iter().enumerate() {
            if meta.deleted[slot] {
                continue;
            }
            if slot_of.insert(id, slot).is_some() {
                return Err(corrupt("duplicate live VectorId"));
            }
        }
        let ml = 1.0 / (meta.params.m.max(2) as f64).ln();
        Ok(HnswIndex {
            // RNG state is not written to disk; after loading, level assignment
            // is re-derived from seed ⊕ n (deterministic, but not identical to
            // the state mid-construction — see DECISIONS.md).
            rng: StdRng::seed_from_u64(meta.params.seed ^ meta.n),
            params: meta.params,
            metric: meta.metric,
            dim,
            ml,
            storage,
            ids: meta.ids,
            slot_of,
            links,
            entry,
            deleted: meta.deleted,
            deleted_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::random_vectors;
    use crate::index::bruteforce::BruteForceIndex;
    use proptest::prelude::*;

    fn build(vecs: &[Vec<f32>], metric: Metric) -> HnswIndex {
        let mut idx = HnswIndex::new(vecs[0].len(), metric, HnswParams::default());
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        idx
    }

    #[test]
    fn empty_index_returns_empty() {
        let idx = HnswIndex::new(4, Metric::L2, HnswParams::default());
        assert!(idx.is_empty());
        assert!(idx.search(&[0.0; 4], 5).is_empty());
    }

    #[test]
    fn single_element() {
        let idx = build(&[vec![1.0, 2.0]], Metric::L2);
        let res = idx.search(&[0.0, 0.0], 1);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].id, VectorId(0));
    }

    #[test]
    fn k_larger_than_len_returns_all() {
        let idx = build(&random_vectors(5, 8, 42), Metric::L2);
        assert_eq!(idx.search(&[0.0; 8], 100).len(), 5);
    }

    #[test]
    fn k_zero_returns_empty() {
        let idx = build(&random_vectors(5, 8, 42), Metric::L2);
        assert!(idx.search(&[0.0; 8], 0).is_empty());
    }

    #[test]
    fn duplicate_vectors_distinct_ids_both_found() {
        let mut vecs = random_vectors(50, 8, 42);
        vecs[10] = vecs[3].clone(); // birebir kopya
        let idx = build(&vecs, Metric::L2);
        let res = idx.search(&vecs[3].clone(), 2);
        let ids: Vec<_> = res.iter().map(|r| r.id).collect();
        assert!(ids.contains(&VectorId(3)) && ids.contains(&VectorId(10)));
    }

    #[test]
    fn duplicate_id_rejected() {
        let mut idx = HnswIndex::new(2, Metric::L2, HnswParams::default());
        idx.insert(VectorId(1), &[1.0, 2.0]).unwrap();
        assert!(matches!(
            idx.insert(VectorId(1), &[3.0, 4.0]),
            Err(IndexError::DuplicateId(_))
        ));
    }

    #[test]
    fn zero_vector_cosine_no_nan() {
        let mut idx = HnswIndex::new(3, Metric::Cosine, HnswParams::default());
        idx.insert(VectorId(0), &[0.0; 3]).unwrap();
        idx.insert(VectorId(1), &[1.0, 0.0, 0.0]).unwrap();
        let res = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(res.len(), 2);
        assert!(res.iter().all(|r| !r.distance.is_nan()));
    }

    #[test]
    fn deterministic_across_builds_same_seed() {
        let vecs = random_vectors(200, 16, 7);
        let a = build(&vecs, Metric::L2);
        let b = build(&vecs, Metric::L2);
        let q = random_vectors(1, 16, 8).pop().unwrap();
        assert_eq!(a.search(&q, 10), b.search(&q, 10));
    }

    #[test]
    fn link_counts_respect_limits() {
        let idx = build(&random_vectors(2_000, 16, 42), Metric::L2);
        for (slot, levels) in idx.links.iter().enumerate() {
            for (level, nbrs) in levels.iter().enumerate() {
                assert!(
                    nbrs.len() <= idx.max_links(level),
                    "slot {slot} level {level}: {} > limit",
                    nbrs.len()
                );
                // Note: edges are NOT GUARANTEED to be bidirectional — when
                // shrink_links prunes one side, the opposite edge remains (the
                // graph is directed; same behaviour as hnswlib). So we only
                // verify that neighbour slots are valid and that the node
                // actually has that level.
                for &nb in nbrs {
                    assert!(nb < idx.links.len());
                    assert!(
                        level < idx.links[nb].len(),
                        "neighbour {nb} does not have level {level}"
                    );
                }
            }
        }
    }

    #[test]
    fn recall_on_small_set_vs_bruteforce() {
        let vecs = random_vectors(2_000, 32, 42);
        let queries = random_vectors(50, 32, 43);
        let hnsw = build(&vecs, Metric::L2);
        let mut bf = BruteForceIndex::new(32, Metric::L2);
        for (i, v) in vecs.iter().enumerate() {
            bf.insert(VectorId(i as u64), v).unwrap();
        }
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let truth: Vec<_> = bf.search(q, 10).iter().map(|r| r.id).collect();
            let got = hnsw.search_with_ef(q, 10, 100);
            hits += got.iter().filter(|r| truth.contains(&r.id)).count();
            total += truth.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.95, "recall {recall} < 0.95");
    }

    // ---- Deletion / compaction tests (phase 4) ----

    /// Parameters with a high threshold so compaction is not triggered.
    fn no_compact_params() -> HnswParams {
        HnswParams {
            tombstone_threshold: 2.0, // asla otomatik tetiklenmez
            ..Default::default()
        }
    }

    fn build_with(vecs: &[Vec<f32>], params: HnswParams) -> HnswIndex {
        let mut idx = HnswIndex::new(vecs[0].len(), Metric::L2, params);
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(VectorId(i as u64), v).unwrap();
        }
        idx
    }

    #[test]
    fn delete_removes_from_results_and_len() {
        let vecs = random_vectors(100, 8, 42);
        let mut idx = build_with(&vecs, no_compact_params());
        idx.delete(VectorId(7)).unwrap();
        assert_eq!(idx.len(), 99);
        let res = idx.search(&vecs[7].clone(), 10);
        assert!(res.iter().all(|r| r.id != VectorId(7)));
        assert_eq!(
            idx.delete(VectorId(7)),
            Err(IndexError::NotFound(VectorId(7)))
        );
    }

    #[test]
    fn delete_entry_point_picks_new_entry_and_search_works() {
        let vecs = random_vectors(500, 8, 42);
        let mut idx = build_with(&vecs, no_compact_params());
        // find and delete the entry point — the critical scenario
        let entry_slot = idx.entry.unwrap();
        let entry_id = idx.ids[entry_slot];
        idx.delete(entry_id).unwrap();
        let new_entry = idx.entry.expect("a new entry should have been chosen");
        assert_ne!(new_entry, entry_slot);
        assert!(!idx.deleted[new_entry]);
        // the new entry must be the highest-level live node
        let max_level = (0..idx.ids.len())
            .filter(|&s| !idx.deleted[s])
            .map(|s| idx.links[s].len())
            .max()
            .unwrap();
        assert_eq!(idx.links[new_entry].len(), max_level);
        // search still works and does not return the deleted id
        let res = idx.search(&vecs[0].clone(), 10);
        assert_eq!(res.len(), 10);
        assert!(res.iter().all(|r| r.id != entry_id));
    }

    #[test]
    fn delete_all_then_reinsert() {
        let vecs = random_vectors(20, 4, 42);
        let mut idx = build_with(&vecs, no_compact_params());
        for i in 0..20 {
            idx.delete(VectorId(i)).unwrap();
        }
        assert_eq!(idx.len(), 0);
        assert!(idx.search(&[0.0; 4], 5).is_empty());
        assert!(idx.entry.is_none());
        // inserting into an emptied index must work like building from scratch
        idx.insert(VectorId(100), &[1.0; 4]).unwrap();
        assert_eq!(idx.search(&[1.0; 4], 1)[0].id, VectorId(100));
    }

    #[test]
    fn deleted_id_can_be_reinserted_with_new_vector() {
        let vecs = random_vectors(50, 4, 42);
        let mut idx = build_with(&vecs, no_compact_params());
        idx.delete(VectorId(5)).unwrap();
        idx.insert(VectorId(5), &[9.0; 4]).unwrap();
        assert_eq!(idx.len(), 50);
        let res = idx.search(&[9.0; 4], 1);
        assert_eq!(res[0].id, VectorId(5));
    }

    #[test]
    fn recall_stays_high_after_20pct_deletion() {
        let vecs = random_vectors(2_000, 16, 42);
        let queries = random_vectors(30, 16, 43);
        let mut idx = build_with(&vecs, no_compact_params());
        let mut bf = BruteForceIndex::new(16, Metric::L2);
        for (i, v) in vecs.iter().enumerate() {
            bf.insert(VectorId(i as u64), v).unwrap();
        }
        // delete every 5th element (20%)
        for i in (0..2_000).step_by(5) {
            idx.delete(VectorId(i)).unwrap();
            bf.delete(VectorId(i)).unwrap();
        }
        let mut hits = 0;
        let mut total = 0;
        for q in &queries {
            let truth: Vec<_> = bf.search(q, 10).iter().map(|r| r.id).collect();
            let got = idx.search_with_ef(q, 10, 100);
            assert_eq!(got.len(), 10, "missing results after deletion");
            hits += got.iter().filter(|r| truth.contains(&r.id)).count();
            total += truth.len();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.95, "recall after deletion {recall} < 0.95");
    }

    #[test]
    fn compaction_triggers_at_threshold_and_frees_memory() {
        let vecs = random_vectors(1_000, 16, 42);
        let mut idx = build_with(
            &vecs,
            HnswParams {
                tombstone_threshold: 0.3,
                ..Default::default()
            },
        );
        let (vec_before, link_before) = idx.memory_bytes();
        // delete up to just below the 30% threshold — compaction must not fire
        for i in 0..299 {
            idx.delete(VectorId(i)).unwrap();
        }
        assert!(idx.tombstone_ratio() > 0.0);
        // a deletion crossing the threshold triggers compaction
        idx.delete(VectorId(299)).unwrap();
        assert_eq!(
            idx.tombstone_ratio(),
            0.0,
            "compaction must leave no tombstones"
        );
        assert_eq!(idx.len(), 700);
        let (vec_after, link_after) = idx.memory_bytes();
        assert!(
            vec_after < vec_before && link_after < link_before,
            "memory did not drop: vec {vec_before}->{vec_after}, link {link_before}->{link_after}"
        );
        // search is healthy after compaction
        let res = idx.search(&vecs[500].clone(), 5);
        assert_eq!(res[0].id, VectorId(500));
    }

    #[test]
    fn persist_roundtrip_with_tombstones() {
        let vecs = random_vectors(200, 8, 42);
        let mut idx = build_with(&vecs, no_compact_params());
        for i in (0..200).step_by(7) {
            idx.delete(VectorId(i)).unwrap();
        }
        // re-insert a deleted id: the duplicate-entry-in-ids scenario
        idx.insert(VectorId(0), &[0.5; 8]).unwrap();
        let loaded = HnswIndex::load_from_bytes(&save_to_bytes(&idx)).unwrap();
        assert_eq!(idx.len(), loaded.len());
        for q in random_vectors(10, 8, 43) {
            assert_eq!(idx.search(&q, 10), loaded.search(&q, 10));
        }
    }

    // ---- Persistence tests (phase 3) ----

    fn save_to_bytes(idx: &HnswIndex) -> Vec<u8> {
        let dir = std::env::temp_dir().join(format!("gvdb-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(format!("t-{:p}.gvdb", idx as *const _));
        idx.save(&path).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        bytes
    }

    #[test]
    fn persist_roundtrip_identical_results() {
        let vecs = random_vectors(1_000, 16, 42);
        let idx = build(&vecs, Metric::L2);
        let loaded = HnswIndex::load_from_bytes(&save_to_bytes(&idx)).unwrap();
        for q in random_vectors(20, 16, 43) {
            assert_eq!(idx.search(&q, 10), loaded.search(&q, 10));
        }
        assert_eq!(idx.len(), loaded.len());
    }

    #[test]
    fn persist_roundtrip_cosine_normalized_data_preserved() {
        let vecs = random_vectors(200, 8, 42);
        let idx = build(&vecs, Metric::Cosine);
        let loaded = HnswIndex::load_from_bytes(&save_to_bytes(&idx)).unwrap();
        for q in random_vectors(5, 8, 43) {
            assert_eq!(idx.search(&q, 5), loaded.search(&q, 5));
        }
    }

    #[test]
    fn persist_empty_index_roundtrip() {
        let idx = HnswIndex::new(4, Metric::L2, HnswParams::default());
        let loaded = HnswIndex::load_from_bytes(&save_to_bytes(&idx)).unwrap();
        assert!(loaded.is_empty());
        assert!(loaded.search(&[0.0; 4], 3).is_empty());
    }

    #[test]
    fn persist_loaded_index_accepts_inserts() {
        let vecs = random_vectors(100, 8, 42);
        let idx = build(&vecs, Metric::L2);
        let mut loaded = HnswIndex::load_from_bytes(&save_to_bytes(&idx)).unwrap();
        loaded.insert(VectorId(999), &[0.5; 8]).unwrap();
        assert_eq!(loaded.len(), 101);
        let res = loaded.search(&[0.5; 8], 1);
        assert_eq!(res[0].id, VectorId(999));
    }

    #[test]
    fn persist_truncated_file_is_error_not_panic() {
        let idx = build(&random_vectors(100, 8, 42), Metric::L2);
        let bytes = save_to_bytes(&idx);
        for cut in [0, 3, 10, bytes.len() / 2, bytes.len() - 1] {
            assert!(
                HnswIndex::load_from_bytes(&bytes[..cut]).is_err(),
                "a truncated file (cut={cut}) should have returned an error"
            );
        }
    }

    #[test]
    fn persist_bitflip_detected_by_crc() {
        let idx = build(&random_vectors(100, 8, 42), Metric::L2);
        let bytes = save_to_bytes(&idx);
        // flip a single bit at various points after the header
        for pos in [8, 20, bytes.len() / 2, bytes.len() - 10] {
            let mut bad = bytes.clone();
            bad[pos] ^= 0x01;
            assert!(
                HnswIndex::load_from_bytes(&bad).is_err(),
                "bit flip @{pos} should have been caught"
            );
        }
    }

    #[test]
    fn persist_wrong_magic_and_version() {
        let idx = build(&random_vectors(10, 4, 42), Metric::L2);
        let bytes = save_to_bytes(&idx);
        let mut bad = bytes.clone();
        bad[0] = b'X';
        assert!(matches!(
            HnswIndex::load_from_bytes(&bad),
            Err(PersistError::Corrupt(_))
        ));
        // change the version and fix the crc so it reaches the version check
        let mut bad = bytes.clone();
        bad[4] = 99;
        let body_len = bad.len() - 4;
        let mut h = crc32fast::Hasher::new();
        h.update(&bad[..body_len]);
        let crc = h.finalize().to_le_bytes();
        bad[body_len..].copy_from_slice(&crc);
        assert!(matches!(
            HnswIndex::load_from_bytes(&bad),
            Err(PersistError::UnsupportedVersion(99))
        ));
    }

    proptest! {
        /// A mini-fuzz (the CI-less counterpart of cargo-fuzz): random bytes
        /// must never cause a panic. The real fuzz target is
        /// fuzz/fuzz_targets/load_index.rs.
        #[test]
        fn prop_load_random_bytes_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..2048)) {
            let _ = HnswIndex::load_from_bytes(&bytes);
        }

        /// Corrupting a random byte of a valid file must either produce an
        /// error or — since there is no location outside the checksum, not even
        /// a pad byte, because the crc covers everything — never panic.
        #[test]
        fn prop_corrupted_valid_file_no_panic(pos in 0usize..500, xor in 1u8..255) {
            let idx = build(&random_vectors(20, 4, 42), Metric::L2);
            let mut bytes = save_to_bytes(&idx);
            let p = pos % bytes.len();
            bytes[p] ^= xor;
            // since the crc covers every byte, corruption must yield Err
            prop_assert!(HnswIndex::load_from_bytes(&bytes).is_err());
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(16))]
        /// Invariant: at a high ef_search, HNSW's FIRST result must largely
        /// agree with brute force's first result.
        #[test]
        fn prop_top1_matches_bruteforce_at_high_ef(seed in 0u64..1000) {
            let vecs = random_vectors(300, 8, seed);
            let queries = random_vectors(20, 8, seed.wrapping_add(1));
            let hnsw = build(&vecs, Metric::L2);
            let mut bf = BruteForceIndex::new(8, Metric::L2);
            for (i, v) in vecs.iter().enumerate() {
                bf.insert(VectorId(i as u64), v).unwrap();
            }
            let mut agree = 0;
            for q in &queries {
                let h = hnsw.search_with_ef(q, 1, 300);
                let b = bf.search(q, 1);
                if h[0].id == b[0].id {
                    agree += 1;
                }
            }
            // top-1 must match in at least 18 of the 20 queries
            prop_assert!(agree >= 18, "top-1 agreement {agree}/20");
        }
    }
}
