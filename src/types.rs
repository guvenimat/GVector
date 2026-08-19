//! Common core types of the project.

/// Externally supplied identifier of a vector.
///
/// A newtype rather than a bare `u64`: a separate type prevents ids from being
/// confused with in-graph offsets (later `usize` slot indexes) at compile time.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct VectorId(pub u64);

/// Owned vector data. Dimension checking is the index's responsibility;
/// this type is only a carrier.
pub type Vector = Vec<f32>;

/// A single search result.
///
/// `distance` always means "smaller is better": the true distance for L2, the
/// negated similarity for cosine/dot (see the `distance` module). This keeps
/// the ordering logic uniform and independent of the metric.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    pub id: VectorId,
    pub distance: f32,
}

impl SearchResult {
    pub fn new(id: VectorId, distance: f32) -> Self {
        Self { id, distance }
    }
}

impl Eq for SearchResult {}

// f32 is not Ord because of NaN; our distance functions never produce NaN
// (including for the zero vector, see the distance module) — so we define a
// total order via total_cmp, which lets this type be used directly in a
// BinaryHeap.
impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}
