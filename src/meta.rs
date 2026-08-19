//! Metadata ve filtre modeli.
//!
//! Filter strategy (rationale in DECISIONS.md #26): non-matching nodes are
//! visited during HNSW traversal but not admitted into the result set — the
//! same mechanism as tombstones. "Search first, filter after" (post-filter)
//! leaves fewer than k results at low selectivity; "filter first, search
//! after" (pre-filter) breaks graph connectivity. In-traversal filtering
//! avoids both traps; and if the result still falls below k, the calling layer
//! runs a brute-force fallback (correctness guarantee).

use crate::types::VectorId;
use std::collections::HashMap;

/// A single metadata value. Numbers are normalized to f64 in comparisons
/// (Int(3) and Float(3.0) are considered equal — no surprises for the user).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum MetaValue {
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl MetaValue {
    fn as_number(&self) -> Option<f64> {
        match self {
            MetaValue::Int(i) => Some(*i as f64),
            MetaValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    fn equals(&self, other: &MetaValue) -> bool {
        match (self.as_number(), other.as_number()) {
            (Some(a), Some(b)) => a == b,
            _ => self == other,
        }
    }
}

pub type Metadata = HashMap<String, MetaValue>;

/// Posting-list key: a hashable projection of `MetaValue`.
/// Floats that are whole numbers are normalized to Int (consistent with the Eq
/// semantics: Int(3) == Float(3.0)); otherwise the bit pattern is used.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetaKey {
    Bool(bool),
    Int(i64),
    FloatBits(u64),
    Str(String),
}

impl MetaValue {
    pub fn key(&self) -> MetaKey {
        match self {
            MetaValue::Bool(b) => MetaKey::Bool(*b),
            MetaValue::Int(i) => MetaKey::Int(*i),
            MetaValue::Float(f) if f.fract() == 0.0 && f.abs() < i64::MAX as f64 => {
                MetaKey::Int(*f as i64)
            }
            MetaValue::Float(f) => MetaKey::FloatBits(f.to_bits()),
            MetaValue::Str(s) => MetaKey::Str(s.clone()),
        }
    }
}

/// f64 → order-preserving u64 (for use as a BTreeMap key).
/// For negatives the bits are inverted, for positives the sign bit is set; the
/// result is an integer that sorts exactly like the original f64.
pub fn ordered_bits(x: f64) -> u64 {
    let b = x.to_bits();
    if b >> 63 == 1 {
        !b
    } else {
        b | (1 << 63)
    }
}

impl Filter {
    /// The posting-list keys of the Eq predicates in the filter.
    /// Under an AND conjunction, the cardinality estimate is the minimum of
    /// these (an upper bound); Range predicates do not take part in the
    /// estimate (no histogram — DECISIONS #28).
    pub fn eq_keys(&self) -> Vec<(&str, MetaKey)> {
        self.must
            .iter()
            .filter_map(|p| match p {
                Predicate::Eq { key, value } => Some((key.as_str(), value.key())),
                Predicate::Range { .. } => None,
            })
            .collect()
    }
}

/// A single predicate. `Range` endpoints are closed (min ≤ x ≤ max); for a
/// one-sided range, pass ±∞ for the other end.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Predicate {
    Eq { key: String, value: MetaValue },
    Range { key: String, min: f64, max: f64 },
}

impl Predicate {
    /// Evaluation independent of HOW the value is accessed.
    ///
    /// There are two stores: the raw `Metadata` (a HashMap) coming from the
    /// user, and the compact `MetaStore` kept internally (9c). So that the
    /// predicate logic lives in exactly one place, both hand a lookup closure
    /// to this function.
    fn matches_with<'a>(&self, get: impl Fn(&str) -> Option<&'a MetaValue>) -> bool {
        match self {
            Predicate::Eq { key, value } => get(key).is_some_and(|v| v.equals(value)),
            Predicate::Range { key, min, max } => get(key)
                .and_then(MetaValue::as_number)
                .is_some_and(|x| *min <= x && x <= *max),
        }
    }

    fn matches(&self, meta: &Metadata) -> bool {
        self.matches_with(|k| meta.get(k))
    }
}

/// The AND conjunction of predicates (an empty filter passes everything).
/// OR/negation are deliberately absent: this would be extended to a tree
/// structure if the need arose.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Filter {
    pub must: Vec<Predicate>,
}

impl Filter {
    pub fn matches(&self, meta: &Metadata) -> bool {
        self.must.iter().all(|p| p.matches(meta))
    }

    /// Slot-independent evaluation over the id → metadata store.
    /// Records with no metadata at all pass only the empty filter.
    pub fn matches_id(&self, store: &MetaStore, id: VectorId) -> bool {
        if self.must.is_empty() {
            return true;
        }
        match store.get(id) {
            None => false,
            Some(rec) => self.must.iter().all(|p| p.matches_with(|k| rec.get(k))),
        }
    }
}

/// id → metadata deposunun KOMPAKT temsili (9c, DECISIONS #65).
///
/// Why: keeping a `HashMap<String, MetaValue>` per record was consuming
/// 499 MB at 1M (measured, BENCHMARKS 9c-0). The bloat is mostly in the
/// HashMap ITSELF — a table header, a bucket array and load-factor slack per
/// record — not in the repetition of the string keys. So the fix is not merely
/// "key interning" but changing the record representation entirely:
///
/// - Field names are stored ONCE in a dictionary (`fields`); records carry a
///   u32 id.
/// - The record body is a `Box<[(u32, MetaValue)]>` — exactly sized, no
///   capacity slack (a 16-byte fat pointer instead of `Vec`'s 24-byte header).
/// - Lookup is LINEAR because there is a handful of fields per record; at that
///   size neither binary search nor a hash table would pay off.
#[derive(Debug, Default)]
pub struct MetaStore {
    fields: Vec<String>,
    field_ids: HashMap<String, u32>,
    records: HashMap<VectorId, Box<[(u32, MetaValue)]>>,
}

/// Read view of a single record: field name → value.
pub struct MetaRef<'a> {
    store: &'a MetaStore,
    rec: &'a [(u32, MetaValue)],
}

impl<'a> MetaRef<'a> {
    pub fn get(&self, key: &str) -> Option<&'a MetaValue> {
        let fid = *self.store.field_ids.get(key)?;
        self.rec.iter().find(|(f, _)| *f == fid).map(|(_, v)| v)
    }

    /// Converts back to a raw `Metadata` (the delete path needs the record's
    /// field-value pairs to update the posting and numeric indexes).
    pub fn to_metadata(&self) -> Metadata {
        self.rec
            .iter()
            .map(|(f, v)| (self.store.fields[*f as usize].clone(), v.clone()))
            .collect()
    }
}

impl MetaStore {
    pub fn new() -> Self {
        Self::default()
    }

    fn field_id(&mut self, name: &str) -> u32 {
        if let Some(id) = self.field_ids.get(name) {
            return *id;
        }
        let id = self.fields.len() as u32;
        self.fields.push(name.to_string());
        self.field_ids.insert(name.to_string(), id);
        id
    }

    pub fn insert(&mut self, id: VectorId, meta: Metadata) {
        let mut rec: Vec<(u32, MetaValue)> = meta
            .into_iter()
            .map(|(k, v)| (self.field_id(&k), v))
            .collect();
        // Sorted by field id: records sharing a schema are laid out in the
        // same order, which makes comparison and debugging predictable.
        rec.sort_by_key(|(f, _)| *f);
        self.records.insert(id, rec.into_boxed_slice());
    }

    pub fn get(&self, id: VectorId) -> Option<MetaRef<'_>> {
        self.records.get(&id).map(|rec| MetaRef {
            store: self,
            rec: rec.as_ref(),
        })
    }

    /// Drops the record and returns its raw form (for the delete path).
    pub fn remove(&mut self, id: VectorId) -> Option<Metadata> {
        let rec = self.records.remove(&id)?;
        Some(
            rec.iter()
                .map(|(f, v)| (self.fields[*f as usize].clone(), v.clone()))
                .collect(),
        )
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (VectorId, MetaRef<'_>)> {
        self.records.iter().map(move |(id, rec)| {
            (
                *id,
                MetaRef {
                    store: self,
                    rec: rec.as_ref(),
                },
            )
        })
    }

    /// Approximate memory usage (see 9c-0: these estimates SYSTEMATICALLY
    /// UNDER-REPORT, DECISIONS #66).
    pub fn memory_bytes(&self) -> usize {
        let dict: usize = self.fields.iter().map(|f| f.len() + 24).sum::<usize>()
            + self.field_ids.capacity() * (std::mem::size_of::<String>() + 4 + 8);
        let entry = std::mem::size_of::<(u32, MetaValue)>();
        let bodies: usize = self.records.values().map(|r| r.len() * entry).sum();
        dict + self.records.capacity() * (8 + 16 + 8) + bodies
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(pairs: &[(&str, MetaValue)]) -> Metadata {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn eq_and_range() {
        let m = meta(&[
            ("renk", MetaValue::Str("mavi".into())),
            ("yil", MetaValue::Int(2020)),
        ]);
        let f = Filter {
            must: vec![
                Predicate::Eq {
                    key: "renk".into(),
                    value: MetaValue::Str("mavi".into()),
                },
                Predicate::Range {
                    key: "yil".into(),
                    min: 2019.0,
                    max: 2021.0,
                },
            ],
        };
        assert!(f.matches(&m));
        let f2 = Filter {
            must: vec![Predicate::Range {
                key: "yil".into(),
                min: 2021.0,
                max: f64::INFINITY,
            }],
        };
        assert!(!f2.matches(&m));
    }

    #[test]
    fn int_float_cross_type_equality() {
        let m = meta(&[("x", MetaValue::Int(3))]);
        let f = Filter {
            must: vec![Predicate::Eq {
                key: "x".into(),
                value: MetaValue::Float(3.0),
            }],
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn missing_key_fails_nonempty_filter() {
        let m = meta(&[]);
        let f = Filter {
            must: vec![Predicate::Eq {
                key: "yok".into(),
                value: MetaValue::Bool(true),
            }],
        };
        assert!(!f.matches(&m));
        assert!(Filter::default().matches(&m));
    }
}
