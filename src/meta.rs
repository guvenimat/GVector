//! Metadata ve filtre modeli.
//!
//! Filtre stratejisi (gerekçe DECISIONS.md #26): eşleşmeyen node'lar HNSW
//! gezintisinde ziyaret edilir ama sonuç kümesine alınmaz — tombstone'larla
//! aynı mekanizma. "Önce ara sonra filtrele" (post-filter) düşük seçicilikte
//! k'dan az sonuç bırakır; "önce filtrele sonra ara" (pre-filter) ise graf
//! bağlantılılığını koparır. Gezinti-içi filtre ikisinin de tuzağından kaçar;
//! yine de sonuç k'nın altında kalırsa çağıran katman brute-force fallback
//! çalıştırır (doğruluk garantisi).

use crate::types::VectorId;
use std::collections::HashMap;

/// Tek metadata değeri. Sayılar karşılaştırmalarda f64'e normalize edilir
/// (Int(3) ile Float(3.0) aynı kabul edilir — kullanıcı sürprizi olmasın).
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

/// Tek koşul. `Range` uçları kapalıdır (min ≤ x ≤ max); tek uç için diğerine
/// ±∞ verilebilir.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Predicate {
    Eq { key: String, value: MetaValue },
    Range { key: String, min: f64, max: f64 },
}

impl Predicate {
    fn matches(&self, meta: &Metadata) -> bool {
        match self {
            Predicate::Eq { key, value } => meta.get(key).is_some_and(|v| v.equals(value)),
            Predicate::Range { key, min, max } => meta
                .get(key)
                .and_then(MetaValue::as_number)
                .is_some_and(|x| *min <= x && x <= *max),
        }
    }
}

/// Koşulların VE bağlacı (boş filtre her şeyi geçirir).
/// VEYA/negasyon bilinçli olarak yok: ihtiyaç çıkarsa ağaç yapısına genişletilir.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Filter {
    pub must: Vec<Predicate>,
}

impl Filter {
    pub fn matches(&self, meta: &Metadata) -> bool {
        self.must.iter().all(|p| p.matches(meta))
    }

    /// id → metadata deposu üzerinden slot bağımsız değerlendirme.
    /// Metadata'sı hiç olmayan kayıtlar yalnız boş filtreden geçer.
    pub fn matches_id(&self, store: &HashMap<VectorId, Metadata>, id: VectorId) -> bool {
        if self.must.is_empty() {
            return true;
        }
        store.get(&id).is_some_and(|m| self.matches(m))
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
