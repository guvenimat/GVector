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

/// Posting-list anahtarı: `MetaValue`'nun hash'lenebilir izdüşümü.
/// Float'lar tam sayıysa Int'e normalize edilir (Eq semantiğiyle tutarlı:
/// Int(3) == Float(3.0)); değilse bit deseni kullanılır.
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

/// f64 → sıralamayı koruyan u64 (BTreeMap anahtarı için).
/// Negatiflerde bitler ters çevrilir, pozitiflerde işaret biti set edilir;
/// sonuç, f64 sıralamasıyla birebir aynı sıralanan bir tamsayıdır.
pub fn ordered_bits(x: f64) -> u64 {
    let b = x.to_bits();
    if b >> 63 == 1 {
        !b
    } else {
        b | (1 << 63)
    }
}

impl Filter {
    /// Filtredeki Eq koşullarının posting-list anahtarları.
    /// Kardinalite tahmini VE bağlacında bunların minimumudur (üst sınır);
    /// Range koşulları tahmine katılmaz (histogramsız — DECISIONS #28).
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

/// Tek koşul. `Range` uçları kapalıdır (min ≤ x ≤ max); tek uç için diğerine
/// ±∞ verilebilir.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Predicate {
    Eq { key: String, value: MetaValue },
    Range { key: String, min: f64, max: f64 },
}

impl Predicate {
    /// Değere ERİŞİM YOLUNDAN bağımsız değerlendirme.
    ///
    /// İki depo var: kullanıcıdan gelen ham `Metadata` (HashMap) ve içeride
    /// tutulan kompakt `MetaStore` (9c). Koşul mantığı tek yerde kalsın diye
    /// ikisi de buraya bir arama kapanışı veriyor.
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
/// Neden: kayıt başına `HashMap<String, MetaValue>` tutmak 1M'de 499 MB
/// yiyordu (ölçüldü, BENCHMARKS 9c-0). Şişkinlik ağırlıkla HashMap'in
/// KENDİSİNDE: her kayıt için tablo başlığı, bucket dizisi ve doluluk payı
/// — string anahtarların tekrarında değil. Bu yüzden çözüm yalnız
/// "anahtar interning" değil, kayıt temsilinin tamamen değişmesi:
///
/// - Alan adları BİR KEZ sözlükte tutulur (`fields`), kayıtlar u32 id taşır.
/// - Kayıt gövdesi `Box<[(u32, MetaValue)]>` — tam boyutlu, kapasite payı
///   yok (`Vec`'in 24 byte'lık başlığı yerine 16 byte'lık fat pointer).
/// - Alan sayısı kayıt başına bir avuç olduğu için arama DOĞRUSAL; bu
///   boyutta ikili arama ya da hash tablosu kurmak kazandırmaz.
#[derive(Debug, Default)]
pub struct MetaStore {
    fields: Vec<String>,
    field_ids: HashMap<String, u32>,
    records: HashMap<VectorId, Box<[(u32, MetaValue)]>>,
}

/// Tek kaydın okuma görünümü: alan adı → değer.
pub struct MetaRef<'a> {
    store: &'a MetaStore,
    rec: &'a [(u32, MetaValue)],
}

impl<'a> MetaRef<'a> {
    pub fn get(&self, key: &str) -> Option<&'a MetaValue> {
        let fid = *self.store.field_ids.get(key)?;
        self.rec.iter().find(|(f, _)| *f == fid).map(|(_, v)| v)
    }

    /// Ham `Metadata`'ya geri çevirir (silme yolu posting/sayısal indeksleri
    /// güncellerken kaydın alan-değer çiftlerine ihtiyaç duyuyor).
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
        // Alan id'sine göre sıralı: aynı şemadaki kayıtlar aynı düzende
        // durur, karşılaştırma ve hata ayıklama öngörülebilir olur.
        rec.sort_by_key(|(f, _)| *f);
        self.records.insert(id, rec.into_boxed_slice());
    }

    pub fn get(&self, id: VectorId) -> Option<MetaRef<'_>> {
        self.records.get(&id).map(|rec| MetaRef {
            store: self,
            rec: rec.as_ref(),
        })
    }

    /// Kaydı düşürür ve ham hâlini döndürür (silme yolu için).
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

    /// Yaklaşık bellek kullanımı (bkz. 9c-0: bu tahminler SİSTEMATİK OLARAK
    /// EKSİK gösteriyor, DECISIONS #66).
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
