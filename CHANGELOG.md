# CHANGELOG

## v0.1.0 — 2026-08-19

First release. A vector search engine written from scratch; no off-the-shelf
ANN library was used. The list below gives the phases and each phase's
**headline finding**; the rationale lives in `DECISIONS.md` and the numbers in
`BENCHMARKS.md`.

### Phases

| phase | what arrived | headline finding |
|---|---|---|
| 0 | Skeleton + measurement infrastructure | Measurement infrastructure was built before the code; every acceptance decision came out of it. |
| 1 | Brute-force index (rayon) | The permanent correctness reference: every ANN result is checked against it. |
| 2 | HNSW (Malkov & Yashunin) | recall 0.999 @ ef=50 with M=16, ef_c=200; 9.6x speedup over scanning. |
| 3 | Persistence (GVDB format) | magic + version + CRC32, atomic writes (tmp + fsync + rename). |
| 4 | Tombstone deletion + compaction | Recall holds under 20% deletion (0.9990 → 0.9990). |
| 5 | Segment model | The Lucene/Qdrant model: immutable segments + a write buffer; readers never stop. |
| 6 | Scalar quantization (f32→u8) | Memory drops 2.35x; search via ADC. |
| — | SIMD (`wide` f32x8) | 9x on the distance micro-benchmark, 2.3x end to end — without enabling `unsafe`. |
| — | Metadata filtering | In-traversal filtering + brute-force fallback (correctness guarantee). |
| — | HTTP API (axum) | insert / search / delete / stats / checkpoint. |
| 7 | Manifest + WAL + crash recovery | The manifest is written **last** and GC runs after it; replay stops at the first inconsistency and truncates the file at the intact prefix. |
| — | Filter planner | **The small-arm decision is never made from an estimate.** The histogram can be off by 49x on skewed data and the arm choice still holds; arm agreement 100%. |
| — | Merge ceiling | The rationale is not latency (equal-recall gain ~20%) but cutting off unbounded growth. |
| 8 | 1M end-to-end reality check | Two of three pre-registered items came back "the problem is somewhere else". |
| 8-correction | #44 | **The finding "reads don't scale at 1M" was wrong** — it had been measured in a dirty process; f32 actually scales 5.4–6.1x. |
| 8a | int8 scaling measurement | The threshold passed but **the assumption was falsified**: int8 is slower than f32 under multiple threads → rejected. |
| 9a-1 | Merge moved to the background | The window in which the writer is blocked: 80.5 s → ~29 s. The tombstone diff-replay race was closed by lock discipline. |
| 9a-2 | Sealing moved to the background | Window **20.8 s → 0–2 µs**. Single worker + queue + backpressure (the first design spawned 35 concurrent threads). |
| 9c | Metadata memory compaction | Metadata 934 → 618 MB (−34%); the id→metadata map 2.7x. Real RSS 3088 → 2504 MB. |
| 10 | Closing | README, `NOT-DONE.md`, `METHODOLOGY.md`, v0.1.0. |

### Thresholds not met in this release

Details in `NOT-DONE.md`. In short: #40 criterion 1 (latency ratio), #49 and
#59 secondary (segment accumulation), #61 primary under the group:20 policy.
Each has a defect record beside it; no threshold was changed after the fact.

### Known limits

Segment accumulation (under sustained heavy write load),
`metadata_memory_bytes()` systematically under-reporting, sorted posting lists
being sensitive to id ordering, no OR/negation in filters, single node with a
single writer.

### Contracts

Single writer; readers never stop during sealing/merging; HTTP 200 means the
durability the policy promises (default `group:20`); seed=42; 1M recall at
ef=100 ≥ 0.99; arm agreement 100%; `#![deny(unsafe_code)]`.
