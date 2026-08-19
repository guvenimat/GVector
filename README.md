# gvector

A vector search engine written **from scratch** in Rust: HNSW graph, segment
model, WAL-based persistence, metadata filtering, and a measurement-driven
query planner.

## What it is / what it isn't

**This is a learning project and a pre-production prototype.** No off-the-shelf
ANN library was used (no hnsw_rs, faiss, or instant-distance); the algorithms
were read from the papers and written by hand. The goal was to build a working
system and to **justify every decision with a measurement**.

**Not included** (deliberately): authentication, authorization, replication,
sharding, multi-tenancy, TLS, rate limiting, distributed coordination. Single
process, single node, single writer. Do not expose it to the internet.

The code is under `#![deny(unsafe_code)]`; everything including SIMD is safe
Rust (via the `wide` crate).

## Headline numbers (SIFT-1M, 128 dimensions, k=10)

Measurement machine: 8 cores, Windows 11. Every number lives in
`BENCHMARKS.md` together with the conditions and date it was taken under.

| measurement | value |
|---|---|
| recall@10 (ef=100, official ground truth) | **0.9970** |
| search p50 / p99 | 805 µs / 1.14 ms |
| write p50 / p99 (excluding sealing) | 600 ns / 7.6 µs |
| 1M build (with metadata, WAL group:20) | 120 s |
| memory (vectors + graph, f32) | 729 MB |
| memory (metadata structures) | 618 MB |
| disk (checkpoint) | 802 MB (841 B/vector) |
| cold start (1M, empty WAL) | 10.2 s |

In filtered search, **arm agreement is 100%** (the planner picks the same arm
as the oracle in 16 of 16 cells) and filter recall is 1.000 (0.999 in one
cell).

## Architecture

**Segment model.** Writes land in an in-memory brute-force buffer. Once the
buffer reaches its threshold it is *sealed*: its contents become an
independent, immutable HNSW segment. A search walks every segment plus the
buffer and merges the results. Deletion uses segment-local tombstones.

**Single writer + background builds.** There is exactly one writer on the write
path (an mpsc channel feeding a single writer task in the server). The
expensive work — HNSW construction and merging — was moved to background
workers; the window in which the writer is blocked dropped from 80.5 s to
microseconds. If the queue grows, writes are **slowed down, not rejected**
(like Lucene's `IndexWriter` stall). Readers never stop during sealing or
merging.

**Merge ceiling.** When the segment count exceeds the ceiling, the two
smallest segments are merged. The rationale is not latency (at equal recall a
full merge buys ~20%) but cutting off unbounded growth.

**Filter planner.** Three arms: direct scan for a small match set, unfiltered
traversal with over-fetch for a large one, and in-traversal filtering when
there is no estimate. The key design decision: **the small-arm decision is
never made from an estimate** — for Eq the count is already exact, and for
Range a bounded count (`enumerate_up_to`) makes it exact. The histogram
estimate only sizes the over-fetch window of the *large* arm, where the cost
of an error is latency rather than recall. That is why the arm choice stays
correct even when the histogram is off by 49x on a skewed distribution.

**Persistence.** WAL framing is `[len][crc32][payload]`; the sync policy is
`none` / `group(T)` / `per_op` (default `group:20`). An HTTP 200 means exactly
the durability that policy promises. At checkpoint time segments are written to
immutable files, the manifest is written **last**, and GC runs only after that
— so at no point can the manifest reference a missing file. Replay stops at the
first inconsistency and truncates the file at the intact prefix.

## Setup and usage

Requires: Rust (stable). For real measurements, extract SIFT-1M into
`data/sift/` (`sift_base.fvecs`, `sift_query.fvecs`, `sift_groundtruth.ivecs`).

Without the dataset, `report` falls back to random vectors and says so with a
warning — so you can run it without downloading anything. But **recall numbers
on random data are not comparable to SIFT results**: random high-dimensional
data is the worst case for ANN, lacking the cluster structure that real
embeddings have.

```bash
cargo build --release
```

```bash
cargo test --release
```

### Running the measurements

Quickest start — parameter sweep (runs without the dataset):

```bash
cargo run --release --bin report -- sweep 10000 128
```

Other modes (`report -- <mode> <n> <queries>`):

| mode | what it measures |
|---|---|
| `sweep` | HNSW parameter sweep (M, ef_construction, ef_search) |
| `sift` | baseline recall/latency |
| `filter` | Eq filter selectivity sweep + arm agreement |
| `rangefilter` | Range estimation, arm agreement, maintenance cost |
| `fullscale` | 1M end to end (build, memory, recall, filters, merge, cold start) |
| `memverify` | validates the metadata memory estimate against real RSS |
| `accumulation` | queue accumulation under sustained load + backpressure |
| `mergewindow` | effect of the sealing/merge window on write latency |
| `postingcost` | sensitivity of sorted posting lists to id ordering |
| `durability`, `wal`, `delete`, `quant`, `coldprofile` | measurements for the corresponding phases |

### HTTP server

```bash
cargo run --release --bin server
```

Endpoints: `POST /vectors`, `DELETE /vectors/:id`, `POST /search`,
`POST /checkpoint`, `GET /stats`.

## Documentation

- **`DECISIONS.md`** — 69 numbered decisions, each with its rationale.
  Rejected ideas and deferrals are here too.
- **`BENCHMARKS.md`** — every measurement, with its conditions and date.
- **`NOT-DONE.md`** — what was not done, unmet thresholds, and known limits.
- **`METHODOLOGY.md`** — the measurement lessons this project produced.
- **`HANDOFF.md`** — handoff note: status, open items, pitfalls.

## Note on language

The documentation is in English; **code comments are in Turkish**, because
that is the language the project was built in and the comments carry the
reasoning behind each algorithmic choice.
