# Benchmark Records

## Phase 9c — metadata memory compaction (RESULTS) — 2026-08-19

### 9c-0: validating the estimate against RSS

The structures were dropped one at a time and the RSS delta measured
(`report -- memverify`). Not `clear()` — each structure was REPLACED with a new
one (clear keeps the capacity).

**Measured on a directory with 1.50M records** (1M of which carry metadata; the
rest are the metadata-free records of the `mergewindow` runs):

| step | RSS | drop | estimate | estimate/real |
|---|---|---|---|---|
| start | 3088 MB | — | — | — |
| −numeric | 2891 MB | 197 MB | 168 MB | 0.85x |
| −postings | 2321 MB | 570 MB | 371 MB | 0.65x |
| −metadata | 1822 MB | 499 MB | 441 MB | 0.89x |
| **total** | | **1266 MB** | 980 MB | **0.77x** |

**The real metadata share is 41.0%** (threshold 25% → GO stands).

The direction of the bias: the estimate UNDER-reports reality. "The OS did not
return the memory" cannot explain this — in that case the real drop would look
SMALLER. So 41% is a LOWER bound, and the GO decision did not rest on an inflated
estimate.

### 9c-1: after the implementation

The comparison uses the SAME estimator on the SAME 1M metadata set (`fullscale`
section 2 — no normalization by record count is needed, because in both runs
exactly 1M records carry metadata):

| item | before | after | gain |
|---|---|---|---|
| the map (id→metadata) | 421 MB | **158 MB** | **2.7x** |
| posting lists | 353 MB | **299 MB** | 1.2x |
| numeric indexes | 160 MB | 160 MB | (untouched) |
| **metadata total** | **934 MB** | **618 MB** | **−34%** |
| metadata share | 51.5% | **45.9%** | |

Real RSS (memverify, measured on a directory with 1.23M records):
**3088 → 2504 MB**, real metadata share **41.0% → 33.3%**.

Why the posting gain stayed small: the dominant term is not the inside of the
`HashSet`s but the **number of distinct (field, value) keys** — numeric fields
produce a great many distinct values, so the outer `HashMap` dominates.

### Correctness and cost (the acceptance criteria)

| criterion | before | after |
|---|---|---|
| arm agreement (fullscale / rangefilter) | 3/3, 13/13 | **3/3, 13/13 (100%)** |
| filter recall (13 cells) | 1.000 (lv 0.1: 0.999) | **identical** |
| 1M recall ef=100 | 0.9970 | **0.9970** (≥0.99 ✓) |
| 1M build (metadata + WAL group:20) | 170.4 s | **120.4 s** |
| metadata maintenance share (rangefilter 100K) | +4% (9.9→10.2 s) | +17% (4.9→5.7 s) |

### The known cost of a sorted posting list

`report -- postingcost 200000`: 200K records into a SINGLE posting list.

| id order | time | records/s |
|---|---|---|
| ascending | 144.6 ms | 1,383,172 |
| **random** | **1.266 s** | **157,945** |

Inserting into a sorted `Vec` costs an O(n) shift. If ids arrive in ascending
order the position is always at the end (O(1)); in random order it is **8.8x**
slower. At 200K the list is 1.6 MB, so it fits in cache and the shift runs at
memory bandwidth — which is why the O(n²) growth does not bite at this size. A
SINGLE 1M-entry list would take roughly 30 s. Because our measurements always
generated ids in ascending order, this difference had not been visible before.


## Phase 9a-2 — criterion 1 after `with_capacity` (pre-registration #61) — 2026-08-19

Measurement results only. `data/fullscale` was rebuilt from scratch (1M, 8
segments) and the same protocol was run under both fsync policies, 130,000
writes.

| measurement | **group:20** (the pre-registered condition) | **sync off** (isolating the difference) |
|---|---|---|
| baseline p50 | 900 ns | 2.7 µs |
| baseline p99 | 10.3 µs | 16.5 µs |
| **PRIMARY: longest write excluding backpressure** | **4.435 ms** | **200.3 µs** |
| PRIMARY ratio (against its own p99) | **426x** | **12x** |
| ratio (against the phase 8 baseline of 7.8 µs) | 569x | 26x |
| the pre-registered 50x threshold | **NOT MET** | **MET** |
| time sealing blocks the writer | 53 µs | 58 µs |
| SECONDARY: stalled writes (backpressure) | **0** | **0** |

### The effect of `with_capacity` (the realloc hypothesis is PROVEN)

The 5 slowest writes, before and after:

| policy | BEFORE `with_capacity` | AFTER |
|---|---|---|
| group:20 | 4.0 – 10.0 ms | 4.03 – 4.44 ms |
| sync off | 1.8 – 4.4 ms | **98 – 200 µs** |

With sync off the spikes shrank **~20x** (from milliseconds to hundreds of
microseconds). The buffer's incremental growth (a ≈64 MB realloc + memcpy) was
the source of the non-fsync spikes; with the capacity allocated up front they are
gone.

### The remaining spike is entirely fsync

The five remaining spikes under group:20 are **4.03, 4.04, 4.06, 4.10, 4.44 ms** —
a tight cluster, spread across 130,000 writes (#14824, #46099, #61013, #107181,
#129620) and unrelated to the sealing point (#21266). With sync off they vanish
completely. So the remaining share is the price of the durability policy, not a
defect in the write path.

### The secondary item: no data

Backpressure never kicked in under either policy (0 stalls): the measurement
covers 130,000 writes and the queue threshold is never crossed. No data could be
produced for #61's secondary item this round; the distribution can only be
observed under sustained load (the `accumulation` mode) — where 21 stalls
totalling 594 s were measured over ten minutes.


## Phase 9a-2 — measuring the two pre-registered criteria (RESULTS) — 2026-08-19

This section contains ONLY measurement results. The decisions drawn from them and
the analysis of the threshold defects are separate records in DECISIONS
(deliberately in separate commits, so the question "did the decision shape the
result" stays unambiguous).

### Criterion 2 — accumulation, 10 minutes (pre-registration #59)

1M SIFT, 600 s of uninterrupted full-speed writing, seal=125K, ceiling=8, WAL
off, sampled every 5 s. 2.875M records were written in total (~4.8K op/s
sustained).

| t (s) | segments | sealing (queue) |
|---|---|---|
| 5 | 0 | 3 |
| 110 | 5 | 3 |
| 200 | 9 | 3 |
| 320 | 12 | 3 |
| 440 | 13 | 3 |
| 530 | 16 | 3 |
| 600 | 16 | 3 |

| #59 item | value | threshold | result |
|---|---|---|---|
| **PRIMARY** queue: first 1/3 → last 1/3 | 3.0 → 3.0 (**−2%**) | settles | **OK** |
| PRIMARY queue peak | **3** | a fixed upper bound | — |
| SECONDARY segment peak (a test of the merge ceiling) | **16** | ≤ 12 | **EXCEEDED** |
| (reference) #49's flawed metric: segments+sealing | 7.7 → 17.7 | ≤ +20% | EXCEEDED |

Backpressure: 21 inserts stalled, 594.4 s in total.

### Criterion 1 — latency (pre-registration #40)

The measurement condition is the same as in phase 8 / 9a-1: `data/fullscale` was
REBUILT FROM SIFT (1M, 8 segments — earlier runs had grown the directory to
1.64M/10 segments), WAL group:20, no commit inside the loop, an isolated process,
a warmup of 5,000 writes, and 130,000 writes measured.

| measurement | run 1 | run 2 |
|---|---|---|
| baseline p50 | 1 µs | 1 µs |
| baseline p99 | 9.9 µs | 10.6 µs |
| **longest single write** | **6.031 ms** | **9.996 ms** |
| **time sealing blocks the writer** | **2 µs** | **0 ns** |
| merge count (background) | 0 | 0 |
| ratio (longest / baseline p99) | **609x** | **943x** |
| the pre-registered 50x threshold | **NOT MET** | **NOT MET** |

**The trajectory of the window that blocks the writer (the real finding):**

| | phase 8 | 9a-1 | 9a-2 |
|---|---|---|---|
| longest window in which the writer is blocked | 80.5 s | 28.5–30.6 s | **0 ns – 2 µs** |
| its components | sealing 20.8 s + merging 59.7 s | sealing 28.5 s | — |

### Diagnosis: where do the 6–10 ms spikes come from?

The index of the 5 slowest writes was compared with the index at which sealing
occurs (sealing at ~#11266):

```
#5444 → 9.9959ms   #71802 → 5.6031ms   #70092 → 4.155ms
#115892 → 4.1223ms #38739 → 4.0845ms
```

The spikes are SPREAD across 130,000 writes and unrelated to the sealing point.

A diagnostic run with WAL sync off (NOT the pre-registered condition,
`GVDB_DIAG_NOWAL=1`): the spikes fell from 6–10 ms to 1.8–4.4 ms but did NOT
DISAPPEAR → fsync is part of the contribution, not all of it. The remaining
hypothesis: the write buffer's capacity growth (125K × 128 floats ≈ a 64 MB
realloc). The hypothesis is not proven yet.

**An unexpected result of the diagnostic run:** with the WAL off the writer sped
up, the queue threshold was crossed and **backpressure kicked in** — the longest
write became **24.9 s** (#121265). That is not a defect but the designed
behaviour of #53: the very mechanism that makes criterion 2 pass.


## Phase 9a-2 — single worker + backpressure, the second criterion-2 measurement — 2026-08-19

1M SIFT, 120 s of uninterrupted full-speed writing, seal=125K, ceiling=8, WAL off.

**(a) The first backpressure signal (sealing+segments > 16) — WRONG (#56):**

| t (s) | segments | sealing | write op/s |
|---|---|---|---|
| 5 | 0 | 10 | 271,293 |
| 10 | 0 | 17 | 153,707 |
| 15–65 | 0→3 | 17→14 | **0** |
| 120 | 6 | 12 | **0** |

The writer stopped completely for 110 seconds; 2 inserts hit the 60 s safety
limit. Because the sum does not fall when a sealing finishes, there is no
feedback.

**(b) The corrected signal (the queue alone, threshold 2):**

| t (s) | segments | sealing | write op/s |
|---|---|---|---|
| 5 | 0 | 3 | 75,000 |
| 40 | 2 | 3 | 25,000 |
| 80 | 4 | 3 | 25,000 |
| 120 | 6 | 3 | 7,595 |

| criterion | value | threshold | result |
|---|---|---|---|
| first 1/3 → last 1/3 avg (segments+sealing) | 3.8 → 7.9 (**+110%**) | ≤ +20% | **EXCEEDED** |
| peak (segments+sealing) | 9 | ≤ 12 | OK |
| queue length | **steady at 3** | (not in the pre-registration) | — |

**CRITERION 2 RESULT: NOT MET** (the threshold was not reinterpreted, #58). The
queue is flat; the growing part is the segment count (0→6), i.e. approaching the
merge ceiling. The 2-minute window ended before the ceiling was reached → it was
extended to 10 minutes with the new pre-registration #59 (a decision taken after
seeing the result).

For comparison — the previous design (unbounded threads, #52): the queue went
0→35 in 60 s, segments stayed at 0, writes fell 273K→11.7K, memory ~2.3 GB. In
the new design the queue is 3, segments are produced steadily, writing settles at
the sealing rate (~5K op/s), and memory is ~2 buffers.


## Phase 9a-2 — the accumulation measurement (pre-registration #49, CRITERION 2) — 2026-08-19

60 s of uninterrupted FULL-SPEED writing; seal=125K, ceiling=8, WAL off (with the
WAL on, the first attempt wrote **4.3 GB** of log in 120 s and drowned the
measurement).

| t (s) | segments | sealing (queue) | write op/s |
|---|---|---|---|
| 5 | 0 | 10 | 273,337 |
| 15 | 0 | 21 | 102,022 |
| 30 | 0 | 28 | 52,253 |
| 45 | 0 | 33 | 28,738 |
| 60 | 0 | **35** | **11,709** |

| criterion | value | threshold | result |
|---|---|---|---|
| first 1/3 → last 1/3 average | 17.8 → 33.8 (**+90%**) | ≤ +20% | **EXCEEDED** |
| peak (segments + sealing) | **35** | ≤ 12 | **EXCEEDED** |

**CRITERION 2 RESULT: IT ACCUMULATES → backpressure is part of 9a-2
(pre-registration #49).**

Three further findings:
1. **The segment count stayed at 0 for the full 60 seconds** — that is, *no*
   sealing could complete. 35 sealings run at once, sharing 8 cores.
2. **The write rate collapsed from 273K to 11.7K op/s** (23x). The system is
   already slowing itself down, but it does so through *memory pressure* — an
   uncontrolled collapse, not designed backpressure.
3. **Memory:** 35 buffers being sealed × 125K records ≈ 2.3 GB; on the first
   attempt the process reached 7.9 GB RSS.


## Phase 9a-1 — merging moved to the background — 2026-08-19 (SIFT 1M, full system)

The measurement protocol is the one from 8a: an isolated process, a warmup (5,000
writes), and a repeat in two separate runs. The measurement condition is the same
as in phase 8 (WAL group:20, no commit inside the loop) — the pre-registered 50x
threshold is defined against an fsync-free baseline.

| measurement | phase 8 (synchronous merge) | 9a-1 (merge in the background) |
|---|---|---|
| **longest window in which the writer is blocked** | **80.5 s** | **28.5 s / 30.6 s** |
| — its components | sealing 20.8 s + merging 59.7 s | sealing only, 28.5 / 30.6 s |
| merge duration | (on the writer) 59.7 s | (in the background) 53.5 s / 54.0 s |
| merge count | 1 | 2 / 4 |
| baseline write p50 | 600 ns | 1.2 µs / 1.3 µs |
| baseline write p99 | 7.8 µs | 10.1 µs / 8.4 µs |
| ratio (max / baseline p99) | 10.3 M x | 2.8 M x / 3.6 M x |
| the pre-registered 50x threshold | not met | **not met (as expected)** |

"Was a merge running as the measurement ended: **YES**" — confirming that
merging overlaps with the write stream; merging now genuinely runs in parallel.

**Readings:**
- The time the writer is blocked went **80.5 s → ~29 s (a 2.8x reduction)**; all
  of the remaining time is **sealing**, so 9a-2's target was confirmed by
  measurement.
- An honest cost: sealing ITSELF went from 20.8 s to 28–30 s (+40%). The reason
  is that merging now runs in parallel and shares the CPU. The net gain is still
  large (80.5 → 29), but "moving merging to the background" is not free.
- The 50x threshold was not met at this step, and this was **already anticipated
  in the pre-registration** ("it will not be met in 9a-1; report it"). The
  threshold will be tested again after 9a-2.
- The segment count temporarily exceeds the ceiling (9 and 10 at the start of the
  runs; sealing outpaces merging). Once writing stops the worker brings it back
  down (both runs ended with 8 segments).


## Phase 8a — int8 multi-reader scaling — 2026-08-19

The machine: **AMD Ryzen 7 7800X3D, 8 physical / 16 logical cores, L3 = 96 MB**
(3D V-Cache). Data: `data/fullscale` (1.13M records, 8 segments). Measurement:
warmup plus the median of 3 repeats; **repeated in two separate processes**.

| index | ef | 1 thread | 2 | 4 | 8 | scaling (8/1) |
|---|---|---|---|---|---|---|
| f32 | 50 | 1286 / 1273 | 2443 / 1920 | 4649 / 3793 | **7865 / 6872** | **6.12x / 5.40x** |
| f32 | 100 | 794 / 738 | 1229 / 1180 | 2371 / 2334 | **4380 / 4355** | **5.52x / 5.90x** |
| int8 | 50 | 1226 / 1236 | 2243 / 2317 | 3242 / 4374 | **3643 / 3399** | **2.97x / 2.75x** |
| int8 | 100 | 741 / 803 | 964 / 1195 | 1694 / 1753 | **2649 / 2751** | **3.58x / 3.42x** |

(cells: run 1 / run 2 — a reproducibility check)

| | value |
|---|---|
| working set | f32 847 MB → int8 424 MB (**2.00x**, the ~2x anticipated in the plan) |
| quantization time | 0.39 s (8 segments) |
| recall loss (f32→int8) | **0.0091** (ef=50) / **0.0101** (ef=100) — threshold 0.02 ✓ |
| 8-thread QPS ratio int8/f32 | **0.46–0.49x** (ef=50), **0.60–0.63x** (ef=100) |

**Conclusions:**
1. **f32 does scale at 1M: 5.4–6.1x** (on 8 physical cores). Phase 8's finding
   that "reads do not scale at 1M (8 threads = single-thread QPS)" was **WRONG** —
   see DECISIONS #44.
2. **int8 scales LESS (2.75–3.58x) and is about 2x SLOWER in absolute terms.**
   The cause is ADC: the dequantization arithmetic (min + scale·code) on every
   distance; the phase 6 micro-benchmark already said 15.6 ns (ADC) vs 7.4 ns
   (f32 L2). On a single thread the memory advantage offsets that cost (1226 vs
   1286); with 8 threads, once the CPU is the bottleneck, ADC dominates.
3. **The L3 hypothesis:** the 8.7x scaling at 100K came from the working set
   (92 MB) fitting in the 96 MB L3. At 1M, f32 is 847 MB and int8 424 MB — **both
   far above L3**, so quantization does not bring the working set into cache, it
   merely halves DRAM traffic. This explains why the expectation that int8 would
   "bring scaling back" came to nothing.

**Methodology note:** the first version was not reproducible (same code, same
data: 5.08x and 1.14x). The cause: opening a second large index in the same
process (memory pressure + cache pollution). The fix: warmup plus the median of 3
repeats, a single index, and verification in separate processes.

**A warning about the absolute recall value:** the 0.8242 in the table comes from
`data/fullscale` containing 130K duplicate records left over from the fullscale
run (the same vectors under different ids; the official GT is for a clean 1M).
What is meaningful is the f32→int8 **loss** (0.009–0.010), not the absolute
value.

## Phase 8 — the 1M end-to-end reality check — 2026-08-19 (full SIFT1M set, full system)

Configuration: 8 segments (seal=125K, ceiling=8), 3 metadata fields + 3 clustered
filter labels, WAL=group:20, f32. The thresholds are pre-registered
(DECISIONS #40).

| Measurement | Value |
|---|---|
| build (1M + metadata, WAL on) | **170.4 s** (a single 1M graph: 802 s → 4.7x faster) |
| checkpoint | 2.44 s, disk **802 MB** (841 B/vector) |
| cold start (median, 3 rounds) | **3.63 s** (1.13M records) |
| cold start + 10K WAL | 3.63 s (the replay effect is not measurable) |
| **recall@10 (official GT, ef=100)** | **0.9970** — threshold ≥0.99 **MET** |
| search p50 / p99 (single thread) | 954.6 µs / 1.17 ms |
| memory (computed) | vectors+graph **882 MB**, metadata **934 MB** |
| memory (peak RSS) | **3167 MB** |

### Cold-start components (for the 9b decision)

| component | time | share |
|---|---|---|
| (a) reading segment files + CRC (812 MB) | 196 ms | 5% |
| (b) segment parse (graph + vector copy) | 722 ms | 20% |
| (c) metadata read + decode (83 MB) | 428 ms | 12% |
| (d) **building the derived indexes** (posting + numeric) | **2.28 s** | **63%** |
| total | 3.62 s | |

### Critical filter cells (clustered × distant query, the real planner path)

| s | matches | arm (oracle) | recall | p50 |
|---|---------|--------------|--------|-----|
| 0.001 | 1,000 | scan (scan) | 1.000 | 131 µs |
| 0.05 | 50,000 | scan (scan) | 1.000 | 9.36 ms |
| 0.3 | 300,000 | post (post) | 0.997 | **92.65 ms** |

Arm agreement **3/3 (100%)** — the 100K result held at 1M. Recall held, but in
the s=0.3 cell latency rose from 3.9 ms at 100K to 92.7 ms (23x for 10x the
data): the cost of the scan arm grows linearly with the number of matches.

### The merge window (the rationale for 9a)

| | value |
|---|---|
| baseline write p50 / p99 (outside the window) | 600 ns / **7.8 µs** |
| **longest single write** | **80.5 s** |
| — its components | sealing **20.8 s** + merging **59.7 s** |
| ratio (max / baseline p99) | 10.3 million x |

### Mixed load: 8 readers + 1 writer (throttled to 200 op/s)

| policy | read QPS | ratio to baseline | write op/s | read p50 |
|---|---|---|---|---|
| writer-free baseline | 945 | 1.00 | — | 8.48 ms |
| `none` | 950 | **1.01** | 200 | 8.42 ms |
| `group:20` | 937 | **0.99** | 200 | 8.56 ms |
| `per_op` | 937 | **0.99** | 200 | 8.55 ms |

**The phase 5 contract passed its test:** the fsync policy puts no measurable
load on readers (ratio 0.99–1.01). This RATIO result remains valid.

> ⚠️ **CORRECTION (2026-08-19, DECISIONS #44):** the **absolute QPS values in
> this table are invalid**, and the conclusion drawn from them — "reads do not
> scale at all at 1M" — was **WRONG**. An isolated measurement (phase 8a) showed
> f32 scaling **5.4–6.1x** at 1M. The source of the error: this table was taken
> as the last section of `fullscale`, in a process that had been running for five
> minutes and had done a 1M build, 130K writes, a merge and three cold starts,
> with RSS at 3.1 GB. The same suspicion applies to the other ABSOLUTE numbers
> from this run (the merge window, the cold-start components), which should be
> confirmed by isolated measurement; the merge window was confirmed in 9a-1 (the
> section above).

### The crash test (1M snapshot + a full WAL)

A 145K-record WAL was cut at 67% → an intact prefix of 103,734 records; opening
took **3.69 s** and the recovered state was **exactly equal** to the intact
prefix. Note: if the number of replayed records exceeds the sealing threshold
(125K), an HNSW build is triggered inside the replay and recovery time stops
being linear — in an invalid first run (random data, 252K records) this was
observed as 206 s. The mechanism is data-independent; checkpoint frequency
directly determines recovery time.

## Phase 7b/7c — WAL: fsync policy and recovery — 2026-08-18

20,000 inserts (SIFT, 128d + 1 metadata field), batch=64 (the behaviour of the
server's writer task), sealing off — the pure WAL path is what is measured.

| policy | time | throughput | fsync/op | WAL | replay |
|---|---|---|---|---|---|
| `none` | 71 ms | **281,609 op/s** | 0.000 | 10.9 MB | 36.8 ms |
| `group:20` | 632 ms | **31,669 op/s** | 0.016 | 10.9 MB | 30.5 ms |
| `per_op` | 40.1 s | **499 op/s** | 1.000 | 10.9 MB | 33.2 ms |

Full WAL replay (100K records / 54.3 MB): **155 ms — 646,000 records/s**.

Readings:
- An fsync takes ~2 ms (Windows, `sync_data`) → per_op's ceiling of 499 op/s
  comes straight from that. Group commit delivers the same durability promise at
  **63x** the throughput: hence the default of `group:20` (DECISIONS #36).
- The 8.9x difference between `none` and `group` is the real price of fsync.
- Replay is fast: 100K records in 155 ms — next to cold start (242 ms, phase 7a)
  it is cheap to keep the checkpoint interval long. WAL size affects replay time
  linearly.

**The crash matrix (deterministic truncation, 5 tests):** at a record boundary /
mid-header / mid-body / one byte short / a cut after a checkpoint plus a corrupt
body → in every case the recovered state EQUALS the WAL's intact prefix, with no
panic, and a second replay is idempotent (because the file was truncated). With
proptest: a random operation sequence × a random cut point (24 cases) and
entirely random bytes → no panic.

## Phase 7a — cold persistence — 2026-08-18 (SIFT 100K, 3 metadata fields, 8 segments)

| Measurement | Value |
|---|---|
| build (with 3 metadata fields) | 8.1 s |
| first checkpoint (writing 8 segments) | 221 ms |
| second checkpoint (no new segments) | 98 ms |
| disk total | 79.7 MB (836 B/vector) |
| cold start (8 segments + derived indexes) | 242 ms |
| recall@10 after reopening | 1.0000 |

Readings:
- The second checkpoint's 98 ms is **entirely the full metadata write** (100K × 3
  fields) plus the manifest; segment writing is zero because the files are
  immutable (DECISIONS #32). At 1M this item grows ~10x — one of the things
  phase 8 will measure.
- Within the 242 ms cold start are the GVDB loads of 8 segments plus rebuilding
  the posting lists and numeric indexes from metadata (they are not written to
  disk).
- 836 B/vector: 512 B of raw vector, part of the ~404 B graph, and the metadata
  snapshot; the graph/vector ratio is consistent with the phase 2 table.

End-to-end HTTP validation (dim=4, persistent mode): 3 inserts + 1 delete →
`POST /checkpoint` (gen=1) → the process was killed → restarted → `GET /stats`
shows 2 records/gen=1, search does not return the deleted one, Eq and Range
filters work (the derived indexes were recovered), and the deleted id can be
inserted again.

## The Range histogram — 2026-08-18 (SIFT 100K, 64 equal-width buckets, k=10)

Maintenance cost: a build without metadata takes 9.9 s → with 3 fields (2
numeric) 10.2 s (**+4%**). scan_limit = 5000. The estimate is an [lower, upper]
interval; "upper/truth" is the error indicator.

| field | s | truth | estimate [lower,upper] | upper/truth | arm (oracle) | recall | p50 |
|------|---|--------|------------------|-----------|--------------|--------|-----|
| v(uniform) | 0.01 | 1000 | [0,1608] | 1.61 | scan (scan) | 1.000 | 112µs |
| v(uniform) | 0.1 | 10000 | [8039,11255] | 1.13 | post (post) | 1.000 | 1.67ms |
| v(uniform) | 0.3 | 30000 | [27333,30549] | 1.02 | post (post) | 1.000 | 727µs |
| v(uniform) | 0.5 | 50000 | [48235,51451] | 1.03 | post (post) | 1.000 | 481µs |
| lv(skewed) | 0.01 | 1000 | [0,48788] | 48.8 | scan (scan) | 1.000 | 117µs |
| lv(skewed) | 0.1 | 10000 | [0,48788] | 4.88 | post (post) | 0.999 | 550µs |
| lv(skewed) | 0.3 | 30000 | [0,48788] | 1.63 | post (post) | 1.000 | 566µs |
| lv(skewed) | 0.5 | 50000 | [48788,79485] | 1.59 | post (post) | 1.000 | 376µs |
| Eq∧Range correlated | 0.1 | 10000 | min-upper: 22510 | 2.25 | post (post) | 1.000 | 1.22ms |

**Against the acceptance criteria:**
- On the uniform distribution the post-band estimation error is 2–13% (< 20% ✓).
- On the skewed one it was measured and the predicted pathology confirmed: in a
  log-normal distribution nearly all the mass sits in the first buckets (48788
  records in the neighbourhood of a single bucket) — upper/truth 1.6–49x. These
  rows are the case for moving to a quantile histogram; BUT it turned out not to
  be necessary for the following reason:
- **Arm agreement 13/13 (100%)** (≥ 95% ✓): because the small-arm decision is
  made not from the histogram but from a bounded count over the value-ordered map
  (`enumerate_up_to(scan_limit)`), the estimation error never leaks into the arm
  choice. The histogram only affects the post-arm's ŝ (the ef'' scale), and the
  error direction there is conservative (an upper bound → ŝ is never
  underestimated → at worst it costs latency, never recall), with the <2k
  fallback as a safety net.
- The correlated Eq∧Range: min-upper is inflated 2.25x (the Fréchet upper bound
  is used rather than independence, so the inflation is again in the conservative
  direction), the arm is correct, and recall is 1.000.
- Maintenance costs +4% on a build, O(log distinct) per insert; DECISIONS #31.


## Segment count × latency/recall curve — 2026-08-18 (SIFT 100K, unfiltered, ef=50)

An input to the merge policy: the same data split with different seal thresholds.

| segments | p50 | p99 | recall@10 | build |
|---------|-----|-----|-----------|------|
| 1 | 57.3µs | 99.2µs | 0.9889 | 16.4s |
| 2 | 110µs | 177µs | 0.9980 | 12.0s |
| 4 | 197.9µs | 321.9µs | 1.0000 | 10.1s |
| 5 | 272.2µs | 557.1µs | 1.0000 | 9.8s |
| 8 | 385.3µs | 596µs | 1.0000 | 8.9s |
| 10 | 466µs | 705.8µs | 1.0000 | 8.4s |

The cost of the ceiling guard (same data, seal=10K):

| | build | segments | search p50 | memory |
|---|---|---|---|---|
| no ceiling | 8.4s | 10 | 446µs | 89 MB |
| ceiling=8 | 12.3s | 8 (2 merges) | 410µs | 88 MB |

Peak merge memory ≈ steady state + 2×source segment (until the swap; +2×9 MB for
10K segments).

Readings:
- The curve is slightly sub-linear but close to linear: ~+45µs per segment (10
  segments cost 8.1x a single segment, not 10x). The shorter traversal of a small
  segment does not fully offset the cost, because each segment is searched at its
  own ef width.
- The recall bonus is real: 1 segment gives 0.9889, ≥4 segments give 1.0000 (a
  candidate pool of 5×ef in total). Merging pays that bonus back.
- **A fair comparison** (at an equal recall of ~0.998): a merged single index at
  ef=100 → 222µs (from the phase 2 sweep); 5 segments at ef=50 → 272µs. The net
  gain of a full merge is ~20%, not a naive "5x". Merging is NOT urgent.
- Build time FALLS with the segment count (16.4s → 8.4s): a small graph is cheap
  to build. Since a merge is a rebuild (there is no cheap way to combine graphs),
  write amplification is worse than in an LSM → the policy must stay
  conservative: merge the oldest/smallest pair once the segment count exceeds a
  ceiling (say 8–10), not on every sealing.


## Filter selectivity sweep + planner — 2026-08-18 (SIFT, k=10, ef=50)

Three match distributions: uniform (in id space), clustered (in vector space:
the nearest s·n neighbours of a centre, with queries grouped near/mid/far by
their distance to that centre), and contig (contiguous in id space). Reference: a
filtered brute-force scan.

### Measurement findings (in-traversal filtering, raw HNSW, 100K)

- The silent recall decline with scale is REAL: clustered×far×s=0.3 → **0.948**
  (the worst at 10K was 0.952), with the fallback counter at 0 in every cell —
  the only signal was the admit/visit ratio collapsing from 0.167 to 0.002.
- The real damage is latency: in the clustered×far cells the traversal spreads
  across the whole graph — at s=0.001 the p50 is **35.3 ms** (an unfiltered
  search is 65 µs).
- The scaled-ef arm (ef'=k/s) was rejected: it only adds latency (up to 39 ms) to
  a mechanism that was already preserving recall.
- An O(n) planning count was rejected: 14.4 ms at 100K.
- The first planner attempt (a visit budget of 24·ef/√ŝ + a scan fallback) worked
  at 10K but regressed at 100K through wrong cutoffs → it was removed from the
  production path (it remains as instrumentation).

### The final planner (posting list + scan / unfiltered over-fetch)

10K: recall 1.000 in ALL 21 cells; the worst p50 is 1.03 ms (the scan baseline).
Sample cells at 100K (before = raw in-traversal, after = with the planner):

| cell | before recall/p50 | after recall/p50 |
|---|---|---|
| uniform s=0.001 | 1.000 / 20.2ms | 1.000 / 12µs |
| clustered×far s=0.001 | 1.000 / 35.3ms | 1.000 / 12µs |
| clustered×far s=0.01 | 1.000 / 25.8ms | 1.000 / 179µs |
| clustered×far s=0.05 | 1.000 / 20.3ms | 1.000 / 1.03ms |
| clustered×mid s=0.1 | 0.997 / 866µs | 0.997 / 1.6ms |
| clustered×far s=0.3 (the critical cell) | **0.948** / 13.6ms | **1.000** / 10.8ms |
| clustered×far s=0.5 | 0.985 / 10.1ms | 1.000 / 20.4ms |
| contig s=0.5 | 0.998 / 127µs | 1.000 / 621µs |
| the s=1.0 rows | 0.989 / ~90µs | 1.000 / ~550–610µs |

Note: the lowest cell at 100K is 0.988 (clustered×mid s=0.3) — above the old
baseline of 0.948.

### The scan_candidates optimization + the ŝ≈1 shortcut (same day, second round)

The first scan arm was ~4x brute force (re-checking metadata per id + a hash
probe per source + a full sort). The fixes: for a single-Eq filter the posting
list is taken as the exact set (no re-check), a source-outer loop (a found id is
not tried again), and a top-k heap. In addition, for a single Eq with est=n the
filter is behaviourally empty → a shortcut to the unfiltered `search_shared`.

| cell (100K) | before opt. | after |
|---|---|---|
| the scan band (s≤0.05) | 12µs–1.03ms | 7.6µs–440µs (~2.4x) |
| clustered×far s=0.3 | 10.8ms | **3.9ms** (raw in-traversal: 13.6ms/0.948) |
| clustered×far s=0.5 | 20.4ms | **6.25ms** (raw: 10.1ms/0.985 — now both faster and recall 1.000) |
| the s=1.0 rows | 550–610µs | 430–530µs = the unfiltered segmented baseline |

An explanation of s=1.0: the "6x against 65µs" comparison in the first report was
misleading — 65µs was the unfiltered p50 of a SINGLE HnswIndex, whereas the
unfiltered baseline of the segmented index is already ~470µs (5 segments × an ef
search; consistent with the 1893 QPS ≈ 528µs from the concurrency measurement).
After the shortcut, s=1.0 sits on that baseline; the regression was not
structural, it was an error of comparison. The recall baseline did not change:
0.988.


## SIMD — 2026-08-18 (wide f32x8 + target-cpu=native)

Micro (128d):

| function | before | after | speedup |
|---|---|---|---|
| dot | 60.4 ns | 6.4 ns | 9.4x |
| l2_squared | 65.2 ns | 7.4 ns | 8.8x |
| ADC (quantized L2) | ~130 ns (scalar) | 15.6 ns | ~8x |

End to end (SIFT 100K, M=16/ef_c=200):

| measurement | before | after |
|---|---|---|
| HNSW build | 40.4 s | 17.1 s |
| HNSW p50 (ef=50) | 124.3 µs | 52.7 µs |
| int8 p50 (ef=50) | 142.9 µs | 61.2 µs |
| brute-force p50 (rayon) | 547 µs | 160 µs |
| the recalls | — | identical (determinism preserved) |

Note: `target-cpu=native` alone gained nothing — because `map().sum()` fixes the
float summation order, LLVM cannot vectorize the reduction. The gain came from an
explicit f32x8 with two accumulators (the change in order produces a ~1 ulp
difference, irrelevant for distance comparison). Brute force is now so fast that
HNSW's apparent "speedup factor" at 100K has fallen — an expected rebalancing,
since both sides use the same kernel.


## The full SIFT1M set — 2026-08-18 (stress test, the official ivecs ground truth)

M=16, ef_c=200. Build: **802 s (13.4 min)**, with a constant time per segment
(~80 s/100K) — the worry that it would take hours was not borne out.

| index | memory | ef=50 | ef=100 | ef=200 |
|---|---|---|---|---|
| f32 | 496 MB vectors + 383 MB graph | 0.9680 / 296µs | 0.9900 / 480µs | 0.9960 / 782µs |
| int8 | 122 MB codes + 252 MB graph | 0.9630 / 325µs | 0.9830 / 479µs | 0.9890 / 778µs |

(cells: recall@10 / p50). The quantization conversion took 1.1 s. The loss stays
≤ 0.011 at 1M too. Note: the int8 graph memory looks smaller because the Vec
capacities settle at their exact size during the copy (the f32 side carries growth
slack).


## Phase 6 — 2026-08-18 (scalar quantization f32→int8, ADC, M=16/ef_c=200)

Pure quantization (no rerank); the graph was built in f32 and then frozen.
Calibration + encoding: 10K → 7 ms, 100K → 92 ms.

### SIFT 10K

| index | ef | recall@10 | p50 | p99 | vectors MB | total MB (incl. graph) |
|--------|----|-----------|-----|-----|-----------|------------------------|
| f32 | 50 | 0.9990 | 65.3µs | 90.6µs | 5.0 | 8.9 |
| int8 | 50 | 0.9890 | 79.5µs | 124.4µs | 1.2 | 3.7 |
| f32 | 100 | 1.0000 | 107.1µs | 131.8µs | 5.0 | 8.9 |
| int8 | 100 | 0.9900 | 124.8µs | 192.8µs | 1.2 | 3.7 |

### SIFT 100K

| index | ef | recall@10 | p50 | p99 | vectors MB | total MB (incl. graph) |
|--------|----|-----------|-----|-----|-----------|------------------------|
| f32 | 25 | 0.9660 | 85µs | 274µs | 49.8 | 88.3 |
| int8 | 25 | 0.9610 | 91.2µs | 318.7µs | 12.2 | 37.6 |
| f32 | 50 | 0.9890 | 129µs | 400.7µs | 49.8 | 88.3 |
| int8 | 50 | 0.9800 | 142.9µs | 366.5µs | 12.2 | 37.6 |
| f32 | 100 | 0.9980 | 222.3µs | 477µs | 49.8 | 88.3 |
| int8 | 100 | 0.9870 | 225.9µs | 556.5µs | 12.2 | 37.6 |

Acceptance check:
- Vector data memory: 49.8 → 12.2 MB = a **4.1x reduction** ✓ (target 4x)
- Total index (including the graph adjacency): 88.3 → 37.6 MB = 2.35x — because
  the graph stays at a constant ~404 B/vector, the total ratio is lower than the
  vector ratio.
- recall@10 loss: between 0.005 and 0.011, all **< 0.02** ✓
- Latency is ~5–10% higher: ADC does an extra mul+add per element
  (dequantization), and at 128d the bandwidth gain does not yet offset that.


## Phase 5 — 2026-08-18 (segment-model concurrency, SIFT 100K)

The structure: 5 × 20K HNSW segments plus a brute-force write buffer.
recall@10 = 1.0000 (a segment-merged search at ef=50; with small segments recall
comes out higher than with a single large index, because the search runs at ef
width five times).

| Scenario | Throughput |
|---|---|
| 1 reader thread | 1893 QPS |
| 4 readers | 8303 QPS (4.4x — it scales, readers do not block one another ✓) |
| 8 readers | 16460 QPS (8.7x) |
| 4 readers + an active writer (continuous delete+insert, sealings included) | 3018 QPS |

Notes:
- The drop in the writer scenario comes from CPU sharing rather than lock
  contention: the writer thread does sealing builds in a hot loop (a ~2 s HNSW
  build every 10K inserts) and steals cores. Searches never STOP for the duration
  of a sealing — with a single-RwLock approach that 2 s build would freeze every
  search; that is the measurable difference.
- The stress test (4 readers + 1 writer, 3K inserts + intermittent deletions, 5+
  sealings): no panic, and the result invariants (no duplicates, no NaN, ordered)
  were verified on every query.


## Phase 4 — 2026-08-18 (deletion + compaction, SIFT 10K, M=16, ef=50)

| Measurement | Value |
|---|---|
| recall@10 before deletion | 0.9990 |
| recall@10 after 20% deletion | 0.9990 (no degradation ✓) |
| compaction time | 1.9 s (rebuilding 8K live elements) |
| memory (vectors) | 5.0 → 4.0 MB (−20% ✓) |
| memory (graph) | 3.8 → 3.1 MB ✓ |
| recall@10 after compaction | 1.0000 |

The entry-point deletion scenario is covered in a separate test: the new entry is
the highest-level live node and search continues uninterrupted
(`delete_entry_point_picks_new_entry_and_search_works`).


## Phase 3 — 2026-08-18 (persistence, SIFT 100K, M=16/ef_c=200)

| Measurement | Value |
|---|---|
| save | 121 ms |
| load (full read; the mmap permission is pending) | 73 ms |
| file size | 71.8 MB (753 B/vector: 512 B of data + graph + ids) |
| results after reloading | identical across 100/100 queries ✓ |
| a truncated/corrupt file | no panic, Err (test + a proptest mini-fuzz) ✓ |


## Phase 2 — 2026-08-18 (HNSW, SIFT1M subsets, k=10, L2)

The brute-force reference (rayon, all cores): 10K p50=617µs; 100K p50=547µs. The
HNSW search is single-threaded. "speedup" = bf_p50 / hnsw_p50.

### SIFT 10K (acceptance: recall ≥ 0.95 ✓)

| M | ef_c | ef_search | recall@10 | p50 | p99 | speedup | build | graph B/vector |
|---|------|-----------|-----------|-----|-----|----------|------|----------------|
| 8 | 100 | 10 | 0.8830 | 15.8µs | 22µs | 39.1x | 1.0s | 233 |
| 8 | 100 | 25 | 0.9700 | 26.7µs | 40.4µs | 23.1x | 1.0s | 233 |
| 8 | 100 | 50 | 0.9890 | 44.3µs | 57µs | 13.9x | 1.0s | 233 |
| 16 | 200 | 10 | 0.9500 | 22.1µs | 32.3µs | 27.9x | 2.3s | 403 |
| 16 | 200 | 25 | 0.9940 | 40.3µs | 53µs | 15.3x | 2.3s | 403 |
| 16 | 200 | 50 | 0.9990 | 64.2µs | 82.3µs | 9.6x | 2.3s | 403 |
| 32 | 400 | 10 | 0.9860 | 32.3µs | 46.9µs | 19.1x | 5.6s | 735 |
| 32 | 400 | 25 | 0.9990 | 55.3µs | 78.9µs | 11.2x | 5.6s | 735 |

### SIFT 100K (acceptance: ≥10x faster ✓, build within minutes ✓)

| M | ef_c | ef_search | recall@10 | p50 | p99 | speedup | build | graph B/vector |
|---|------|-----------|-----------|-----|-----|----------|------|----------------|
| 8 | 100 | 25 | 0.9100 | 47.3µs | 64.9µs | 11.6x | 14.4s | 233 |
| 8 | 100 | 50 | 0.9740 | 73.9µs | 98.3µs | 7.4x | 14.4s | 233 |
| 16 | 200 | 10 | 0.8770 | 43.1µs | 69.5µs | 12.7x | 40.4s | 404 |
| 16 | 200 | 25 | 0.9660 | 81.7µs | 118.2µs | 6.7x | 40.4s | 404 |
| 16 | 200 | 50 | 0.9890 | 124.3µs | 168.3µs | 4.4x | 40.4s | 404 |
| 16 | 200 | 100 | 0.9980 | 218.2µs | 297.4µs | 2.5x | 40.4s | 404 |
| 32 | 400 | 25 | 0.9860 | 110.7µs | 145.7µs | 4.9x | 106.8s | 740 |
| 32 | 400 | 50 | 0.9960 | 188µs | 248.6µs | 2.9x | 106.8s | 740 |

Notes:
- The graph's memory cost (M=16): ~404 B/vector — about 79% overhead on top of
  the 512 B/vector of raw data. Choosing M=8 brings it down to 233 B, at the cost
  of raising ef_search for recall.
- The brute-force reference is MULTI-core; against a single thread the speedup
  would be far higher (on the order of 5.5ms/547µs). The 10x acceptance was met
  under the conservative reading.
- The sweet spot: M=16, ef_c=200, ef_search 25–50 (recall 0.966–0.989, 4–7x).


## Phase 1 — 2026-08-18 (the brute-force index, SIFT1M subsets)

Data: the first n vectors of the SIFT base, 100 real SIFT queries, k=10, L2. The
ground truth was generated over the subset with an exact scan (the bundled GT is
for the 1M base).

| Measurement | SIFT 10K | SIFT 100K |
|---|---|---|
| recall@10 | **1.0000** | **1.0000** |
| search p50 | 611.7 µs (serial path) | 672.7 µs (rayon parallel) |
| search p99 | 776.7 µs | 1.09 ms |
| build time | 2.2 ms | 21.4 ms |
| index memory | 8.5 MB (886 B/vector) | 67.6 MB (709 B/vector) |
| raw data | 512 B/vector | 512 B/vector |

Notes:
- The per-vector overhead (886/709 vs 512 B) comes from `Vec` capacity growth
  slack plus the id map; acceptable for brute force, and reported separately for
  HNSW.
- 10x the data ≈ the same p50: above the 20K threshold the scan is distributed
  across rayon (a local top-k heap per parallel chunk plus a lock-free merge).


Environment: Windows 11, rustc 1.97.1, the release profile. Seed = 42,
reproducible.

## Phase 0 — 2026-08-18 (validating the measurement infrastructure, random data)

Data: 10,000 × 128d random vectors (uniform [-1,1)), 100 queries, k=10, metric
L2. There is no index yet; what is measured is the reference exact scan
(`eval::exact_top_k`).

| Measurement | Value |
|---|---|
| recall@10 (exact vs exact GT) | 1.0000 (a pipeline check) |
| latency p50 (single-thread exact) | 634.6 µs |
| latency p99 | 666.7 µs |
| ground-truth generation (100 queries, rayon) | 6.4 ms |
| memory (raw f32 vectors) | 512 bytes/vector (128 × 4B), 4 MB in total |
| build time | — (no index) |

Criterion micro-bench (128d, a single pair of vectors):

| Function | Time |
|---|---|
| dot | ~58.4 ns |
| l2_squared | ~61.3 ns |
| cosine (pre-normalized, -dot) | ~57.8 ns |

Note: cosine costing the same as dot is the expected consequence of our
normalization policy (normalize at insert time); no norm is computed during
search.
