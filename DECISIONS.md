# Architectural Decisions

## Phase 9c — metadata memory compaction (DECISIONS) — 2026-08-19

### 64. Posting lists: `HashSet<VectorId>` → sorted `Vec<VectorId>`
In sealed segments the set never changes; membership is O(log n) via binary
search and exactly 8 bytes are held per id (16 plus load-factor slack plus a
table header in a `HashSet`). The gain came out smaller than expected
(353 → 299 MB): the dominant term is not the INSIDE of the sets but the
**number of distinct (field, value) keys** — every distinct value of a numeric
field produces a posting entry, so the outer `HashMap` dominates. The right
item was fixed correctly, but the distribution *within* that item differed from
the estimate.

**Known cost (measured):** inserting into a sorted `Vec` costs an O(n) shift.
If ids arrive in ascending order the position is always at the end (O(1)); in
random order it is **8.8x** slower (144 ms → 1.27 s for a single 200K list).
The O(n²) growth does not bite at this size because the list (1.6 MB) fits in
cache; a SINGLE 1M-entry list would take ~30 s. Accepted: at that selectivity
(millions of records under one value) the scan arm never engages anyway.

### 65. id→metadata: a schema-based compact representation (`MetaStore`)
Instead of a `HashMap<String, MetaValue>` per record: field names stored ONCE in
a dictionary (u32 ids), with the record body as `Box<[(u32, MetaValue)]>`.
`Box<[T]>` was chosen because it has no capacity slack and its header is 8 bytes
smaller than `Vec`'s. Lookup is LINEAR because there is a handful of fields per
record — building a hash table at that size does not pay off, and that was the
problem in the first place.

**421 → 158 MB (2.7x)**, the real gain of 9c. The predicate logic was kept in
one place (`Predicate::matches_with`, independent of the access path), otherwise
two stores would have become two separate filter implementations.

### 66. `metadata_memory_bytes()` SYSTEMATICALLY under-reports
In the 9c-0 validation it was off in the same direction for ALL THREE items
(0.77x in total). With random error the directions would have been mixed. The
likely cause: allocator overhead and the empty buckets of a `HashMap` are not
counted.

Fixing it was left out of scope, but it goes on record: **the same function is
also used by `/stats`**, so the user is systematically shown a LOW memory
figure. Anyone doing capacity planning with it will find real usage is about
1.3x what was displayed.

### 67. Numeric indexes DEFERRED — the reason is risk/reward, not memory share
197 MB (measured), about 16% of the metadata total. Left untouched.

The reason is **not the memory share**: that structure (`BTreeMap` + histogram)
feeds the **exact-counting arm** of the filter planner — it sits underneath the
most delicate correctness mechanism in the project. Reaching for the riskiest
component on the way to closing out is the wrong timing.

**Revisit condition:** if memory genuinely becomes constraining AND the
regression tests for the exact-counting arm can be strengthened in the same
round.

### 68. The remaining share is still above the threshold: the next step was written and DEFERRED
The real share went from 41.0% to **33.3%** (threshold 25%). Per the rule, the
next step is written down and deferred:

**Next step — stop keeping Eq postings for numeric fields.** The two largest
remaining items (the posting lists' outer `HashMap` and the numeric indexes)
share one root cause: every distinct value of a numeric field produces BOTH a
posting key AND a `BTreeMap` entry. Range is already served from the numeric
index; numeric Eq could be served from there too. This would be **the removal of
a duplication**, not an optimization.

### 69. A `with_capacity` regression: lazy allocation disappeared
The `with_capacity` work from #61 caused a **panic with a capacity overflow**
wherever the threshold is given as `usize::MAX` (measurement/test: "no sealing in
practice") — allocation used to be lazy. The allocation is now bounded in bytes
(512 MB); above that bound `Vec` falls back to its old incremental growth. A
regression test was added (`huge_seal_threshold_does_not_panic`).

The lesson: a performance fix can change the behaviour of extreme inputs
OUTSIDE the path being fixed.


## Phase 9a-2 — evaluating the criteria (DECISIONS) — 2026-08-19

The measurement results are in BENCHMARKS and in a SEPARATE commit (ef2ce04).
This section holds the decisions drawn from those results.

### 60. Criterion 1 (#40) NOT MET — and the world underneath the threshold changed
The result STAYS "not met" (609x / 943x, threshold 50x). The threshold was not
reinterpreted. A defect record is added beside it:

**#40 was defined in a system where backpressure DID NOT EXIST.** Back then a
long write was ALWAYS the symptom of a defect — the writer never waited on
purpose. The moment backpressure arrived with #53, that ontology changed: a long
write can now be proof that the system is working CORRECTLY (the only way to
bound the queue is to stall the writer). Indeed, in the WAL-off diagnostic run
the longest write was 24.9 s, and all of it was designed backpressure.

So the "longest write" metric now sums TWO DIFFERENT PHENOMENA (a defect and a
deliberate restriction) into one number. This is **structurally the same class
of error as the criterion-2 flaw in #58, only inverted**: there we summed two
different QUANTITIES (queue + segments), here two different CAUSES.

**This is not an error in the threshold; it is the world underneath the
threshold changing.**

**AND THIS MUST STAND CLEARLY: 9a-2 HIT its target.** The time sealing blocks
the writer went from 20.8 s to **0 ns – 2 µs**. The threshold not being met is
unrelated to that; the remaining 6–10 ms spikes occur at points unrelated to
sealing and come from fsync plus (hypothesis) buffer realloc. Someone reading
the table six months from now must NOT conclude that 9a-2 failed.

### 61. NEW PRE-REGISTRATION — the second version of criterion 1 (written BEFORE the measurement)
- **Primary:** the longest write **EXCLUDING backpressure** / baseline p99.
  Backpressure duration is a design parameter (a function of the queue threshold
  and the sealing rate), not a defect — it is separated out in the measurement.
- **Secondary (this round it is ONLY MEASURED, NO THRESHOLD):** the distribution
  of backpressure-induced waits — how many writes were affected and how long the
  longest wait was. Rationale: removing backpressure from the measurement
  entirely would make "the system stalls writes for hours but the criterion
  passes" possible. There is not enough data to set a threshold; the threshold is
  left to the NEXT round.
- **Condition:** the measurement is run under BOTH fsync policies (group:20 and
  off), so the fsync contribution separates from the buffer-realloc contribution.
- **Work to do first:** `with_capacity` on the write buffer (which removes the
  realloc). Then measure; the realloc hypothesis is either proven or refuted.

### 62. Segment accumulation is a SEPARATE ITEM — not in this arc
The numbers: sealing PRODUCES a segment about every 25 s, merging REMOVES one
about every 54 s. Under sustained heavy load, accumulation is inevitable (0 → 16
over ten minutes).

This is not 9a-2's problem but the **merge ceiling mechanism's** problem; tying
it to 9a-2's acceptance would lock two separate pieces of work together. But the
record must be strong: **this is currently the one dimension of the system that
grows without bound**, and we had closed exactly this before (the segment
ceiling). It came back because **sealing got faster and merging did not** — a
side effect of 9a's own success.

Possible fixes (both separate design work): parallel merging, or having a merge
take three segments instead of two.

**Revisit condition:** "if sustained write load becomes part of the intended
usage". In the target scenarios (RAG, internal tooling) there is no sustained
5K op/s write stream — a real problem, but a THEORETICAL one for now.

### 63. A known limit of pre-registration discipline: criteria can conflict
The real lesson of this round: **the mechanism that made criterion 2 pass made
criterion 1 impossible to pass.** Criteria are written independently, but the
system is a single whole. Recording the defect and moving on is the right
answer; from now on, though, writing a threshold also means asking:

> **"Which other criterion could this one conflict with?"**

This is a permanent clause of the pre-registration rule.


## Phase 9a-2 — the second criterion-2 measurement + a new pre-registration — 2026-08-19

### 56. The backpressure signal was chosen wrongly: designing "slow down" and getting "stop"
The first backpressure threshold was **sealing + segments > 2×ceiling**. That
signal is wrong: the sum does not drop when a sealing FINISHES — the element
moves from the queue into the segments and the sum stays constant. Only a merge
lowers it, and a merge runs only once the segment count exceeds the ceiling. The
result in the 1M measurement: the writer was at **0 op/s for 110 of 120
seconds**, and two inserts hit the 60 s safety limit. Without that limit it would
have hung forever in production.

**The general rule (the lesson from this project):** the signal of a control
loop must be a quantity that the loop CAN INFLUENCE. The sum did not fall as the
writer slowed down — there was no feedback, only a wall. Designing "slow down"
and getting "stop" is a classic class of error in distributed systems.

**The fix:** the signal is the **queue length alone**, threshold 2 (one being
built while one waits). The merge ceiling already bounds the segment count; the
dimension growing without bound was the queue. The result: the queue held steady
at 3 and the write rate settled at the sealing rate (~5K op/s) instead of
collapsing to zero.

### 57. Acceptance criteria are measured AT SCALE — and measurement, not tests, caught this
Unit tests could not catch the flaw in #56: at small scale merging turns over
quickly, the sum really does fall, and equilibrium is reached. At 1M, where
sealing takes ~20 s, that equilibrium breaks. The tests were green; the system
was not working. This is the best example of the rule "develop at 10K, accept at
100K/1M".

### 58. The metric flaw in pre-registration #49 — the RESULT IS AGAIN "NOT MET"
In the second measurement (corrected signal, 2 min, 1M): the queue held steady
**at 3**, the peak was 9 ≤ 12 (**OK**), but the first-third → last-third average
went 3.8 → 7.9 (**+110%**, threshold +20%) → **EXCEEDED**. As with #40, the
result STAYS "not met"; the threshold was not reinterpreted.

**Defect analysis (by the user, who wrote the threshold):** the sum
`segments + sealing` adds two numbers from DIFFERENT REGIMES — the segment count
is already bounded by the merge ceiling, whereas the queue was (at that time)
unbounded. What was meant to be measured was "is there a dimension growing
without bound"; what was measured was the sum of the two. This record does NOT
CHANGE the threshold; it documents the gap between what the threshold meant to
measure and what it measured.

### 59. NEW PRE-REGISTRATION — 9a-2 criterion 2, second version (written BEFORE the measurement was run)
**Transparency note:** both this pre-registration and the decision to extend the
duration were taken AFTER seeing the result of the 2-minute measurement. The
reason: a 2-minute window was not enough for the segment count to reach the merge
ceiling, so the curve was misread (growth appeared as monotone increase rather
than as approaching a ceiling). Let the reader apply their own discount.

- **Primary criterion (9a-2 itself):** under 10 minutes of sustained full-speed
  writing, the **queue length** settles at a fixed upper bound (observed bound:
  3). If it grows monotonically → not met.
- **Secondary item (a test of the merge ceiling, NOT of 9a-2):** the segment
  count stays within the merge ceiling + tolerance (8+4=12). Recorded as a
  separate item; it does not determine 9a-2's acceptance.
- **Duration 10 minutes**, because the segment count needs to reach the ceiling
  and trigger merging.
- Criterion 1 (latency, #40) is unchanged and is measured separately. 9a-2 is
  accepted only if the primary criterion and criterion 1 both pass.


## Phase 9a-2 — single worker + backpressure — 2026-08-19

### 53. Sealing on a SINGLE worker + queue; backpressure on the write path
The fix for the flaw in #52. `seal()` no longer spawns threads; `SealContext`
(the twin of the `MergeContext` pattern used for merging: a CAS flag, a single
loop that runs until the queue drains, and a double check when releasing the
flag) consumes the `sealing` list FIFO. Concurrent construction did not reduce
the total work, it only slowed all of it down; a sequential worker does the same
work but each sealing finishes IN TURN, so the queue is drained and memory is
returned.

**Backpressure:** once the queue plus segment count exceeds 2×ceiling, the write
path waits in 1 ms sleeps (Lucene's `IndexWriter` stall). Writes are not
rejected, they are slowed; the single-writer contract is preserved; the wait
holds no lock, so readers are unaffected. Upper limit 60 s (so the writer does
not wait forever if the worker unexpectedly fails to progress). Observability:
`stall_stats()`.

### 54. LOCK ORDER (a global rule): segments → sealing → buffer
Writing this fix produced TWO deadlocks, both of which surfaced in the tests as a
**hang** (not a panic — a silent hang):

1. `sealing.read().len() + segments.read().len()` — temporaries in a single
   expression live until the end of the statement, so the two locks are taken
   NESTED, in the reverse order from the worker's. **Rule: never take two locks
   in one expression**; assign them on separate lines so each guard drops
   immediately.
2. `while let Some(x) = sealing.read()...first().cloned()` — `while let`
   desugars to `loop { match EXPR {...} }`, and the temporaries of the match
   scrutinee live **including through the body**; so the read lock is held while
   `build_one` asks for the write lock. A plain `while COND` is safe (the
   condition's temporaries drop as soon as it is evaluated) — which is why the
   merge worker was sound. Rust 2024 fixed this for `if let`, but `while let`
   still has the old behaviour. **Rule: take a lock-acquiring scrutinee on its
   own line, separated with `let ... else`.**

So that the same mistake is not repeated when a fourth lock appears, the order is
fixed here: **segments → sealing → buffer**. If more than one is needed they are
taken in that order; where possible they are not nested at all.

### 55. A timeout was added to CI
The symptom of a deadlock is a hang; without a timeout, CI waits the default six
hours and the cause stays unclear. 15 min for the job, 10 min for the test step.
Cheap insurance in a project that is adding concurrency code (user suggestion).


## Phase 9a-2 — NOT ACCEPTED (criterion 2) — 2026-08-19

### 51. Sealing was moved to the background, but 9a-2 is NOT ACCEPTED yet
The code and tests are complete (below), but **the second criterion of
pre-registration #49 was exceeded**: under 60 s of full-speed writing the sealing
queue went from 0 to **35** (+90% growth against a +20% threshold; peak 35
against a threshold of 12). Per the pre-registration, **backpressure is part of
9a-2**, and 9a-2 is not accepted until it is done. The threshold was not
reinterpreted; the result was recorded as "not met".

**What was done (it stands, and works correctly):**
- `seal()` only swaps the buffer on the writer task (µs); the HNSW build runs in
  the background. The window in which the writer is blocked has practically
  vanished.
- The "two buffers" state is handled correctly on all three paths (DECISIONS
  #50): search (`search_shared_with_ef`, both arms of the filtered search,
  `scan_candidates`), duplicate-id (`validate_insert` →
  `sealing_contains_live`) and delete (buffer → sealing → segments; a tombstone
  in the one being sealed plus diff-replay when the build finishes).
- `checkpoint()` now calls `wait_for_background()`: without waiting, the data in
  the `sealing` list would be in no segment, the manifest would not see it, and
  the WAL rotation would orphan it — silent data loss.
- The tests follow the 9a-1 pattern and assert that the window actually occurred
  (`seal_in_flight() > 0`, `saw_sealing > 0`).

### 52. The DESIGN FLAW the measurement exposed: unbounded sealing threads
In the accumulation measurement the segment count **stayed at 0** for the full
60 s: 35 sealings ran at once and, sharing 8 cores, none could finish. The cause
is that `seal()` calls `thread::spawn` on every invocation. This is a flaw
independent of backpressure and must be fixed first.

**Design of the next piece of work (in a new session):**
1. **A single sealing worker + queue** (the `MergeContext` pattern used for
   merging): one worker instead of unbounded threads, sealing the queued buffers
   in order.
2. **Backpressure:** once the queue length crosses a threshold, a short wait on
   the write path (like Lucene's `IndexWriter` stall) — slowing writes down, not
   rejecting them. The single-writer contract is preserved.
3. Then BOTH criteria of pre-registration #49 are measured again (latency +
   accumulation); only then is 9a-2 accepted.


## Phase 9a-2 PRE-REGISTRATION — 2026-08-19 (before implementation and measurement)

### 49. 9a-2 has TWO acceptance criteria: latency AND accumulation
9a-2 moves sealing into the background too. While that helps with the latency
threshold, it carries the risk of **making another dimension of the system
unbounded**: the writer will no longer wait for any long operation, i.e. it will
accept writes at an unbounded rate, while the background build work proceeds at a
fixed rate.

The accumulation arithmetic (from the existing measurements): at ~100K op/s the
writer fills a 125K buffer in **~1.3 s**, while sealing takes **~25 s**. If that
ratio holds, the sealing queue grows monotonically → the segment count rises,
search slows and memory swells. This is the "one dimension growing without
bound" problem closed in #30, returning through a different door.

**Criterion 1 — latency (pre-registered in #40, unchanged):** the p99 of writes
coinciding with the sealing/merge window must not exceed **50x** the baseline
p99. The measurement condition is the same as in phase 8 and 9a-1 (WAL group:20,
no commit inside the loop; an isolated process, warmup, two runs).

**Criterion 2 — accumulation (NEW, fixed by this pre-registration):** while the
writer writes AT FULL SPEED, without interruption, for **at least 2 minutes**,
the segment count is sampled every 5 seconds. Outcome:
- **STABILIZES** (the average of the last third exceeds that of the first third
  by at most 20%) **AND** no sample exceeds ceiling+4 (ceiling 8 → **12**) →
  9a-2 is ACCEPTED without backpressure.
- **GROWS MONOTONICALLY** or exceeds ceiling+4 → **backpressure is part of 9a-2
  and is done in the SAME arc**; without it 9a-2 is not accepted. (This is no
  longer a "revisit condition" but an acceptance criterion.)
- Peak RSS during the accumulation is reported as well.

If backpressure is implemented, its form: once the segment count exceeds
2×ceiling, a short wait on the write path (like Lucene's `IndexWriter` stall) —
slowing writes down, not rejecting them. The single-writer contract is preserved.

### 50. The structural risk of 9a-2: the "two buffers" state
Once sealing moves to the background, the buffer being sealed and the new write
buffer coexist for a while. This affects ALL THREE sources:
- **Search:** segments + the buffer being sealed + the new buffer must all be
  walked.
- **Delete:** the record goes wherever it lives (a tombstone in the one being
  sealed).
- **Duplicate-id (the sneakiest):** if an id in the buffer being sealed is
  inserted a second time into the new buffer and the check only looks at the new
  buffer, the collision surfaces only AFTER sealing finishes — and by then both
  copies are permanent.

The tests are written in the 9a-1 pattern: that the race actually occurred (**"the
two-buffer state was observed"**) is asserted inside the test, otherwise the test
silently weakens.


## Phase 9a-1 — merging in the background — 2026-08-19

### 46. Merging was separated from the writer task; the remaining window is entirely sealing
Implementation: the `segments` field became `Arc<RwLock<...>>` (auto-deref meant
existing calls were unchanged), merging moved into the free function
`merge_smallest_pair_bg`, and it runs on a background thread via
`spawn_merge_if_needed`. "At most one merge at a time" is enforced with a CAS
flag; if the ceiling is exceeded again the worker loop continues (no new thread
is spawned, and triggers queue up naturally).

**Measurement (BENCHMARKS, two runs):** the longest window in which the writer is
blocked went **80.5 s → 28.5/30.6 s (a 2.8x reduction)**; ALL of the remaining
time is sealing. Merging now takes 53–54 s but runs **in the background** (the
overlap was confirmed by "was a merge running as the measurement ended: YES").

- **The pre-registered 50x threshold was NOT MET** (the ratio is ~2.8–3.6
  million x). This was already anticipated in the pre-registration: "it will not
  be met in 9a-1; report it." The threshold will be tested again after 9a-2; the
  only remaining obstacle is sealing.
- **An honest cost:** sealing itself went from 20.8 s to 28–30 s (+40%). The
  reason is that merging now runs in parallel and shares the CPU. "Moving it to
  the background is not free" — the net gain is still 80.5 → 29 s.

### 47. The REAL acceptance criterion: the tombstone diff-replay race (passed)
A sealed segment takes no inserts but it does take **deletes**. If new tombstones
land on the sources while the merge rebuild runs for seconds, those records are
copied into the merged segment as LIVE and deleted records **silently come
back** — no latency measurement would catch that. The fix: a SNAPSHOT of the
tombstones is taken at the start of the build, and at swap time, under the write
lock, the DIFFERENCE against the current tombstones is computed and applied to
the merged segment (diff-replay).

The race window is closed **structurally**: `delete_vector_only` writes the
tombstone while holding the `segments` READ lock, and the swap takes the WRITE
lock → the two are mutually exclusive. The interleaving "I read the diff, then a
tombstone arrived, then I swapped" cannot occur.

The test (`merge_carries_tombstones_created_during_build`) asserts via
`during_merge > 0` that the race was actually triggered — otherwise the test
would silently weaken. Search consistency during a merge, and recovery from a
crash mid-merge, are covered by separate tests.

### 48. New behaviour: the segment count can temporarily exceed the ceiling
Because merging is asynchronous, when sealing outpaces merging the segment count
rises above the ceiling (9–11 segments were observed in the measurement); once
writing stops the worker brings it back down. This is normal in Lucene-like
systems too. **Deferred work:** under a sustained high write rate the
accumulation could be unbounded; if needed, backpressure is added (e.g. slowing
writes once the segment count exceeds 2×ceiling). Revisit condition: if under a
sustained write load the segment count exceeds 2×ceiling.


## Phase 8a — int8 scaling — 2026-08-19

### 44. CORRECTION: the finding "reads do not scale at 1M" (#43) was WRONG
The mixed-load section of phase 8 measured 945 QPS for 8 readers and concluded
"scaling collapses, a memory-bandwidth wall". An isolated measurement refuted
this: **f32 scales 5.4–6.1x at 1M** (8 physical cores). The source of the error
was the measurement environment: that table was taken as section 8 of
`fullscale`, in a process that had been running for five minutes and had done a
1M build, 130K writes, a merge and three cold starts, with RSS at 3.1 GB.

- **The part that remains valid:** the fsync-policy comparison in that same table
  (ratio 0.99–1.01) — since all three policies were measured in the same dirty
  process, the *relative* result holds. The decision "fsync does not burden
  readers" stands.
- **The invalid part:** the absolute QPS values and the "scaling collapses"
  interpretation.
- **The lesson (methodology):** performance measurements must be taken in an
  isolated, fresh process, with warmup and the median of repeats. A throughput
  measurement appended to the end of a long run like `fullscale` does not measure
  what you think it measures. This lesson was also written into the `int8scale`
  mode as a code comment.
- **The L3 explanation:** the 8.7x scaling at 100K came from the working set
  (92 MB) fitting in the machine's **96 MB L3** (Ryzen 7800X3D, 3D V-Cache). At
  1M it does not fit and scaling drops from 8.7x to ~5.5x: that is the real
  effect, not a "collapse".

### 45. The 8a decision: int8 integration will NOT be done on performance grounds
The pre-registered threshold (from the plan): 8 threads / 1 thread ≥ 2.0 → GO.
**Measured: 2.75–3.58x, so the threshold was technically MET.** But the
assumption underlying the threshold ("f32 does not scale at 1M, int8 will bring
scaling back") was refuted by #44. The threshold is not changed; the result is
recorded as it stands, and the decision follows the collapse of the assumption:

- **int8 scales LESS than f32** (2.75–3.58x vs 5.40–6.12x).
- **int8 is about 2x SLOWER in absolute terms**: the 8-thread QPS ratio is
  0.46–0.63x.
- The cause is the dequantization arithmetic of ADC (phase 6: 15.6 ns vs
  7.4 ns). On a single thread the memory advantage offsets it; with many threads,
  once the CPU is the bottleneck, ADC dominates.
- **Decision:** "int8 segmented integration for 1M+" does NOT enter the backlog
  as a performance item. The memory rationale (a 2.00x working set) has been known
  since phase 6 and is a separate decision.
- **Revisit conditions:** (a) if ADC's SIMD is improved (the u8→f32 conversion is
  currently scalar per lane; a fully vectorized path has not been measured),
  (b) if the data no longer fits in RAM, the memory rationale outranks the
  performance one, (c) if the graph representation is shrunk too (u32 slots in
  the adjacency lists) the working set could approach L3 — at which point the
  table becomes meaningful again.

## Phase 8 results — go/no-go decisions — 2026-08-19

### 42. Threshold-based decisions (per pre-registration #40)

**9a — moving merging to an independent task: DONE UNCONDITIONALLY (per the
pre-registration).** The rationale was measured: the longest single write is
**80.5 s** (baseline p99 7.8 µs), a ratio of 10.3 million x. A client inside that
window waits 80 seconds for the response to a write.
- **CRITICAL FINDING — 9a alone CANNOT satisfy the acceptance criterion.** The
  window consists of two parts: **sealing 20.8 s + merging 59.7 s**. 9a only moves
  merging to the background; the remaining sealing window of 20.8 s is
  **2.7 million times** the baseline p99, so the 50x acceptance threshold is still
  not met. The pre-registration's clause "if exceeded, it is not accepted and the
  cause is investigated" applies: **the cause is sealing**. Either 9a's scope must
  widen (moving sealing to a background task as well) or sealing must become
  incremental — this is not changing the pre-registration, it is the new work the
  measurement revealed.

**9b — lazy loading via mmap: NO-GO. `unsafe` STAYS CLOSED.**
Cold start was broken down into components (BENCHMARKS): the work mmap could
remove is only (a) file reading, **196 ms**, plus the vector-copying share of
(b) ≈ an upper bound of **~0.6 s** in total. The threshold: a gain of ≥ 40%
(**1.45 s**) AND ≥ 2 s. **Neither can be met** — even the upper bound is below
half the threshold.
- `deny(unsafe_code)` stays in place crate-wide.
- **Revisit condition** (from #40): if the vector data no longer fits in RAM
  (>70% of physical memory), mmap becomes a necessity rather than an
  optimization.
- **The real opportunity the measurement revealed lies elsewhere:** **63% of cold
  start is rebuilding the derived indexes** (posting + numeric), at 2.28 s. In #34
  we chose not to write them to disk so that metadata would be the single source;
  the measurement now puts a number on the price of that decision. Snapshotting
  them would be a gain roughly 4x larger than mmap's, and it requires no `unsafe`.

**9c — metadata compaction for sealed segments: GO.**
The metadata share is **51.5%** (934 MB) against a 25% threshold — more than
double. Metadata is **larger** than vectors+graph (882 MB): it has stopped being
a "second-class passenger" and has become the item that determines capacity.
- The distribution: the id→metadata map 421 MB, Eq posting lists 353 MB, numeric
  indexes 160 MB.
- Against 1816 MB of computed size, **peak RSS was 3167 MB** (a 1351 MB
  difference: allocator fragmentation, `Vec` capacity slack, and the two sources
  plus the merged segment coexisting during a merge).
- Note: the "sorted array + binary search" transformation 9c planned targets the
  numeric indexes (160 MB); the measurement shows the real bloat is in the
  **id→metadata map and the posting lists**. Repeating the string keys in every
  record (a HashMap<String, MetaValue> per record) is the likely main waste —
  9c's scope should be reviewed accordingly.

### 43. Findings not tied to a threshold but worth recording
- **Read scaling collapses at 1M:** a single thread does 1048 QPS while 8 threads
  total 945 QPS (at 100K it scaled 8.7x). A memory-bandwidth wall. Phase 5's
  contract "readers never block one another" still holds at the *lock* level; the
  limit has moved from software to hardware. This is where int8 quantization (4x
  less memory traffic) would earn its keep at 1M+.
- **The fsync policy does not burden readers** (ratio 0.99–1.01): the first real
  test of the phase 5 contract was passed.
- **Filter latency degrades with scale:** in the clustered×distant s=0.3 cell,
  p50 went from 3.9 ms (100K) to 92.7 ms (1M); recall holds (0.997) but the cost
  of the scan arm is linear in the number of matches.
- **The segment model speeds up building:** a single 1M graph takes 802 s, while
  8 segments take 170 s (4.7x) — smaller graphs escape the super-linearity of
  construction.
- **Replay is not linear:** if the number of replayed records exceeds the sealing
  threshold, an HNSW build is triggered inside recovery. Checkpoint frequency =
  the ceiling on recovery time.

## Phase 9 PRE-REGISTRATION — 2026-08-18 (written BEFORE the phase 8 measurement was run)

### 40. Go/no-go thresholds and acceptance criteria
The rule: thresholds are written before the measurement and CANNOT BE CHANGED
after the result is seen. All of them are **ratios against a baseline** measured
in the same run; the baselines are defined here too, so that the argument about
"which baseline to use" is settled in advance.

#### 9a — moving merging to an independent task: DONE UNCONDITIONALLY
Here the threshold is NOT a go/no-go gate but an **acceptance criterion**. The
reason: a merge means rebuilding ~250K records and it blocks the writer task; the
window extrapolated from the 100K measurements is 10–40 s, so any reasonable
threshold will be exceeded by three orders of magnitude. Rather than spending
budget on a gate whose answer is already known, the measurement documents the
*rationale* for 9a while the threshold tests its *success*.

- **Baseline:** the write p99 measured OUTSIDE the merge window, under the same
  load.
- **Acceptance (after 9a):** if the p99 of writes coinciding with the merge
  window **does not exceed 50x** the baseline p99, 9a is successful. (50x ≈
  100–150 ms: the upper bound of "slow but acceptable" for an HTTP write.) If it
  is exceeded, 9a is **not accepted** and the cause is investigated.
- **The REAL acceptance criterion is not latency but the tombstone race:** a
  sealed segment takes no inserts, but it does take deletes (tombstones).
  Tombstones landing on the source segments while a merge runs must be
  diff-replayed into the merged segment at swap time, under the write lock. If
  that race is wrong, then even with the window closed **data is silently lost**,
  and no latency measurement would ever catch it. A test dedicated to that
  scenario is 9a's primary acceptance condition.

#### 9b — lazy loading via mmap (the gate for permitting unsafe)
- **Baseline:** the 1M cold-start time and the search p50 (ef=50, unfiltered).
- **Go (permission is requested) — BOTH conditions together:**
  1. Cold start must shorten by **≥ 40%** **and** the absolute gain must be
     **≥ 2 s** (a second anchor so the percentage does not become meaningless on
     a small base);
  2. The search p50 regression must **not exceed 10%**.
- **No-go:** if either condition fails, `unsafe` is not enabled and
  `deny(unsafe_code)` stays crate-wide. **Revisit condition:** if the vector data
  no longer fits in RAM (>70% of physical memory), mmap becomes a necessity
  rather than an optimization, and the threshold is redefined at that point.
- **The recall rule (not a threshold):** mmap reads the same bytes by a different
  path; a change in recall is not a matter of thresholds but **an indication of a
  bug**. If a difference appears, no decision is made — **we stop and find the
  error**.

#### 9c — metadata compaction for sealed segments
- **Baseline — the configuration is explicitly f32:** total index memory =
  vectors + graph, **in f32 mode** (the system's default operating mode;
  `QuantizedHnsw` is not integrated into the segment model). Under int8 the vector
  share shrinks 4x, so the same metadata would automatically occupy a much larger
  proportion — the threshold would yield two different numbers, so the baseline is
  fixed.
- **Measurement method:** the threshold is evaluated against the **computed**
  structure sizes (metadata cannot be isolated within RSS); RSS is reported too,
  and the difference (allocator fragmentation + Vec capacity slack) is information
  in its own right.
- **Go:** if metadata memory **exceeds 25%** of total index memory. Rationale:
  above that ratio metadata stops being a second-class passenger beside the
  vectors and becomes the item that determines capacity.
- **No-go:** below 25% it is rejected. **Revisit condition:** if the number of
  numeric fields grows beyond 3, or if we move to a 10M scale (where an absolute
  memory ceiling comes into play), it is measured again.
- If 9c is done, its own acceptance criteria: arm agreement must stay 100%,
  filter recall must not change, and the O(n log n) conversion cost during sealing
  must be measured.

### 41. The phase 8 measurement protocol (so the thresholds are measurable)
- **The merge window:** a few thousand ops are collected both BEFORE and AFTER
  the window; otherwise the baseline p99 is nothing but noise. The merge is
  triggered deliberately by sealing a 9th segment (ceiling 8).
- **Memory:** the computed sizes (vectors, graph and the metadata structures
  separately) are reported together with the process RSS.
- **The 1M crash test:** phase 7's matrix ran on small data; at least one
  truncation scenario is repeated with a 1M snapshot and a full WAL — this is
  where it shows whether replay time continues linearly from 155 ms/100K.
- **Mixed load:** 8 readers + 1 writer × 3 fsync policies. This is the first real
  test of phase 5's "readers never stop" contract: if waiting on fsync burdens
  reader QPS, the contract has weakened in practice.

## Phase 7b/7c — WAL and recovery — 2026-08-18

### 36. The HTTP 200 contract is defined per policy
The write ordering is **write-ahead**: (1) validation (dimension/duplicate — NO
mutation), (2) WAL append + the policy's fsync, (3) apply to memory. In the
reverse order you would get "we returned an error to the client but the record
stayed in memory and the next checkpoint made it permanent". When
`IndexError::Storage` is returned, the mutation has NOT been applied to memory.

A 200 (POST /vectors → 201, DELETE → 204) means:

| policy | 200 = | survives | measured |
|---|---|---|---|
| `none` | memory + WAL append (OS cache) | a process crash; **NOT a power loss** | 281,609 op/s |
| `group:T` (default) | the fsync covering the record completed | a power loss | 31,669 op/s (batch=64) |
| `per_op` | the record's own fsync completed | a power loss | 499 op/s |

**The default is `group:20`** — the measured rationale: under per_op an fsync
takes ~2 ms and throughput slams into 499 op/s; group gives the same durability
promise at 63x the throughput. The price is that the response is delayed by the
batch window. For real group commit the writer task batches commands, performs a
SINGLE commit at the end of the batch, and sends the responses **only after
that**; a "group that does not wait for the fsync" would silently weaken the
contract.

### 37. Recovery: stop at the first inconsistency and TRUNCATE the file there
Replay stops on a partial record / CRC mismatch / implausible length; a phantom
op is never synthesized. The critical detail: the file is truncated with
`set_len` at the end of the intact prefix. Without truncating, the next append
would write on top of the corrupted tail and the file would stay permanently
inconsistent (a second replay would give a different result — there is a test).

During replay `self.wal` is NOT yet attached; that makes the "log what I just
replayed" bug structurally impossible.

### 38. The crash-test method: deterministic truncation, not killing a process
A sequence of operations is applied to a real index, the WAL file is cut at
record boundaries AND mid-record (mid-header, mid-body, one byte short), and the
index
is reopened. This is portable (Windows included), reproducible, and the cut
point is under exact control. The correctness criterion: the recovered state ==
the state of the WAL's intact prefix (nothing missing, nothing extra). With
proptest: a random operation sequence × a random cut point, plus entirely random
bytes → no panic at any point.

### 39. WAL rotation is tied to checkpoints
A checkpoint first seals the buffer (all data moves into segments), then opens a
NEW WAL file, then writes the manifest. By the time the manifest points at the
new WAL, every record of the old one is already in the segments. On an
interruption the old manifest still points at the old WAL — consistent. There is
NO checkpoint marker in the WAL: the information "everything before this point is
in the segments" is the file boundary itself.

## Phase 7a — cold persistence — 2026-08-18

### 32. Segment files are immutable and their names carry the generation
`segment-<gen>-<idx>.gvdb` is written once and NEVER overwritten; later
checkpoints only reference it from the manifest. Three benefits: (1) every
checkpoint writes only the NEW segments — at 100K the first checkpoint took
221 ms and the second (no new segments) 98 ms; this is what determines checkpoint
cost at 1M. (2) It is compatible with Windows file locking: we never write to a
file with an open handle. (3) It is a precondition for phase 9b's mmap — the
immutability of the mapped file is already guaranteed.

### 33. The manifest is the single source of truth, swapped atomically, written LAST
The write order: new segments → metadata snapshot → **manifest** → GC. At every
instant the manifest on disk is consistent with all the files it references;
whichever step is interrupted, the OLD manifest stays valid and the new files are
orphaned (a later GC collects them). GC runs after the manifest — the reverse
order could delete a file that is still referenced.

**No directory fsync on Windows:** file contents are fsynced with `sync_all` and
the rename is atomic (MoveFileEx REPLACE_EXISTING), but the durability of the
directory entry is left to the operating system (Rust std does not open a
directory handle). Consequence: the scenario "the checkpoint is on disk but the
directory entry was lost" is theoretically possible — which is why recovery will
always be completed by a WAL replay (7b).

### 34. Tombstones live in the manifest; derived structures are NOT on disk
- Tombstones cannot be written into the segment file (the immutability rule) and
  cannot be left to the WAL (a checkpoint rotates the WAL). The manifest is
  already atomic and small; merging/compaction clears tombstones regularly.
- Eq posting lists and numeric field indexes are **not written to disk**: they can
  be derived exactly from the metadata. A single source → structurally no risk of
  drift. The price is rebuilding them at startup (100K + 3 fields: total cold
  start 242 ms; phase 8 will measure it at 1M).
- The metadata snapshot is a full write (not incremental): the hot path will be
  carried by the WAL.

### 35. The disk representation is separate from the API representation: `MetaValueRepr`
`MetaValue` is `#[serde(untagged)]` for the HTTP JSON shape — natural bodies like
`{"color":"blue"}` require it. But untagged deserialization needs
`deserialize_any`, and since bincode is not self-describing it does NOT SUPPORT
that (not silently — it is a compile/runtime error). The disk and WAL
representation is therefore a separate, tagged enum. The separation is healthy
anyway: one is an external contract, the other an internal format, and they can
evolve independently. A regression test round-trips every MetaValue variant.

## The Range histogram — 2026-08-18

### 31. Range estimation: an equal-width histogram [lower,upper] + bounded counting
- **64 equal-width buckets**, not quantiles: simple, O(1) updates. The risk of a
  skewed distribution was measured (log-normal: upper/truth up to 49x) but we did
  not move to quantiles, because the estimation error does not leak into the arm
  choice (see below). Quantiles only come up if the post-arm's ef'' scaling costs
  measurable latency on skewed data — in that measurement recall was 0.999+ and
  the p50s were indistinguishable from the uniform distribution.
- **The estimate is an [lower, upper] interval rather than a single number** (the
  user's design): fully contained buckets give the lower bound, including the
  boundary buckets gives the upper. No within-bucket uniformity is ever assumed;
  the uncertainty is carried openly and the planner always uses the conservative
  side (the upper bound for the small arm, the upper bound for ŝ → errors push on
  latency, never on recall).
- **The critical addition — bounded counting**: alongside the histogram there is a
  value-ordered BTreeMap (bit-ordered f64). The small-arm decision is made NOT
  from an estimate but exactly, via `enumerate_up_to(scan_limit)`; the matching
  ids fall out for free for the scan arm. This is what makes the arm agreement in
  the acceptance criterion structurally 100% (measured: 13/13, including the
  skewed and correlated cells).
- **The AND conjunction**: NO independence assumption; the minimum of the upper
  bounds (Fréchet). In the correlated Eq∧Range cell it is inflated 2.25x, but
  conservatively.
- **A 12.5% widening margin**: it amortizes histogram rebuilds under a monotone
  value stream (there is a test).
- Maintenance: insert/remove is O(log distinct); +4% on a 100K build (BENCHMARKS).
- The memory price: the sorted map costs ~24B per id per numeric field —
  expensive next to a histogram-only design, but it buys small-arm exactness plus
  the fallback enumeration.


## The segment ceiling guard — 2026-08-18

### 30. Merging: a minimal ceiling guard, the two smallest, ceiling 8
The segcurve measurement (BENCHMARKS): the curve is close to linear
(~+45µs/segment) and in an equal-recall comparison a full merge gains ~20% — so
the rationale for merging is NOT latency but cutting off unbounded growth (40
segments would be ≈1.8 ms; the curve does not saturate). The policy:
- The **two smallest** segments are merged (not the oldest): rebuild cost scales
  with n, so this is the cheapest merge and sizes stay balanced. An HNSW merge is
  not a true merge but a rebuild (there is no cheap way to combine graphs); write
  amplification is worse than in an LSM → the policy stays conservative.
- The mechanism is the "two inputs, one output" variant of sealing: a lock-free
  rebuild, then an atomic swap under a single write lock (retain+push under the
  same lock — a reader sees either the old pair or the merged segment, never both
  and never neither).
- A merge is a natural compaction: tombstoned records are not carried over.
- The single-writer contract is preserved: a merge keeps the writer busy for the
  duration of the rebuild (+3.9 s in total at 100K/ceiling-8) while readers never
  stop. Peak memory: steady state plus the two source segments (until the swap;
  +2×9 MB for 10K segments).
- Ceiling 8: in segcurve, 8 segments ≈ 385 µs — an acceptable baseline; the
  ceiling is only checked after sealing, never a merge per sealing.


## The filter planner — 2026-08-18

### 28. A measurement finding: the fragility is in latency, not recall (a revision of #26)
The selectivity sweep (BENCHMARKS, the filter section) showed the opposite of the
hypothesis: in-traversal filtering PRESERVES recall (the worst cell is 0.952)
because it keeps expanding until the admitted set is full — the price is that
with clustered matches and a distant query the traversal spreads across the whole
graph (the admit/visit ratio collapses from 0.19 to 0.01, p50 25µs→1.3ms). The
old binary fallback (found < k) never caught that pathology (it fired 0 times in
the sweep). A scaled-ef arm was tested and REJECTED: recall was already high, it
only added latency.

### 29. A three-arm planner: scan / post-filter (over-fetch) / in-traversal
- **O(1) cardinality estimation**: for Eq, posting lists of (field, value) → the
  set of live ids (maintained on insert/delete; consistency is tested). An O(n)
  metadata count was rejected for the planner: 14.4 ms at 100K — hundreds of times
  the search being planned. Range predicates do not take part in the estimate.
- **Arm 1 — scan**: est ≤ max(16k, 0.05n) → the graph is never opened, exact
  top-k over the smallest posting list. The cost is bounded by est and independent
  of query position (12µs–1ms at 100K).
- **Arm 2 — post-filter (over-fetch)**: if est is larger, an UNFILTERED graph
  search with `ef'' = clamp(5k/ŝ, ef, 8ef)`, with the filter applied to the
  results. The critical insight (the 100K measurement): in-traversal filtering
  spreads across the entire graph with clustered matches and a distant query
  (35 ms), and a silent recall decline was setting in with scale (0.948, without
  the fallback ever firing — the visited/admitted collapse
  was the only signal). Unfiltered traversal is STRUCTURALLY immune to that
  pathology. ŝ is the Eq-minimum upper bound; if fewer than 2k results remain
  (the window missed, or the estimate was inflated — AND-conjunction correlation)
  it falls back to an exact scan. β=5: with β=3 the number of results sufficed
  while quality slipped away (the 0.979 cell), and the 2k threshold did not catch
  that.
- **Arm 3 — in-traversal**: only for filters without Eq (Range-only), with the old
  found<k safety net. The only option when there is no estimate.
- TRIED AND REJECTED: scaled ef (together with in-traversal filtering — it was
  already preserving recall, so this only added latency); a visit budget + scan
  fallback (it worked at 10K, but at 100K the wrong cutoffs cost 30K-element scans
  — the budget API remains as measurement instrumentation).
- The calibrated result (10K): recall 1.000 in ALL 21 cells; the worst cell is
  1.03 ms (the scan baseline), where the old worst was 1.3 ms with 0.952 recall.


## Metadata filtering — 2026-08-18

### 26. In-traversal filtering + a brute-force fallback
There were three options: post-filter (search, then discard — fewer than k
results at low selectivity), pre-filter (find the matches first and search among
them — graph connectivity breaks), and in-traversal (a non-matching node is
traversed as a bridge but never enters the results — a generalization of the
tombstone mechanism). The third was chosen; an optional slot predicate was added
to `search_layer`. The correctness guarantee: if the graph search finds fewer
than k results it falls back to a filtered linear scan (slow but complete under a
highly selective filter — a single-match scenario in the tests verifies this).

### 27. Metadata lives at the id level, separate from the segments
`SegmentedIndex.metadata: HashMap<VectorId, Metadata>` — the segments stay
immutable while metadata flows by id through deletion and re-insertion (a
deletion drops the metadata, so old metadata cannot leak into a new record). The
filter model is deliberately narrow: the AND conjunction of Eq + Range;
OR/negation would be extended into a tree if the need arose.


## SIMD — 2026-08-18

### 25. Explicit SIMD: `wide::f32x8`, without unsafe
`std::simd` requires nightly and intrinsics require unsafe; `wide` avoids both
with a safe API (`deny(unsafe_code)` is preserved). The change in float summation
order is accepted deliberately: distances are only compared, and a ~1 ulp
difference does not affect the outcome. `target-cpu=native` is set in
.cargo/config.toml — local performance over binary portability (this is a
learning project).


## Phase 6 — 2026-08-18

### 22. The quantization architecture: build in f32, freeze and quantize, search with ADC
The graph is built at f32 precision (neighbour selection benefits from full
precision); `QuantizedHnsw::from_hnsw` copies the graph and converts the vectors
into u8 codes; once the f32 source is dropped, only the codes remain in memory.
Search is asymmetric (ADC): the query stays f32 and the codes are dequantized on
the fly — quantizing both sides would expose the result to the error twice. On a
frozen index insert/delete return `Unsupported`: in the segment model (phase 5)
writes go to the buffer anyway, and a quantized index plays the role of "the
compressed form of a sealed segment".

### 23. NO rerank (pure quantization)
The options: (a) pure SQ — low memory, reasonable recall; (b) SQ + reading f32
from disk to re-rank the top-k — high recall, an IO dependency. **(a) was
chosen.** The measurement: on SIFT 100K the loss is 0.005–0.011, not even half of
the 0.02 target — the recall rerank would buy is practically nil, while it would
add a disk IO path, a file-lifecycle dependency and p99 uncertainty. Rerank will
be reconsidered if the recall budget genuinely tightens (e.g. a move to PQ).

### 24. Per-dimension min/max calibration
Per dimension rather than global: on data like SIFT, where the dynamic range
differs across dimensions, this shrinks the error margin dimension by dimension.
A constant dimension (max==min) yields scale=0; the code is 0 and decoding
returns min — no NaN is produced (there is a test).


## Phase 5 — 2026-08-18

### 18. Concurrency: the segment model (approved strategy 2)
Immutable segments plus an append-only buffer were chosen over COW/RCU (with the
user's approval). The rationale: no O(n) copy per write; compaction becomes
"sealing" and can move to the background; it is the architecture real vector
databases use. Sharded locking was rejected because the unit of locking (a shard)
does not coincide with the unit of access (a traversal path): every step jumps to
a random shard, and an insert needs many locks at once for bidirectional linking
plus multi-node pruning → either deadlock or one giant lock.

### 19. Lock discipline: expensive work is never done under a lock
A reader clones the segment list as a `Vec<Arc<Segment>>` (a read lock held for
microseconds) and performs the HNSW search lock-free. Sealing (an HNSW build,
taking seconds) runs holding no lock at all; the publication order is "append the
segment first, drain the buffer after" — any duplicates visible in the window
between are absorbed by the id-based deduplication in search (a duplicate was
preferred over data loss).

### 20. Tombstones are segment-LOCAL
A global deleted-set would resurrect the old copy of an id that is deleted and
re-inserted (removing it from the set would make the old vector in the segment
visible again). With a segment-local set the old copy stays permanently shadowed
in its own segment.

### 21. The single-writer contract
Mutations work through `&self` under locks (`insert_shared`), but because the
duplicate check is check-then-act it can race with multiple writers; the contract
is "many readers + one writer" (consistent with the phase's acceptance criteria).
Multiple writers could later be added by partitioning the id space or with a write
queue.


## Phase 4 — 2026-08-18

### 15. Deletion: a tombstone that bridges during traversal and is excluded from results
A deleted node is not removed from the graph (repairing edges is expensive and
risks connectivity); it is marked with a `deleted` flag. `search_layer` keeps
TRAVERSING tombstones (their neighbours are explored, so they go on serving as
bridges) but never admits them into the result set. During construction new nodes
may link to tombstones — compaction clears them wholesale.

### 16. If the entry point is deleted: the highest-level live node becomes the new entry
A tombstone could have worked as a waypoint, but having every search start from a
dead node adds fragility; `pick_new_entry` runs at deletion time. If every element
is deleted, entry becomes None and the next insert builds from scratch.

### 17. Compaction: a full rebuild triggered by a threshold
When the tombstone ratio exceeds `tombstone_threshold` (default 0.3), a delete
triggers an automatic compaction: the live records are re-inserted into a fresh
index. A full rebuild rather than in-place slot reclamation: simple, correctness
guaranteed, and HNSW construction is fast anyway (10K → ~2 s). The price: a
compaction causes a momentary pause — phase 5's concurrency model can move it to
the background.


## Phase 3 — 2026-08-18

### 11. The file format: magic + version + bincode meta + a raw f32 section + CRC32
The vector data is kept OUTSIDE meta, in a raw section aligned to 4 bytes — so it
can be accessed without copying via mmap. The CRC32 covers the whole file, and
the checksum is verified BEFORE meta is parsed (never showing a corrupt byte to
the deserializer shrinks the fuzz surface). Writing goes through a temporary file
plus an atomic rename: a half-write can never corrupt the real file.

### 12. memmap2 lazy loading is ON HOLD: it needs permission for unsafe
`memmap2::Mmap::map` is an `unsafe fn` (UB if the file changes while the mapping
is alive) and
the crate is compiled with `#![deny(unsafe_code)]`. Per the rule, the user was
asked before lifting it; until approval, `load(path, lazy)` performs a safe full
read on both paths. The `VectorStorage::Mmap` infrastructure is ready (bytemuck
cast, copy-on-write insert) and activation is a one-liner once permitted.

### 13. RNG state is not written to disk
After loading, the level RNG is re-derived from `seed ^ n`. Inserts into a loaded
index are deterministic, but the graph may not be bit-identical to one built
without interruption — search correctness is unaffected, and this was accepted.

### 14. `load_from_bytes` is a separate surface
The fuzz target, the tests and the file path share the same parsing code;
`rebuild` bounds-checks every slot/entry reference — so even a corrupt file whose
crc happens to match (deliberately constructed) yields Err rather than a panic.


## Phase 2 — 2026-08-18

### 7. HNSW neighbour selection: the Algorithm 4 heuristic + keepPrunedConnections
The paper's heuristic instead of a naive top-M: a candidate is discarded if it is
closer to an already-selected neighbour than it is to the query. This prunes
redundant intra-cluster edges while preserving inter-cluster bridges; that is
where recall's robustness to data clustering comes from. Topping up to M with the
discarded ones (keepPrunedConnections) is enabled — so no node is left with a low
degree.

### 8. After pruning, the graph is directed
When `shrink_links` prunes a node's list, the opposite edge is not deleted (the
same behaviour as hnswlib). Enforcing bidirectionality would require scanning the
opposite lists on every prune and brings no practical benefit; the tests verify
only the degree limit and neighbour validity.

### 9. Level assignment and default parameters
`level = floor(-ln(U) * mL)`, `mL = 1/ln(M)` (the optimum from §4.1 of the
paper), `M_max0 = 2M`. Defaults: M=16, ef_c=200. The sweet spot from the sweep:
M=16 with ef_search 25–50 (the phase 2 table in BENCHMARKS.md).

### 10. A `Vec<bool>` for `visited` in `search_layer`
A flag per slot rather than a HashSet: at 100K nodes that is a single 100KB
allocation per query with no per-branch hashing cost. Since it stays query-local
through the concurrency phase, it creates no sharing problem.


## Phase 0 — 2026-08-18

### 1. The `VectorIndex::insert` signature: `&mut self`
**Decision:** mutations in the trait take `&mut self`; no interior mutability.

**Rationale:** index algorithms (HNSW insert in particular) are written in their
simplest, most testable form under a single-writer assumption. The concurrency of
phase 5 will be added not by burying `RwLock`/atomics inside the trait but by
wrapping a layer **on top** of the index (COW/arc-swap or an immutable segment
model — to be compared and chosen in that phase). Had we chosen `&self` +
interior mutability, every implementation would have to reason about lock
granularity, and even the single-threaded brute-force index would carry
unnecessary synchronization. `&mut self` plus an outer layer separates "the
algorithm" from "the concurrency policy"; changing strategy in phase 5 does not
break the trait.

### 2. The cosine normalization policy: normalize at insert/query time, use a dot product in search
**Decision:** an index built with `Metric::Cosine` normalizes a vector once at
insert time and the query once at the start of a search; the hot distance loop
runs `-dot`.

**Rationale:** an HNSW search computes thousands of distances; taking two norms
(including a sqrt) in each would roughly triple the cost. Normalization is paid
once per vector and the resulting ordering is identical. The price: the original
(un-normalized) vector cannot be read back from the index — acceptable for a
search engine; if needed, the originals could be stored separately in phase 3's
persistence layer. The zero-vector edge case: it is left un-normalized (no NaN is
produced) and its similarity to everything is taken as 0.

### 3. The distance contract: "smaller = closer", squared L2, similarities negated
**Rationale:** a single directional ordering contract lets the top-k/heap/recall
code be written once, independent of the metric. Since `sqrt` is monotone, it is
skipped for L2.

### 4. Graph representation: index-based (`Vec<Vec<usize>>`), no Rc/RefCell
**Rationale:** (to be implemented in phase 2, binding from now on.) Flat vectors
with slot indexes remove all borrow-checker friction, are cache-friendly, and are
trivial to serialize in phase 3.

### 5. The measurement infrastructure is independent of the indexes
`eval::exact_top_k` is a plain linear scan outside the trait and remains the
ground-truth generator for the whole project. Even phase 1's brute-force index is
tested against it; if the reference and the thing under test were the same code,
the test would be meaningless.

### 6. Reproducibility: `StdRng::seed_from_u64(42)`
All randomness (data generation and, later, HNSW level assignment) comes from a
`StdRng` with a fixed default seed; benchmarks are deterministic.
