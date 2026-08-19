# What was not done, unmet thresholds, and known limits

This file is deliberately the "bad news" file. Knowing what a system does not
do, and which thresholds it failed to meet, matters as much as knowing what it
does. The long form of each record is in `DECISIONS.md`.

## Unmet acceptance thresholds

Under the pre-registration rule these thresholds were written **before** the
measurement that would evaluate them, and were never changed afterwards. A
threshold that was not met **stays** "not met".

| threshold | result | defect record alongside it |
|---|---|---|
| **#40 criterion 1** — p99 of writes coinciding with the sealing/merge window must not exceed 50x the baseline p99 | **not met** (609x / 943x; 426x after `with_capacity`) | #60: the threshold was defined in a system where backpressure **did not exist**. Back then a long write was always the symptom of a defect; after backpressure was added, a long write can also be proof that the system is working *correctly*. The metric now sums two different causes into one number. |
| **#49** — under 2 min of load, segment+sealing count must stabilize (≤+20%, peak ≤12) | **not met** (+90%, peak 35) | #58: the metric adds two numbers from **different regimes** — segment count is already bounded by the merge ceiling, whereas the queue was unbounded at the time. |
| **#59 secondary** — segment count must stay within merge ceiling + 4 | **not met** (peak 16 over 10 min) | #62: this tests the merge ceiling mechanism, not 9a-2; tracked as a separate item. |
| **#61 primary** — longest write excluding backpressure / baseline p99 | **426x** under group:20, **12x** with sync off | The entire residual spike is fsync (isolated with three independent pieces of evidence). The pre-registration never stated which policy decides acceptance — that gap is recorded. |

Thresholds that *were* met are in `BENCHMARKS.md`: #59 primary (queue held at
3 for a full 10 minutes), 1M recall ≥0.99, arm agreement 100%, and the 9c
memory reduction.

## Tried and rejected

- **Scaled-ef arm** — considered as a third planner arm, measured, rejected:
  recall was not degrading in the first place, so the arm had no problem to
  solve.
- **Integrating int8 quantization into the segment model (8a)** — the
  pre-registered threshold passed, but **the assumption underneath it was
  falsified**: int8 turned out slower than f32 under multiple threads. A
  threshold and a decision are different things; a threshold can pass while
  the assumption it rested on collapses.
- **In-traversal filtering as the default path** — a 100K measurement showed
  the traversal spreading across the entire graph (up to 35 ms) for clustered
  matches with a distant query, and a silent recall decline setting in with
  scale. It was replaced by unfiltered traversal + over-fetch, which is
  structurally immune to that pathology.

## Measured, gain judged insufficient, deferred

- **Aggressive/tiered merge** — the gain is ~20% in an equal-recall
  comparison. That is why the merge ceiling exists to bound growth rather than
  to buy latency.
- **Quantile histogram** — the current equal-width histogram is off by up to
  49x on a skewed distribution, but it **changes no arm decision** (the
  small-arm decision comes from an exact count). Large error, no effect.
- **mmap / enabling `unsafe` (9b)** — the ceiling on cold-start improvement
  from mmap is 0.6 s, below the threshold. `#![deny(unsafe_code)]` stays.
  **Revisit condition:** if vector data no longer fits in RAM.
- **Snapshotting derived indexes (9d)** — expected gain ~1.5 s of cold start;
  invisible in a single-user system. Deferred.
- **Compacting the numeric indexes (9c)** — 197 MB, about 16% of the metadata
  total. The reason is **not the memory share but the risk/reward**: that
  structure feeds the exact-counting arm of the filter planner, i.e. it sits
  underneath the most delicate correctness mechanism in the project. Reaching
  for the riskiest component on the way to closing out is the wrong timing.
- **Dropping Eq postings for numeric fields** — the next step written after
  9c. Every distinct value of a numeric field produces **both** a posting key
  **and** a `BTreeMap` entry; Range is already served from the numeric index.
  This would be the removal of a duplication, not an optimization. Deferred.

## Known limits (it works, but this is how it works)

- **Segment accumulation — the one dimension of the system that still grows
  without bound.** Sealing produces a segment about every 25 s; merging
  removes one about every 54 s. Under sustained heavy write load the segment
  count grows (0 → 16 over ten minutes). This is a side effect of 9a's own
  success: sealing got faster, merging did not. The fix is a separate design
  task (parallel merge, or merging three segments at once). **Revisit
  condition:** if sustained write load becomes part of the intended usage. In
  the target scenarios (RAG, internal tooling) there is no sustained 5K op/s
  write stream — a real problem, but currently a theoretical one.
- **`metadata_memory_bytes()` systematically under-reports** (measured 0.77x,
  in the same direction for all three items). The same function is used by
  `/stats`, so anyone doing capacity planning sees about 77% of actual usage.
- **Sorted posting lists are sensitive to id ordering.** Insertion in
  ascending order is O(1); in random order it costs an O(n) shift — an 8.8x
  difference for a single 200K list. Measured and accepted; a single 1M-entry
  list would take roughly 30 s.
- **No OR / negation in filters.** Only the AND conjunction of predicates. It
  would be extended to a tree structure if the need arose.
- **Deletion is by tombstone; space reclamation depends on merging.** Under a
  delete-heavy workload, disk and memory stay inflated until a merge is
  triggered.
- **Single node, single writer.** Write throughput is capped by the single
  writer task; scaling is vertical, not horizontal.
