# Handoff note — project CLOSED (v0.1.0, 2026-08-19)

This file carries the **context** that must not be lost when a session ends.
Rationale lives in `DECISIONS.md`, measurements in `BENCHMARKS.md`, what was
not done in `NOT-DONE.md`, and measurement lessons in `METHODOLOGY.md`.

## Status

Phases 0–10 are complete and **v0.1.0 is tagged**. The project is in a closed
state: no new features are being developed. If work resumes, it should be
picked from the backlog below.

| phase | status |
|---|---|
| 0–7 | done (HNSW, persistence, WAL, filters, planner, merge ceiling) |
| 8 + 8a | done — phase 8's wrong finding corrected in #44; int8 rejected (#45) |
| 9a-1, 9a-2 | done — window 80.5 s → 0–2 µs; single worker + backpressure (#53) |
| 9b | NO-GO — `deny(unsafe_code)` was not lifted |
| 9c | done — metadata 934 → 618 MB (#64, #65) |
| 9d | deferred |
| 10 | done — README, NOT-DONE, METHODOLOGY, CHANGELOG, v0.1.0 |

## Backlog (with revisit conditions)

| item | revisit condition |
|---|---|
| **Segment accumulation** (#62) — the one unbounded dimension left; sealing produces one every 25 s, merging removes one every 54 s | if sustained write load becomes part of the intended usage |
| **Drop Eq postings for numeric fields** (#68) — removal of a duplication; the two largest remaining metadata items share one root cause | if metadata memory becomes constraining again |
| **Compact the numeric indexes** (#67) — 197 MB | if memory becomes constraining **and** the regression tests for the exact-counting arm can be strengthened in the same round |
| **9d — snapshotting derived indexes** | if cold-start time becomes visible to users (multi-user / frequent restarts) |
| **mmap / unsafe (9b)** (#40) | if vector data no longer fits in RAM |
| **Fix `metadata_memory_bytes()`** (#66) | if anyone plans capacity from the `/stats` number (it currently shows ~77% of actual) |
| **A threshold for #61's secondary item** — backpressure stall distribution | data comes from the `accumulation` mode; `mergewindow` does not produce it |

## Contracts (no regressions allowed)

Single writer (mpsc → one writer task) · readers never stop during
sealing/merging · HTTP 200 = the durability the policy promises (#36, default
group:20) · seed=42 · 1M recall at ef=100 ≥ 0.99 · arm agreement 100% ·
`search_shared` shortcut when the filter is empty or equivalent-to-empty ·
`#![deny(unsafe_code)]` · clippy clean + rustfmt + all tests green (currently
**118 unit + 6 crash**).

## The pre-registration rule

A threshold is written **before** the measurement that will evaluate it, and is
not changed afterwards. A threshold that is not met stays "not met"; if
warranted, a defect record is written beside it. When writing a threshold, also
ask: **"Which other criterion could this one conflict with?"** (#63 — in 9a-2
the mechanism that passed criterion 2 made criterion 1 impossible to pass.)

## Pitfalls learned in this project

The long form is in `METHODOLOGY.md`. Short list:

1. **Measure in an isolated process** — a measurement appended to the end of a
   long run does not measure what you think (#44). Fresh process, warmup,
   median of 3, confirmation across two runs.
2. **Never read measurement output through `grep`** — panics are hidden and the
   pipe returns `exit 0`.
3. **Concurrency tests must assert the race was triggered**
   (`during_merge > 0`, `stalls > 0`, `max_queue > 0`).
4. **`data/fullscale` is persistent** — measurement modes must guard against id
   collisions, and the directory must be rebuilt before any comparison (runs
   accumulate: 1M → 1.64M).
5. **A threshold and a decision are different things** — a threshold can pass
   while the assumption underneath it collapses (#45).
6. **The official SIFT ground truth is valid only for the full 1M base.**
7. **bincode cannot deserialize untagged serde** (#35).
8. **Rust locking pitfalls** (#54): never take two locks in one expression
   (temporaries live to the end of the statement); a `while let` scrutinee
   lives for the whole **body** (use `let ... else`). Lock order:
   **segments → sealing → buffer**. The symptom of a deadlock is a hang — CI
   has timeouts for that reason (#55).

## Re-runnable measurements

```
cargo run --release --bin report -- sweep 10000 128         # quick start
cargo run --release --bin report -- fullscale 1000000 99    # 1M end to end (~10 min)
cargo run --release --bin report -- memverify 1000000 99    # metadata memory (real RSS)
cargo run --release --bin report -- accumulation 1000000 99 # accumulation + backpressure (10 min)
cargo run --release --bin report -- mergewindow 1000000 99  # write latency window
cargo run --release --bin report -- postingcost 200000 99   # posting id-order sensitivity
cargo run --release --bin report -- rangefilter 100000 99   # arm agreement + Range estimation
```
