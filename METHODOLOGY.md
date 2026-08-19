# Measurement methodology — lessons from this project

The most transferable output of this project is not the code but the
measurement discipline. Every item below came out of **a mistake actually made
here**; the examples are not hypothetical, they are recorded in
`BENCHMARKS.md` and `DECISIONS.md`.

## 1. Measurement infrastructure comes before the code

Before writing an optimization, the measurement that could see it must already
exist. Otherwise you cannot tell "I improved it" apart from "I believe I
improved it". In this project the `report` binary grew with every phase, and
every acceptance decision came out of it.

## 2. Acceptance thresholds are written before the result is seen, and never changed

The pre-registration rule: a threshold goes into `DECISIONS.md` before the
measurement that will evaluate it is run. If it is not met, the result
**stays "not met"** — the threshold is not reinterpreted.

Without this rule, 9a-2's latency threshold ("50x") could easily have become
"well, the actual sealing window dropped to microseconds, so call it a pass"
once the numbers were in. It did not: the result stayed unmet and a **defect
record** was written next to it. A defect record does not change the
threshold; it documents the gap between what the threshold *meant* to measure
and what it actually measured.

Decisions taken after seeing a result (for instance, extending a measurement
from 2 minutes to 10) are taken **with that fact written down explicitly**.
Let the reader apply their own discount.

## 3. Thresholds are measured at scale

A design that passes at small scale can collapse at large scale — while the
tests stay green.

What happened: the first backpressure design was fine in unit tests, because
at small scale merging turns over quickly and equilibrium is reached. At 1M,
where sealing takes ~20 s, that equilibrium broke and **the writer stopped
completely for 110 seconds** (0 op/s). The tests were green; the system was
not working.

## 4. Control the measurement environment itself

A measurement bolted onto the end of a long-running process does not measure
what you think it measures.

What happened: the finding "reads don't scale at 1M" was **wrong**. It had
been taken in a process that had been running for five minutes with RSS at
3.1 GB, as the last section of a long run. In a clean process, f32 reads scale
5.4–6.1x. The correction is `DECISIONS.md #44`.

The protocol: **fresh process, warmup, median of 3 repeats, confirmation
across two separate runs.**

## 5. Numerator and denominator are measured under the same conditions

If you define a ratio threshold, write down the conditions the baseline was
measured under, right next to the threshold.

What happened: the baseline p99 in #40 (7.8 µs) was measured **without fsync**.
Evaluating that same threshold in a run that includes fsync would manufacture
a failure no matter how good the improvement was. The measurement conditions
were kept identical across every repeat, and those conditions were recorded
beside the threshold.

The same principle applies to the dataset: a persistent measurement directory
grows as runs accumulate (1M → 1.64M), so "the same measurement" is no longer
the same measurement. The directory was rebuilt from scratch before every
comparison.

## 6. One criterion can conflict with another

Criteria are written independently, but the system is a single whole.

What happened: the mechanism that made 9a-2 pass criterion 2 (backpressure —
stalling the writer to bound the queue) made criterion 1 (no write may take
longer than 50x the baseline p99) **impossible to pass**.

A permanent clause was therefore added to the pre-registration rule: when
writing a threshold, also ask **"which other criterion could this one conflict
with?"**

## 7. A race test must also assert that the race happened

If a concurrency test only asserts "nothing went wrong", it stays green when
the race it targets never occurs — that is, it silently measures nothing.

The tests here separately assert that the race did occur: `during_merge > 0`,
`seal_in_flight() > 0`, `saw_sealing > 0`, `max_queue > 0`, `stalls > 0`. In
one test the writer finished before the readers started, `iters` came out 0,
and the test **broke** — correctly, because what it measured no longer
happened.

## 8. Think about the direction of the bias in advance

If a measurement method has a bias, decide before running it which way it
skews: which result would be safe, and which would be suspect?

What happened: validating the metadata memory estimate meant dropping
structures and measuring the RSS delta. Freed memory may not return to the OS
immediately → the real drop would look **smaller** than it is → that bias makes
"the estimate is inflated" a safe conclusion and "the estimate is accurate" a
suspect one. The result landed on the safe side (the estimate was
*under*-reporting), so the decision held.

## 9. Show the raw number and the normalization together

Presenting a normalized number as if it were raw will confuse whoever measures
again later. The form "measured: X at 1.5M, 1M equivalent: Y" is both
verifiable and comparable.

## 10. Never read measurement output through a filter

A `grep`-ed output hides panics, and the pipe returns `exit 0`. In this
project a measurement mode was crashing with `DuplicateId` and it went
unnoticed for two runs. Read the output unfiltered; check the exit code
separately.
