# Causalitybot Audit — 12 November 2026

## Critical

**Bug 1 — CausalAttractor timing.** `generate_signal` called
`attractor.compute(event)` on the event that had just arrived and was
therefore the last element in `poset.events`. The `future_events` filter
produced an empty list, `compute` returned `(None, 0.0, {})`, and
`generate_signal` returned `None` on every call. `n_signals` was zero on
every run in every mode (backtest, live, synthetic). The bug is invisible
to unit tests that check arithmetic and visible only when the streaming
path is exercised end to end.

**Fix.** Track active horizons in `self.active_horizons`. Append on
detection, evaluate on subsequent ticks, expire after lookahead.

**Bug 2 — BottleneckScanner misses vacuums.** `_volume_profile` only
inserted price levels with `volume > 0`, so empty gaps were never present
in the levels iterated by `scan`. The score `(1 + V_left) * (1 + V_right) /
(V_centre + 1e-9)` is designed to diverge as `V_centre → 0`, but it could
not be applied to the levels where it matters most.

**Fix.** Build a contiguous grid from `min(profile)` to `max(profile)` at
`tick` resolution. Empty levels now score at full strength.

## High

**Bug 3 — `_predecessors` too permissive.** `depth_overlap` triggered on
almost every pair within `delta` because top-of-book depth changes on
nearly every tick. The poset became a near-complete DAG, and the causal
interval collapsed to a volume-weighted average of future prices.

**Fix.** Require price overlap for trade events and both price overlap and
depth overlap for limit and cancel events.

**Bug 4 — `_evict_oldest` memory leak.** Evicted IDs remained in
surviving neighbours' adjacency sets, so BFS visited dangling IDs on every
eviction and the sets grew unbounded across churn.

**Fix.** Discard the evicted ID from every surviving neighbour's set.

**Bug 5 — autocorrelated z-score baseline.** Consecutive `local_depth`
values shared most of their contributing events, so the effective sample
size was far below the nominal 100 and the variance estimate was inflated.

**Fix.** Sample the baseline at non-overlapping bucket boundaries.

**Bug 6 — `ddof=0`.** Sample standard deviation used the population
estimator.

**Fix.** `ddof=1`.

## Medium

**Bug 7 — `curvature_window` units.** Documented as event count, used as
seconds. Fix: documented as seconds in the docstring.

**Bug 8 — `break` on out-of-order events.** `_predecessors` used `break`
on `dt > delta`, which truncated the scan on late-arriving events.
Fix: `continue`.

**Bug 9 — mutable hash on `MarketEvent`.** `__hash__` returned `self.id`,
which was reassigned by the poset. Fix: hash on immutable fields
(`timestamp`, `price`, `volume`, `event_type`).

**Bug 10 — two-element level unpacking.** `ingest_orderbook` assumed
`[price, volume]` per level. Fix: index positions explicitly.

**Bug 11 — Sharpe annualisation.** Multiplied by `sqrt(252)` despite
per-event returns. Fix: annualise by measured event frequency.

**Bug 12 — synthetic random walk.** Driftless random walk cannot validate
an attractor. Fix: Ornstein–Uhlenbeck around a hidden level.

## Low

**Bug 13 — unused `mid` in `BottleneckScanner.scan`.** Fix: filter distant
levels by `mid * mid_filter_fraction`.

**Bug 14 — comment contradicts code in `generate_signal`.** Fix: comment
now matches the actual rule.

**Bug 15 — live timestamp units.** Assumed milliseconds. Fix: normalise
by magnitude.

**Bug 16 — symmetric curvature windows.** `abs(dt) <= window` peeked into
the future on historical events. Fix: trailing only.

**Bug 17 — `interval` early return.** The early return became incorrect
when terminal events had lost their direct neighbours to eviction.
Fix: remove the early return.