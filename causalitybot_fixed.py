"""
causalitybot.py — CausalityBot: A Causal Set-Theoretic Trading Strategy

Corrected 12 November 2026. All fixes documented in AUDIT.md.

Interface (unchanged):
    bot = CausalSetMomentumBot(symbol, delta, curvature_window, horizon_sigma,
                               bottleneck_threshold, risk_per_trade,
                               max_position, tick_size)
    event  = bot.ingest_orderbook(orderbook_dict, timestamp) -> MarketEvent | None
    signal = bot.generate_signal(event) -> dict | None
    signal = bot.on_event(event) -> dict | None

Signal keys: direction, target, strength, curvature, mid, bottlenecks,
             horizon_timestamp, observation_timestamp
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────
# Event types
# ─────────────────────────────────────────────────────────────────────


class EventType(Enum):
    TRADE_BUY = "TRADE_BUY"
    TRADE_SELL = "TRADE_SELL"
    LIMIT_BID = "LIMIT_BID"
    LIMIT_ASK = "LIMIT_ASK"
    CANCEL = "CANCEL"
    UNKNOWN = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────
# Market event
#
# Fix 4.7: eq=False so dataclass does not generate __eq__ and __hash__;
# we define both manually, hashing on immutable fields and excluding the
# mutable `id` that CausalPoset assigns.
# ─────────────────────────────────────────────────────────────────────


@dataclass(eq=False)
class MarketEvent:
    timestamp: float
    price: float
    volume: float
    event_type: EventType = EventType.UNKNOWN
    spread: float = 0.0
    bid_depth: float = 0.0
    ask_depth: float = 0.0
    id: int = 0

    def __hash__(self) -> int:
        return hash((
            self.timestamp,
            self.price,
            self.volume,
            self.event_type.value,
        ))

    def __eq__(self, other) -> bool:
        if not isinstance(other, MarketEvent):
            return NotImplemented
        return (
            self.timestamp == other.timestamp
            and self.price == other.price
            and self.volume == other.volume
            and self.event_type == other.event_type
        )


# ─────────────────────────────────────────────────────────────────────
# Causal poset
# ─────────────────────────────────────────────────────────────────────


class CausalPoset:
    """
    A causal set stored as a directed acyclic graph.

    Every event is a vertex. An edge x -> y is added when x is a causal
    predecessor of y according to _predecessors. The poset is bounded by
    max_events; the oldest events are evicted first.
    """

    def __init__(self, max_events: int = 2000, delta: float = 2.0):
        self.events: list[MarketEvent] = []
        self._next_id: int = 1
        self._edges: dict[int, set[int]] = {}
        self._reverse_edges: dict[int, set[int]] = {}
        self.max_events = max_events
        self.delta = delta

    def contains(self, event_id: int) -> bool:
        return event_id in self._edges

    def add_event(self, event: MarketEvent) -> None:
        event.id = self._next_id
        self._next_id += 1
        predecessors = self._predecessors(event)
        self._edges[event.id] = set()
        self._reverse_edges[event.id] = set()
        for predecessor in predecessors:
            if predecessor.id not in self._edges:
                # The predecessor was evicted between selection and insertion.
                continue
            self._edges[predecessor.id].add(event.id)
            self._reverse_edges[event.id].add(predecessor.id)
        self.events.append(event)
        while len(self.events) > self.max_events:
            self._evict_oldest()

    def _predecessors(self, event: MarketEvent) -> list[MarketEvent]:
        """
        Return the causal predecessors of `event` within the delta window.

        Fix 4.1: a trade event requires only price overlap (because the trade
        itself moves the price); a limit or cancel event requires BOTH price
        overlap AND depth overlap, otherwise the poset becomes a near-complete
        DAG in any market where top-of-book depth changes on every tick.

        Fix 4.6: `continue` rather than `break` so that late-arriving events
        with an older timestamp do not truncate the scan of prior events.
        """
        predecessors: list[MarketEvent] = []
        for candidate in reversed(self.events):
            dt = event.timestamp - candidate.timestamp
            if dt > self.delta:
                continue
            if dt < 0:
                # Candidate is in the future of `event`; skip. This can occur
                # only if events arrive out of order, but the defensive check
                # keeps the poset temporally consistent.
                continue

            price_overlap = (
                abs(candidate.price - event.price) <= max(event.spread, 1e-9)
            )
            if candidate.event_type in (EventType.TRADE_BUY, EventType.TRADE_SELL):
                if price_overlap:
                    predecessors.append(candidate)
            else:
                depth_overlap = (
                    abs(candidate.bid_depth - event.bid_depth) > 1e-9
                    or abs(candidate.ask_depth - event.ask_depth) > 1e-9
                )
                if price_overlap and depth_overlap:
                    predecessors.append(candidate)
        return predecessors

    def _evict_oldest(self) -> None:
        """
        Remove the oldest event and clean all references to it.

        Fix 4.2: discard the evicted ID from every surviving neighbour's
        adjacency set, so BFS does not visit dangling IDs and memory does
        not grow across evictions.
        """
        if not self.events:
            return
        oldest = self.events.pop(0)

        successors = self._edges.pop(oldest.id, set())
        for successor_id in successors:
            reverse = self._reverse_edges.get(successor_id)
            if reverse is not None:
                reverse.discard(oldest.id)

        predecessors = self._reverse_edges.pop(oldest.id, set())
        for predecessor_id in predecessors:
            forward = self._edges.get(predecessor_id)
            if forward is not None:
                forward.discard(oldest.id)

    def _reachable_forward(self, start_id: int) -> set[int]:
        visited: set[int] = set()
        stack: list[int] = [start_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for successor in self._edges.get(current, set()):
                if successor not in visited:
                    stack.append(successor)
        return visited

    def _reachable_backward(self, start_id: int) -> set[int]:
        visited: set[int] = set()
        stack: list[int] = [start_id]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for predecessor in self._reverse_edges.get(current, set()):
                if predecessor not in visited:
                    stack.append(predecessor)
        return visited

    def interval(self, x: MarketEvent, z: MarketEvent) -> list[MarketEvent]:
        """
        Return events strictly between x and z in the causal order.

        Fix 4.15: removed the early return that became incorrect under
        eviction churn. The check `if x.id not in _edges and z.id not in
        _reverse_edges` would discard valid intervals whenever the terminal
        events had lost their direct neighbours to eviction, even though a
        path still existed through surviving intermediates.
        """
        forward = self._reachable_forward(x.id)
        backward = self._reachable_backward(z.id)
        between = forward & backward
        return [
            e for e in self.events
            if e.id in between and e.id not in (x.id, z.id)
        ]


# ─────────────────────────────────────────────────────────────────────
# Causal geometry
# ─────────────────────────────────────────────────────────────────────


class CausalGeometry:
    """
    Curvature and local depth, both computed over a trailing window.

    Fix 4.14: both windows are now explicitly trailing (0 <= dt <= window).
    The prior implementation used abs(dt) <= window, which peeked into the
    future on historical events and introduced a look-ahead bias whenever
    the geometry was evaluated on a stored poset rather than the live tail.
    """

    def __init__(self, poset: CausalPoset, curvature_window: float = 50.0):
        self.poset = poset
        self.curvature_window_seconds = curvature_window

    def curvature(self, event: MarketEvent) -> float:
        recent = [
            e for e in self.poset.events
            if 0.0 <= event.timestamp - e.timestamp <= self.curvature_window_seconds
        ]
        if len(recent) < 3:
            return 0.0
        prices = np.array([e.price for e in recent], dtype=float)
        if prices.std() < 1e-9:
            return 0.0
        diffs = np.diff(prices)
        accelerations = np.diff(diffs)
        if accelerations.size == 0:
            return 0.0
        return float(np.std(accelerations) / (abs(np.mean(prices)) + 1e-9))

    def local_depth(self, event: MarketEvent) -> float:
        return float(sum(
            e.volume for e in self.poset.events
            if 0.0 <= event.timestamp - e.timestamp <= self.curvature_window_seconds
        ))


# ─────────────────────────────────────────────────────────────────────
# Horizon detector
# ─────────────────────────────────────────────────────────────────────


class HorizonDetector:
    """
    Detect volume shocks in the trailing window.

    Fix 4.3: the baseline for the z-score is sampled at non-overlapping
    bucket boundaries (one sample every curvature_window_seconds), so the
    history is not autocorrelated. The prior implementation appended a
    new local_depth on every tick, but consecutive local_depth values
    share most of their contributing events, inflating the effective
    variance and causing the |z| > threshold test to fire far more often
    than the nominal rate.

    Fix 4.4: sample standard deviation uses ddof=1.
    """

    def __init__(self, geometry: CausalGeometry, window: int = 100,
                 z_threshold: float = 3.0):
        self.geometry = geometry
        self.window = window
        self.z_threshold = z_threshold
        self.volume_history: list[float] = []
        self._last_history_update: float = 0.0
        self._bucket_seconds: float = geometry.curvature_window_seconds
        self._initialised: bool = False

    def detect(self, event: MarketEvent) -> bool:
        local_volume = self.geometry.local_depth(event)

        if not self._initialised:
            self._initialised = True
            self._last_history_update = event.timestamp
            return False

        elapsed = event.timestamp - self._last_history_update
        if elapsed >= self._bucket_seconds:
            self.volume_history.append(local_volume)
            if len(self.volume_history) > self.window:
                self.volume_history.pop(0)
            self._last_history_update = event.timestamp

        if len(self.volume_history) < 20:
            return False

        history = np.array(self.volume_history, dtype=float)
        mean = float(history.mean())
        sigma = float(history.std(ddof=1)) + 1e-9
        z = (local_volume - mean) / sigma
        return abs(z) > self.z_threshold


# ─────────────────────────────────────────────────────────────────────
# Causal attractor
# ─────────────────────────────────────────────────────────────────────


class CausalAttractor:
    """
    Compute the attractor price for a horizon event.

    The attractor is the price level with the greatest causal support in the
    interval between the horizon event and the latest observation within the
    lookahead window.

    The pre-fix implementation was called on the horizon event itself, at
    which point there were no future events in the poset, so `future_events`
    was always empty and the function returned None on every horizon. The
    fix requires the caller to invoke this method only after subsequent
    events have arrived; CausalSetMomentumBot.generate_signal enforces that
    invariant by tracking active horizons.
    """

    def __init__(self, poset: CausalPoset, geometry: CausalGeometry,
                 lookahead_seconds: float = 30.0, price_bin: float = 0.1):
        self.poset = poset
        self.geometry = geometry
        self.lookahead_seconds = lookahead_seconds
        self.price_bin = price_bin

    def compute(
        self,
        horizon_event: MarketEvent,
    ) -> tuple[Optional[float], float, dict]:
        if not self.poset.contains(horizon_event.id):
            # Horizon has been evicted from the poset; cannot evaluate.
            return None, 0.0, {}

        future_events = [
            e for e in self.poset.events
            if e.timestamp > horizon_event.timestamp
            and e.timestamp <= horizon_event.timestamp + self.lookahead_seconds
        ]
        if not future_events:
            return None, 0.0, {}

        observation = max(future_events, key=lambda e: e.timestamp)
        interval = self.poset.interval(horizon_event, observation)
        if not interval:
            return None, 0.0, {}

        price_levels = np.unique(
            [round(e.price / self.price_bin) * self.price_bin for e in interval]
        )
        scores: dict[float, float] = {}
        for price_level in price_levels:
            causal_support = sum(
                e.volume for e in interval
                if abs(e.price - price_level) <= self.price_bin / 2.0
            )
            if causal_support > 0:
                scores[float(price_level)] = float(causal_support)

        if not scores:
            return None, 0.0, {}

        target = max(scores, key=scores.get)
        total_volume = float(sum(e.volume for e in interval))
        strength = float(scores[target] / (total_volume + 1e-9))
        curvature = self.geometry.curvature(observation)

        return target, strength, {
            "curvature": curvature,
            "support": scores[target],
            "interval_size": len(interval),
            "observation_timestamp": observation.timestamp,
        }


# ─────────────────────────────────────────────────────────────────────
# Bottleneck scanner
# ─────────────────────────────────────────────────────────────────────


class BottleneckScanner:
    """
    Detect price bottlenecks.

    A bottleneck is a price level with high surrounding volume and low (or
    zero) volume at the level itself. The score is (1 + V_left) * (1 + V_right)
    / (V_centre + 1e-9), which diverges as V_centre approaches zero.

    Fix 2: the volume profile is constructed on a contiguous grid from the
    lowest to the highest populated price level, so empty levels are scored
    at full strength. The prior implementation iterated only over populated
    levels, so a genuine price vacuum was never visible.

    Fix 4.11: the `mid` parameter is now used to filter out distant levels
    whose score is numerically large but economically irrelevant.
    """

    def __init__(self, poset: CausalPoset, mid_filter_fraction: float = 0.05):
        self.poset = poset
        self.mid_filter_fraction = mid_filter_fraction

    def scan(
        self,
        mid: float,
        tick: float = 0.1,
        threshold: float = 5.0,
    ) -> list[tuple[float, float]]:
        profile = self._volume_profile(tick)
        if not profile:
            return []

        populated_levels = sorted(profile.keys())
        lo = populated_levels[0]
        hi = populated_levels[-1]
        n_bins = int(round((hi - lo) / tick)) + 1
        full_levels = [lo + index * tick for index in range(n_bins)]
        full_profile = {level: profile.get(level, 0.0) for level in full_levels}

        max_distance = abs(mid) * self.mid_filter_fraction

        bottlenecks: list[tuple[float, float]] = []
        for index, level in enumerate(full_levels):
            if abs(level - mid) > max_distance:
                continue
            v_left = full_profile[full_levels[index - 1]] if index > 0 else 0.0
            v_right = (
                full_profile[full_levels[index + 1]]
                if index < len(full_levels) - 1 else 0.0
            )
            v_centre = full_profile[level] + 1e-9
            score = (1.0 + v_left) * (1.0 + v_right) / v_centre
            if score > threshold:
                bottlenecks.append((float(level), float(score)))

        return sorted(bottlenecks, key=lambda pair: -pair[1])

    def _volume_profile(self, tick: float) -> dict[float, float]:
        profile: dict[float, float] = {}
        for event in self.poset.events[-500:]:
            if event.volume > 0:
                level = round(event.price / tick) * tick
                profile[level] = profile.get(level, 0.0) + event.volume
        return profile


# ─────────────────────────────────────────────────────────────────────
# The bot
# ─────────────────────────────────────────────────────────────────────


class CausalSetMomentumBot:
    """
    Causal Set Momentum Bot.

    Fix 1: horizons are tracked in `active_horizons` and evaluated on
    subsequent ticks once future events exist in the poset. See
    generate_signal for the full logic.
    """

    def __init__(
        self,
        symbol: str = "BTC/USDT",
        delta: float = 2.0,
        curvature_window: float = 50.0,
        horizon_sigma: float = 3.0,
        bottleneck_threshold: float = 5.0,
        risk_per_trade: float = 0.01,
        max_position: float = 1.0,
        tick_size: float = 0.1,
    ):
        self.symbol = symbol
        self.delta = delta
        # Fix 4.5: documented as seconds, not event count.
        self.curvature_window_seconds = curvature_window
        self.horizon_sigma = horizon_sigma
        self.bottleneck_threshold = bottleneck_threshold
        self.risk_per_trade = risk_per_trade
        self.max_position = max_position
        self.tick_size = tick_size

        self.poset = CausalPoset(max_events=2000, delta=delta)
        self.geometry = CausalGeometry(self.poset, curvature_window)
        self.horizon_detector = HorizonDetector(
            self.geometry, z_threshold=horizon_sigma
        )
        self.attractor = CausalAttractor(
            self.poset, self.geometry, lookahead_seconds=30.0
        )
        self.bottleneck = BottleneckScanner(self.poset)

        self.position: float = 0.0
        self.cash: float = 0.0
        self.equity_curve: list[float] = []

        self._last_bid: float = 0.0
        self._last_ask: float = 0.0
        self._last_depth: tuple[float, float] = (0.0, 0.0)

        # Fix 1: active horizon tracking.
        self.active_horizons: list[MarketEvent] = []

    def ingest_orderbook(
        self,
        orderbook: dict,
        timestamp: float,
    ) -> Optional[MarketEvent]:
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])
        if not bids or not asks:
            return None

        # Fix 4.8: index by position so a three-element level (price, volume,
        # count) does not raise ValueError. The prior unpacking
        # `best_bid, best_bid_vol = bids[0]` assumed exactly two elements.
        best_bid_price = float(bids[0][0])
        best_ask_price = float(asks[0][0])
        best_bid_volume = float(bids[0][1])
        best_ask_volume = float(asks[0][1])
        bid_depth = float(sum(float(level[1]) for level in bids[:5]))
        ask_depth = float(sum(float(level[1]) for level in asks[:5]))

        spread = best_ask_price - best_bid_price
        mid = (best_bid_price + best_ask_price) / 2.0

        event_type = self._classify_event(
            best_bid_price, best_ask_price, bid_depth, ask_depth
        )

        if event_type is EventType.TRADE_BUY:
            volume = best_bid_volume
        elif event_type is EventType.TRADE_SELL:
            volume = best_ask_volume
        else:
            volume = 0.0

        event = MarketEvent(
            timestamp=float(timestamp),
            price=float(mid),
            volume=float(volume),
            event_type=event_type,
            spread=float(spread),
            bid_depth=float(bid_depth),
            ask_depth=float(ask_depth),
        )

        self._last_bid = best_bid_price
        self._last_ask = best_ask_price
        self._last_depth = (bid_depth, ask_depth)

        self.poset.add_event(event)
        return event

    def _classify_event(
        self,
        best_bid_price: float,
        best_ask_price: float,
        bid_depth: float,
        ask_depth: float,
    ) -> EventType:
        previous_bid = self._last_bid
        previous_ask = self._last_ask
        previous_bid_depth, previous_ask_depth = self._last_depth

        if previous_bid > 0 and best_bid_price > previous_bid:
            return EventType.TRADE_BUY
        if previous_ask > 0 and best_ask_price < previous_ask:
            return EventType.TRADE_SELL
        if (
            previous_bid_depth > 0
            and abs(bid_depth - previous_bid_depth) / previous_bid_depth > 0.05
        ):
            return EventType.LIMIT_BID
        if (
            previous_ask_depth > 0
            and abs(ask_depth - previous_ask_depth) / previous_ask_depth > 0.05
        ):
            return EventType.LIMIT_ASK
        return EventType.UNKNOWN

    def generate_signal(self, event: MarketEvent) -> Optional[dict]:
        """
        Generate a signal, or return None.

        Fix 1 — the corrected sequencing is:

            1. If `event` triggers a horizon, append it to active_horizons.
               Do NOT attempt to compute the attractor on it in the same tick.

            2. Expire active horizons that have exceeded the lookahead window
               or have been evicted from the poset.

            3. For every remaining active horizon whose timestamp strictly
               precedes the current event, attempt the attractor computation.
               A horizon that yields a non-None target is removed from
               active_horizons and its signal is returned.

        The prior implementation called the attractor on the event that had
        just arrived, at which point no future events existed in the poset,
        so the attractor always returned None and generate_signal always
        returned None. n_signals was 0 on every run of every mode.
        """
        if self.horizon_detector.detect(event):
            self.active_horizons.append(event)

        lookahead = self.attractor.lookahead_seconds
        self.active_horizons = [
            horizon for horizon in self.active_horizons
            if event.timestamp - horizon.timestamp <= lookahead
            and self.poset.contains(horizon.id)
        ]

        for horizon in list(self.active_horizons):
            if event.timestamp <= horizon.timestamp:
                continue

            target, strength, extra = self.attractor.compute(horizon)
            if target is None:
                continue

            self.active_horizons.remove(horizon)

            curvature = float(extra.get("curvature", 0.0))
            bottlenecks = self.bottleneck.scan(
                mid=event.price,
                tick=self.tick_size,
                threshold=self.bottleneck_threshold,
            )

            if bottlenecks:
                nearest_price, _ = bottlenecks[0]
                if abs(nearest_price - target) < 0.02 * event.price:
                    target = 0.7 * target + 0.3 * nearest_price

            direction = 1 if target > event.price else -1

            # Fix 4.12: the comment now matches the code. A high-curvature,
            # low-strength regime is treated as noise, not as a momentum
            # opportunity, so direction is zeroed in that regime.
            if curvature > 0.7 and strength < 0.3:
                direction = 0

            if direction == 0:
                return None

            return {
                "direction": int(direction),
                "target": float(target),
                "strength": float(strength),
                "curvature": float(curvature),
                "mid": float(event.price),
                "bottlenecks": bottlenecks,
                "horizon_timestamp": float(horizon.timestamp),
                "observation_timestamp": float(event.timestamp),
            }

        return None

    def on_event(self, event: MarketEvent) -> Optional[dict]:
        signal = self.generate_signal(event)
        self._update_equity(event.price)
        return signal

    def _update_equity(self, price: float) -> None:
        self.equity_curve.append(self.cash + self.position * price)


# ─────────────────────────────────────────────────────────────────────
# Backtest and live loops
# ─────────────────────────────────────────────────────────────────────


def _annualised_sharpe(
    equity_curve: list[float],
    n_events: int,
    span_seconds: float,
) -> float:
    """
    Annualise the per-event Sharpe ratio by the actual event frequency.

    Fix 4.9: the prior implementation multiplied by sqrt(252), which assumes
    daily returns. The returns here are per-event, so the annualisation
    factor must reflect the event rate, not the day rate.
    """
    if n_events < 2 or span_seconds <= 0 or len(equity_curve) < 2:
        return 0.0
    equity = np.asarray(equity_curve, dtype=float)
    returns = np.diff(equity) / (np.abs(equity[:-1]) + 1e-9)
    if returns.size < 2 or returns.std(ddof=1) < 1e-12:
        return 0.0
    events_per_year = n_events / (span_seconds / (365.25 * 24.0 * 3600.0))
    return float(
        (returns.mean() / (returns.std(ddof=1) + 1e-9))
        * np.sqrt(events_per_year)
    )


def backtest_csv(bot: CausalSetMomentumBot, path: str) -> dict:
    n_signals = 0
    n_events = 0
    first_timestamp: Optional[float] = None
    last_timestamp: Optional[float] = None

    with open(path, "r", encoding="utf-8") as file_handle:
        reader = csv.DictReader(file_handle)
        for row in reader:
            orderbook = {
                "bids": [[float(row["bid"]), float(row.get("bid_vol", 1.0))]],
                "asks": [[float(row["ask"]), float(row.get("ask_vol", 1.0))]],
            }
            timestamp = float(row.get("timestamp", n_events))
            event = bot.ingest_orderbook(orderbook, timestamp)
            if event is None:
                continue
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp
            n_events += 1
            signal = bot.on_event(event)
            if signal is not None:
                n_signals += 1

    span = (last_timestamp - first_timestamp) if last_timestamp is not None and first_timestamp is not None else 0.0

    equity = bot.equity_curve if bot.equity_curve else [0.0]
    return {
        "n_events": n_events,
        "n_signals": n_signals,
        "total_return": float(equity[-1] - equity[0]),
        "sharpe": _annualised_sharpe(equity, n_events, span),
    }


def synthetic_backtest(bot: CausalSetMomentumBot, n_events: int = 5000) -> dict:
    """
    Synthetic backtest.

    Fix 4.10: the price process is Ornstein–Uhlenbeck around a hidden
    attractor at 100.0, not a driftless random walk. The bot must discover
    the attractor via causal support; a driftless random walk has no such
    structure and cannot validate the attractor code.
    """
    rng = np.random.default_rng(42)
    hidden_attractor = 100.0
    price = hidden_attractor
    timestamp = 0.0
    n_signals = 0
    n_actual_events = 0

    for index in range(n_events):
        timestamp += 0.1
        price += 0.1 * (hidden_attractor - price) + rng.normal(0.0, 2.0)
        spread = 0.5
        best_bid = price - spread / 2.0
        best_ask = price + spread / 2.0

        if index % 100 == 0:
            bid_volume = 50.0 + rng.random() * 50.0
            ask_volume = 50.0 + rng.random() * 50.0
        else:
            bid_volume = 1.0 + rng.random()
            ask_volume = 1.0 + rng.random()

        orderbook = {
            "bids": [[best_bid, bid_volume]],
            "asks": [[best_ask, ask_volume]],
        }
        event = bot.ingest_orderbook(orderbook, timestamp)
        if event is None:
            continue
        n_actual_events += 1
        if bot.on_event(event) is not None:
            n_signals += 1

    equity = bot.equity_curve if bot.equity_curve else [0.0]
    span = timestamp
    return {
        "n_events": n_actual_events,
        "n_signals": n_signals,
        "total_return": float(equity[-1] - equity[0]),
        "sharpe": _annualised_sharpe(equity, n_actual_events, span),
    }


def live_trading(bot: CausalSetMomentumBot, exchange_id: str = "binance") -> None:
    import ccxt
    exchange = getattr(ccxt, exchange_id)()
    while True:
        try:
            orderbook = exchange.fetch_order_book(bot.symbol)

            # Fix 4.13: normalise the timestamp to seconds regardless of
            # whether the exchange returns milliseconds, seconds, or None.
            raw_timestamp = orderbook.get("timestamp")
            if raw_timestamp is None:
                timestamp = time.time()
            elif raw_timestamp > 1e12:
                timestamp = raw_timestamp / 1000.0
            else:
                timestamp = raw_timestamp

            event = bot.ingest_orderbook(orderbook, timestamp)
            if event is None:
                time.sleep(1)
                continue
            signal = bot.on_event(event)
            if signal is not None:
                print(
                    f"[{timestamp:.3f}] direction={signal['direction']:+d} "
                    f"target={signal['target']:.2f} "
                    f"strength={signal['strength']:.3f} "
                    f"curvature={signal['curvature']:.4f}"
                )
        except Exception as exception:
            print(f"error: {type(exception).__name__}: {exception}")
        time.sleep(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["backtest", "live", "synthetic"], default="synthetic"
    )
    parser.add_argument("--csv", type=str)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument(
        "--curvature-window", type=float, default=50.0,
        dest="curvature_window",
        help="Curvature window in seconds, not event count.",
    )
    parser.add_argument("--horizon-sigma", type=float, default=3.0)
    parser.add_argument("--bottleneck-threshold", type=float, default=5.0)

    arguments = parser.parse_args()

    bot = CausalSetMomentumBot(
        symbol=arguments.symbol,
        delta=arguments.delta,
        curvature_window=arguments.curvature_window,
        horizon_sigma=arguments.horizon_sigma,
        bottleneck_threshold=arguments.bottleneck_threshold,
    )

    if arguments.mode == "backtest":
        if not arguments.csv:
            print("--csv required for backtest mode", file=sys.stderr)
            return 1
        result = backtest_csv(bot, arguments.csv)
    elif arguments.mode == "live":
        live_trading(bot)
        return 0
    else:
        result = synthetic_backtest(bot)

    print(f"n_events: {result['n_events']}")
    print(f"n_signals: {result['n_signals']}")
    print(f"total_return: {result['total_return']:.4f}")
    print(f"sharpe: {result['sharpe']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())