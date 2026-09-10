"""
Causal Set Momentum Trading Bot
================================
A non-chronological, order-flow-driven trading system based on
Causal Set Theory (Order + Number = Geometry).

Dependencies:
    pip install ccxt pandas numpy scipy networkx

Author: Quantitative Causal Set Research
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from enum import Enum
import logging
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("CausalSetMomentum")


# ============================================================
# SECTION 1: DISCRETE SPACETIME ATOMS (Market Events)
# ============================================================

class EventType(Enum):
    LIMIT_PLACE   = "L"   # New resting liquidity
    LIMIT_CANCEL  = "C"   # Liquidity withdrawal
    MARKET_EXEC   = "M"   # Liquidity consumption
    TRADE_BUY     = "TB"  # Aggressor buy
    TRADE_SELL    = "TS"  # Aggressor sell


@dataclass
class MarketEvent:
    """
    A discrete spacetime atom x in the market causal set C.

    Attributes
    ----------
    id : int
        Unique identifier (auxiliary clock index).
    timestamp : float
        Clock time (used only for bookkeeping, never as a causal parameter).
    event_type : EventType
    price : float
        Price level of the event.
    volume : float
        Volume involved (0 for pure cancellations).
    side : int
        +1 for bid-side, -1 for ask-side.
    bid : float
        Best bid at event time.
    ask : float
        Best ask at event time.
    bid_depth : float
        Total bid volume at best level.
    ask_depth : float
        Total ask volume at best level.
    """
    id: int
    timestamp: float
    event_type: EventType
    price: float
    volume: float
    side: int
    bid: float
    ask: float
    bid_depth: float
    ask_depth: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0 if self.bid > 0 and self.ask > 0 else self.price

    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.ask > self.bid > 0 else 0.0

    def __hash__(self):
        return self.id

    def __repr__(self):
        return (f"E{self.id}[{self.event_type.value}] "
                f"p={self.price:.4f} v={self.volume:.2f} "
                f"mid={self.mid:.4f}")


# ============================================================
# SECTION 2: CAUSAL POSET (Partially Ordered Set)
# ============================================================

@dataclass
class CausalPoset:
    """
    The market poset (C, ≺) with incremental construction.

    The causal relation x ≺ y holds iff:
        (C1) t_x < t_y                     (temporal precedence)
        (C2) im(x) ∩ dom(y) ≠ ∅            (state overlap)
        (C3) ∂Φ_y / ∂Φ_x ≠ 0               (causal susceptibility)

    We implement a tractable approximation: x ≺ y if x is within
    the causal window δ of y AND x modifies a book component that
    y depends upon (price level overlap or depth change overlap).
    """
    delta: float = 0.5          # causal window (seconds)
    max_events: int = 2000      # rolling memory bound

    events: List[MarketEvent] = field(default_factory=list)
    _id_counter: int = 0
    _edges: Dict[int, Set[int]] = field(default_factory=lambda: defaultdict(set))
    _reverse_edges: Dict[int, Set[int]] = field(
        default_factory=lambda: defaultdict(set)
    )

    def add_event(self, event: MarketEvent) -> None:
        """Insert a new event and compute its causal relations."""
        event.id = self._id_counter
        self._id_counter += 1
        self.events.append(event)

        # Compute causal predecessors
        for prior in self._predecessors(event):
            self._edges[prior.id].add(event.id)
            self._reverse_edges[event.id].add(prior.id)

        # Enforce rolling memory
        if len(self.events) > self.max_events:
            self._evict_oldest()

    def _predecessors(self, y: MarketEvent) -> List[MarketEvent]:
        """
        Find all x in the poset such that x ≺ y.

        Approximate causal relation:
            - t_y - t_x ≤ δ
            - |p_x - p_y| ≤ spread_y  (price-level overlap)
              OR x modified depth at a level that y depends on
        """
        preds = []
        for x in reversed(self.events):
            if x.id == y.id:
                continue
            dt = y.timestamp - x.timestamp
            if dt <= 0:
                continue
            if dt > self.delta:
                break  # events are time-ordered; stop early

            # (C2) State overlap: price-level proximity
            price_overlap = abs(x.price - y.price) <= max(y.spread, 1e-9)

            # (C3) Causal susceptibility: x changes depth that y uses
            depth_overlap = (
                (x.event_type in (EventType.LIMIT_PLACE,
                                  EventType.LIMIT_CANCEL,
                                  EventType.MARKET_EXEC))
                and (abs(x.bid_depth - y.bid_depth) > 1e-9
                     or abs(x.ask_depth - y.ask_depth) > 1e-9)
            )

            if price_overlap or depth_overlap:
                preds.append(x)
        return preds

    def _evict_oldest(self) -> None:
        """Remove the oldest events to bound memory."""
        n_evict = len(self.events) - self.max_events
        evicted_ids = {e.id for e in self.events[:n_evict]}
        self.events = self.events[n_evict:]
        for eid in evicted_ids:
            self._edges.pop(eid, None)
            self._reverse_edges.pop(eid, None)

    def interval(self, x: MarketEvent, z: MarketEvent) -> List[MarketEvent]:
        """
        Return the causal interval [x, z] = {y : x ≺ y ≺ z}.
        Uses BFS on the DAG.
        """
        if x.id not in self._edges and z.id not in self._reverse_edges:
            return []
        forward = self._reachable_forward(x.id)
        backward = self._reachable_backward(z.id)
        between = forward & backward
        return [e for e in self.events if e.id in between]

    def _reachable_forward(self, start_id: int) -> Set[int]:
        visited, stack = set(), [start_id]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            stack.extend(self._edges.get(nid, set()))
        return visited

    def _reachable_backward(self, end_id: int) -> Set[int]:
        visited, stack = set(), [end_id]
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            stack.extend(self._reverse_edges.get(nid, set()))
        return visited

    def predecessors(self, event: MarketEvent) -> List[MarketEvent]:
        return [e for e in self.events if e.id in self._reverse_edges.get(event.id, set())]

    def successors(self, event: MarketEvent) -> List[MarketEvent]:
        return [e for e in self.events if e.id in self._edges.get(event.id, set())]


# ============================================================
# SECTION 3: CAUSAL GEOMETRY (Order + Number → Geometry)
# ============================================================

class CausalGeometry:
    """
    Computes emergent geometric quantities from the market poset.

    Key quantities:
        V(x,z)   : interval volume (transacted value in [x,z])
        f(x,z)   : ordering fraction = |[x,z]| / N_max
        R(x)     : causal curvature at x
    """

    def __init__(self, poset: CausalPoset, curvature_window: int = 50):
        self.poset = poset
        self.curvature_window = curvature_window

    def interval_volume(self, x: MarketEvent, z: MarketEvent) -> float:
        """V(x,z) = Σ V_y for trade events y in [x,z]."""
        interval = self.poset.interval(x, z)
        return sum(e.volume for e in interval
                   if e.event_type in (EventType.TRADE_BUY,
                                       EventType.TRADE_SELL,
                                       EventType.MARKET_EXEC))

    def ordering_fraction(self, x: MarketEvent, z: MarketEvent) -> float:
        """
        f(x,z) = N([x,z]) / N_max(x,z)

        N_max is approximated as the total number of events in the
        temporal window between x and z.
        """
        interval = self.poset.interval(x, z)
        n_interval = len(interval)
        if n_interval == 0:
            return 0.0
        n_window = sum(1 for e in self.poset.events
                       if x.timestamp <= e.timestamp <= z.timestamp)
        return n_interval / max(n_window, 1)

    def causal_curvature(self, event: MarketEvent) -> float:
        """
        R(x) = (1/N) Σ_{z ∈ N(x)} (1 - f(x,z))

        High R → fragmented causal relations → high volatility.
        Low R  → dense causal relations → trending regime.
        """
        recent = [e for e in self.poset.events
                  if abs(e.timestamp - event.timestamp) <= self.curvature_window
                  and e.id != event.id]
        if not recent:
            return 0.0
        fractions = []
        for z in recent[:20]:  # cap for performance
            f = self.ordering_fraction(event, z)
            fractions.append(1.0 - f)
        return float(np.mean(fractions)) if fractions else 0.0

    def local_depth(self, event: MarketEvent) -> float:
        """Local causal density = total volume in causal window."""
        window_events = [e for e in self.poset.events
                         if abs(e.timestamp - event.timestamp) <= self.curvature_window]
        return sum(e.volume for e in window_events) + 1e-9


# ============================================================
# SECTION 4: HORIZON DETECTION
# ============================================================

class HorizonDetector:
    """
    Detects market causal horizons: events that truncate the
    causal past by consuming liquidity.

    A horizon exists at x if the local volume change ΔV exceeds
    a dynamic threshold (e.g., 3σ of recent volume distribution).
    """

    def __init__(self, geometry: CausalGeometry, sigma_threshold: float = 3.0):
        self.geom = geometry
        self.sigma = sigma_threshold
        self.volume_history: List[float] = []

    def detect(self, event: MarketEvent) -> bool:
        """
        Returns True if `event` generates a causal horizon.
        """
        local_v = self.geom.local_depth(event)
        self.volume_history.append(local_v)
        if len(self.volume_history) < 30:
            return False

        mu = np.mean(self.volume_history[-100:])
        sigma = np.std(self.volume_history[-100:]) + 1e-9
        z = (local_v - mu) / sigma

        is_horizon = abs(z) > self.sigma
        if is_horizon:
            logger.info(f"Horizon detected at E{event.id} | z={z:.2f}")
        return is_horizon


# ============================================================
# SECTION 5: CAUSAL ATTRACTOR (Momentum Forcing)
# ============================================================

class CausalAttractor:
    """
    Computes the causal attractor A = set of future price paths
    causally reachable from a horizon event x_H.

    Ω(p) = Σ_{z: p(z)=p} N([x_H, z])
    p_attractor = argmax_p Ω(p)
    """

    def __init__(self, poset: CausalPoset, geometry: CausalGeometry,
                 price_bin: float = 0.5):
        self.poset = poset
        self.geom = geometry
        self.price_bin = price_bin

    def compute(self, horizon_event: MarketEvent,
                lookahead_seconds: float = 30.0) -> Tuple[Optional[float], float, Dict[float, float]]:
        """
        Returns
        -------
        p_attractor : float or None
            Target price level (causal attractor).
        omega_max : float
            Maximum Ω(p) value.
        omega_map : dict
            Full distribution Ω(p).
        """
        future_events = [e for e in self.poset.events
                         if e.timestamp > horizon_event.timestamp
                         and e.timestamp <= horizon_event.timestamp + lookahead_seconds]

        if not future_events:
            return None, 0.0, {}

        omega: Dict[float, float] = defaultdict(float)

        for z in future_events:
            iv = self.poset.interval(horizon_event, z)
            if not iv:
                continue
            n_causal = len(iv)
            p_bin = round(z.price / self.price_bin) * self.price_bin
            omega[p_bin] += n_causal * (z.volume + 1e-6)

        if not omega:
            return None, 0.0, {}

        total = sum(omega.values())
        omega_norm = {p: v / total for p, v in omega.items()}
        p_attractor = max(omega_norm, key=omega_norm.get)
        return p_attractor, omega_norm[p_attractor], omega_norm


# ============================================================
# SECTION 6: BOTTLENECK SCANNER
# ============================================================

class BottleneckScanner:
    """
    Identifies price nodes with negligible liquidity surrounded
    by dense liquidity (causal bottleneck / liquidity vacuum).

    B(p) = (1 / V(p)) * Π_{p' ∈ N(p)} (1 + V(p'))
    """

    def __init__(self, poset: CausalPoset, n_levels: int = 10):
        self.poset = poset
        self.n_levels = n_levels

    def volume_profile(self, mid: float, tick: float = 0.1) -> Dict[float, float]:
        """Build volume profile from recent events."""
        profile: Dict[float, float] = defaultdict(float)
        for e in self.poset.events[-500:]:
            if e.volume > 0:
                level = round(e.price / tick) * tick
                profile[level] += e.volume
        return dict(profile)

    def scan(self, mid: float, tick: float = 0.1,
             threshold: float = 5.0) -> List[Tuple[float, float]]:
        """
        Returns list of (price_level, bottleneck_score) above threshold.
        """
        profile = self.volume_profile(mid, tick)
        if len(profile) < 3:
            return []

        levels = sorted(profile.keys())
        scores = []
        for i, p in enumerate(levels):
            v_p = profile[p] + 1e-9
            neighbors = []
            if i > 0:
                neighbors.append(levels[i - 1])
            if i < len(levels) - 1:
                neighbors.append(levels[i + 1])
            prod = 1.0
            for pn in neighbors:
                prod *= (1.0 + profile[pn])
            score = prod / v_p
            if score > threshold:
                scores.append((p, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores


# ============================================================
# SECTION 7: MAIN TRADING BOT
# ============================================================

class CausalSetMomentumBot:
    """
    Causal Set Momentum Trading Bot.

    Pipeline:
        1. Ingest LOB event → MarketEvent
        2. Insert into CausalPoset
        3. Update CausalGeometry
        4. Detect Horizon
        5. Compute Causal Attractor → target price
        6. Scan Bottlenecks
        7. Generate signal (direction, target)
        8. Risk management & execution
    """

    def __init__(self,
                 symbol: str = "BTC/USDT",
                 delta: float = 0.5,
                 curvature_window: int = 50,
                 horizon_sigma: float = 3.0,
                 bottleneck_threshold: float = 5.0,
                 risk_per_trade: float = 0.01,
                 max_position: float = 1.0,
                 tick_size: float = 0.1):

        self.symbol = symbol
        self.tick_size = tick_size
        self.risk_per_trade = risk_per_trade
        self.max_position = max_position

        # Core components
        self.poset = CausalPoset(delta=delta)
        self.geometry = CausalGeometry(self.poset, curvature_window)
        self.horizon_detector = HorizonDetector(self.geometry, horizon_sigma)
        self.attractor = CausalAttractor(self.poset, self.geometry, tick_size)
        self.bottleneck = BottleneckScanner(self.poset)

        # State
        self.position: float = 0.0
        self.entry_price: float = 0.0
        self.capital: float = 10000.0
        self.equity_curve: List[float] = [self.capital]
        self.signal_log: List[dict] = []

        # LOB snapshot cache
        self._last_bid: float = 0.0
        self._last_ask: float = 0.0
        self._last_bid_depth: float = 0.0
        self._last_ask_depth: float = 0.0

    # --------------------------------------------------------
    # 7.1 Event Ingestion
    # --------------------------------------------------------

    def ingest_orderbook(self, ob: dict, timestamp: float) -> Optional[MarketEvent]:
        """
        Convert a ccxt-style order book snapshot into a MarketEvent.

        Parameters
        ----------
        ob : dict
            {'bids': [[price, volume], ...], 'asks': [[price, volume], ...]}
        timestamp : float
        """
        bids = ob.get("bids", [])
        asks = ob.get("asks", [])
        if not bids or not asks:
            return None

        best_bid, best_bid_vol = bids[0]
        best_ask, best_ask_vol = asks[0]
        bid_depth = sum(v for _, v in bids[:5])
        ask_depth = sum(v for _, v in asks[:5])

        # Classify event type from LOB deltas
        event_type = self._classify_event(
            best_bid, best_ask, best_bid_vol, best_ask_vol,
            bid_depth, ask_depth
        )

        # Determine price and side of the event
        if event_type in (EventType.TRADE_BUY,):
            price, side, volume = best_ask, +1, best_ask_vol
        elif event_type in (EventType.TRADE_SELL,):
            price, side, volume = best_bid, -1, best_bid_vol
        elif event_type == EventType.LIMIT_PLACE:
            price, side, volume = best_bid, +1, best_bid_vol
        elif event_type == EventType.LIMIT_CANCEL:
            price, side, volume = best_bid, +1, 0.0
        else:
            price, side, volume = (best_bid + best_ask) / 2, 0, 0.0

        event = MarketEvent(
            id=0, timestamp=timestamp, event_type=event_type,
            price=price, volume=volume, side=side,
            bid=best_bid, ask=best_ask,
            bid_depth=bid_depth, ask_depth=ask_depth
        )

        # Update cache
        self._last_bid = best_bid
        self._last_ask = best_ask
        self._last_bid_depth = bid_depth
        self._last_ask_depth = ask_depth

        self.poset.add_event(event)
        return event

    def _classify_event(self, bid, ask, bid_vol, ask_vol,
                        bid_depth, ask_depth) -> EventType:
        """Heuristic event classification from LOB deltas."""
        if self._last_bid == 0.0:
            return EventType.LIMIT_PLACE

        bid_delta = bid_depth - self._last_bid_depth
        ask_delta = ask_depth - self._last_ask_depth
        spread = ask - bid

        # Price moved up through ask → buy aggression
        if bid > self._last_bid + 1e-9:
            return EventType.TRADE_BUY
        # Price moved down through bid → sell aggression
        if ask < self._last_ask - 1e-9:
            return EventType.TRADE_SELL
        # Depth increased significantly
        if bid_delta > 0.5 * self._last_bid_depth:
            return EventType.LIMIT_PLACE
        if ask_delta > 0.5 * self._last_ask_depth:
            return EventType.LIMIT_PLACE
        # Depth decreased significantly
        if bid_delta < -0.3 * self._last_bid_depth:
            return EventType.LIMIT_CANCEL
        if ask_delta < -0.3 * self._last_ask_depth:
            return EventType.LIMIT_CANCEL
        return EventType.MARKET_EXEC

    # --------------------------------------------------------
    # 7.2 Signal Generation
    # --------------------------------------------------------

    def generate_signal(self, event: MarketEvent) -> Optional[dict]:
        """
        Full causal-set signal pipeline for a single event.

        Returns a dict with keys:
            direction : +1 / -1 / 0
            target    : target price
            strength  : attractor strength (0–1)
            curvature : causal curvature
            bottlenecks : list of (price, score)
        """
        # 1. Compute curvature
        curvature = self.geometry.causal_curvature(event)

        # 2. Horizon detection
        is_horizon = self.horizon_detector.detect(event)
        if not is_horizon:
            return None

        # 3. Causal attractor
        p_target, strength, omega_map = self.attractor.compute(event)
        if p_target is None:
            return None

        # 4. Direction from attractor vs. current price
        current_mid = event.mid
        if p_target > current_mid + self.tick_size:
            direction = +1
        elif p_target < current_mid - self.tick_size:
            direction = -1
        else:
            direction = 0

        # 5. Bottleneck scan (filter / confirm)
        bottlenecks = self.bottleneck.scan(
            current_mid, self.tick_size, threshold=5.0
        )
        bottleneck_prices = [p for p, _ in bottlenecks]

        # If target is behind a bottleneck, adjust target
        if direction > 0:
            ahead = [p for p in bottleneck_prices if p > current_mid]
            if ahead:
                p_target = min(ahead)
        elif direction < 0:
            ahead = [p for p in bottleneck_prices if p < current_mid]
            if ahead:
                p_target = max(ahead)

        # 6. Curvature filter: avoid high-curvature regimes for
        #    mean-reversion, favor for momentum
        if curvature > 0.7 and strength < 0.3:
            direction = 0

        signal = {
            "event_id": event.id,
            "timestamp": event.timestamp,
            "direction": direction,
            "target": p_target,
            "strength": strength,
            "curvature": curvature,
            "mid": current_mid,
            "bottlenecks": bottlenecks,
        }
        self.signal_log.append(signal)
        return signal

    # --------------------------------------------------------
    # 7.3 Risk Management
    # --------------------------------------------------------

    def position_size(self, signal: dict) -> float:
        """
        Kelly-inspired sizing scaled by attractor strength.
        """
        if signal["direction"] == 0:
            return 0.0
        base = self.risk_per_trade * self.capital
        strength = signal.get("strength", 0.0)
        size = base * (0.5 + strength) / max(signal["mid"], 1e-9)
        return float(np.clip(size, 0.0, self.max_position))

    def check_stop_loss(self, current_price: float) -> bool:
        """Stop if unrealized loss exceeds 2% of capital."""
        if self.position == 0.0:
            return False
        pnl = (current_price - self.entry_price) * self.position
        if pnl < -0.02 * self.capital:
            logger.warning(f"Stop-loss triggered | PnL={pnl:.2f}")
            return True
        return False

    # --------------------------------------------------------
    # 7.4 Execution (paper / live)
    # --------------------------------------------------------

    def execute(self, signal: dict, current_price: float) -> None:
        """Paper execution of the signal."""
        direction = signal["direction"]
        if direction == 0:
            return

        size = self.position_size(signal)
        if size == 0.0:
            return

        # Close opposite position
        if self.position * direction < 0:
            pnl = (current_price - self.entry_price) * self.position
            self.capital += pnl
            self.position = 0.0

        # Open / add
        self.position += direction * size
        self.entry_price = current_price
        logger.info(
            f"EXECUTED dir={direction} size={size:.4f} "
            f"@ {current_price:.2f} | target={signal['target']:.2f}"
        )

    def update_equity(self, current_price: float) -> None:
        """Mark-to-market equity."""
        unrealized = (current_price - self.entry_price) * self.position
        equity = self.capital + unrealized
        self.equity_curve.append(equity)

    # --------------------------------------------------------
    # 7.5 Main Loop
    # --------------------------------------------------------

    def on_event(self, event: MarketEvent) -> Optional[dict]:
        """
        Called for every new market event.
        Returns the signal if one was generated.
        """
        signal = self.generate_signal(event)
        if signal is None:
            return None

        mid = event.mid
        if self.check_stop_loss(mid):
            self.position = 0.0
            return None

        self.execute(signal, mid)
        self.update_equity(mid)
        return signal


# ============================================================
# SECTION 8: BACKTESTING ENGINE
# ============================================================

def backtest_csv(csv_path: str,
                 symbol: str = "BTC/USDT",
                 **bot_kwargs) -> dict:
    """
    Backtest the bot on a CSV of LOB snapshots.

    Expected CSV columns:
        timestamp, bid, ask, bid_volume, ask_volume
    """
    df = pd.read_csv(csv_path)
    bot = CausalSetMomentumBot(symbol=symbol, **bot_kwargs)

    signals = []
    for _, row in df.iterrows():
        ob = {
            "bids": [[row["bid"], row["bid_volume"]]],
            "asks": [[row["ask"], row["ask_volume"]]],
        }
        event = bot.ingest_orderbook(ob, row["timestamp"])
        if event is None:
            continue
        sig = bot.on_event(event)
        if sig:
            signals.append(sig)

    equity = pd.Series(bot.equity_curve)
    returns = equity.pct_change().dropna()

    results = {
        "n_events": len(bot.poset.events),
        "n_signals": len(signals),
        "final_equity": equity.iloc[-1],
        "total_return": (equity.iloc[-1] / equity.iloc[0] - 1) * 100,
        "sharpe": (returns.mean() / (returns.std() + 1e-9)) * np.sqrt(252),
        "max_drawdown": ((equity / equity.cummax() - 1).min()) * 100,
        "signals": signals,
        "equity_curve": equity,
    }

    logger.info(
        f"Backtest complete | "
        f"Return={results['total_return']:.2f}% | "
        f"Sharpe={results['sharpe']:.2f} | "
        f"MaxDD={results['max_drawdown']:.2f}% | "
        f"Signals={results['n_signals']}"
    )
    return results


# ============================================================
# SECTION 9: LIVE TRADING (ccxt.pro)
# ============================================================

async def live_trading(symbol: str = "BTC/USDT",
                       exchange_id: str = "binance",
                       delta: float = 0.5,
                       **bot_kwargs):
    """
    Live trading loop using ccxt.pro WebSocket order book stream.

    Requires:
        pip install ccxt[pro]
    """
    try:
        import ccxt.pro as ccxtpro
    except ImportError:
        logger.error("ccxt.pro not installed. Run: pip install ccxt[pro]")
        return

    exchange = getattr(ccxtpro, exchange_id)()
    bot = CausalSetMomentumBot(symbol=symbol, delta=delta, **bot_kwargs)

    logger.info(f"Live trading started | {symbol} on {exchange_id}")

    try:
        while True:
            ob = await exchange.watch_order_book(symbol, limit=20)
            ts = ob.get("timestamp", time.time() * 1000) / 1000.0
            event = bot.ingest_orderbook(ob, ts)
            if event is None:
                continue

            signal = bot.on_event(event)
            if signal:
                logger.info(
                    f"SIGNAL | dir={signal['direction']} "
                    f"target={signal['target']:.2f} "
                    f"strength={signal['strength']:.3f} "
                    f"curv={signal['curvature']:.3f}"
                )
                # Place live order here via exchange.create_order(...)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await exchange.close()


# ============================================================
# SECTION 10: ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Causal Set Momentum Trading Bot"
    )
    parser.add_argument("--mode", choices=["backtest", "live"],
                        default="backtest")
    parser.add_argument("--csv", type=str, default=None,
                        help="Path to CSV for backtesting")
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--delta", type=float, default=0.5,
                        help="Causal window in seconds")
    parser.add_argument("--horizon-sigma", type=float, default=3.0)
    parser.add_argument("--risk", type=float, default=0.01)

    args = parser.parse_args()

    if args.mode == "backtest":
        if args.csv is None:
            # Synthetic demo data
            np.random.seed(42)
            n = 5000
            price = 50000.0
            prices = [price]
            for _ in range(n):
                price += np.random.randn() * 2.0
                prices.append(max(price, 1.0))
            df = pd.DataFrame({
                "timestamp": np.arange(n + 1) * 0.1,
                "bid": np.array(prices) - 0.5,
                "ask": np.array(prices) + 0.5,
                "bid_volume": np.random.exponential(10, n + 1),
                "ask_volume": np.random.exponential(10, n + 1),
            })
            df.to_csv("/tmp/demo_lob.csv", index=False)
            args.csv = "/tmp/demo_lob.csv"
            logger.info("Generated synthetic demo data")

        results = backtest_csv(
            args.csv, symbol=args.symbol,
            delta=args.delta,
            horizon_sigma=args.horizon_sigma,
            risk_per_trade=args.risk,
        )
        print("\n=== BACKTEST RESULTS ===")
        for k, v in results.items():
            if k not in ("signals", "equity_curve"):
                print(f"  {k}: {v}")

    elif args.mode == "live":
        import asyncio
        asyncio.run(live_trading(
            symbol=args.symbol,
            exchange_id=args.exchange,
            delta=args.delta,
            horizon_sigma=args.horizon_sigma,
            risk_per_trade=args.risk,
        ))