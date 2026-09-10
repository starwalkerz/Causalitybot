# Causal Set Momentum (CSM)
### A Non-Chronological Trading Framework Derived from Causal Set Theory

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Research](https://img.shields.io/badge/status-research--prototype-orange.svg)]()

---

## Table of Contents

1. [Overview](#overview)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Architecture](#architecture)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Configuration Reference](#configuration-reference)
7. [Module Reference](#module-reference)
8. [Mathematical Reference](#mathematical-reference)
9. [Backtesting](#backtesting)
10. [Live Trading](#live-trading)
11. [Performance & Limitations](#performance--limitations)
12. [Roadmap](#roadmap)
13. [Disclaimer](#disclaimer)
14. [References](#references)
15. [License](#license)

---

## Overview

**Causal Set Momentum (CSM)** is a research-grade algorithmic trading framework
that translates the core principles of **Causal Set Theory** (a discrete approach
to quantum gravity) into a predictive, **non-chronological** pattern-recognition
system for financial markets.

The central hypothesis is that market structure — trends, support, resistance,
momentum — is not fundamental. It is an **emergent geometric illusion** generated
by the aggregate network of discrete, causally linked transactions. Traditional
technical indicators (moving averages, RSI, MACD) impose an artificial uniform
clock on a system that has no fundamental clock. CSM abandons clock-time entirely
and replaces it with:

> **Order + Number = Geometry**

- **Order** → the causal relation ≺ between order-flow events
- **Number** → the counting measure (transacted volume) on the poset
- **Geometry** → the emergent market metric, curvature, horizons, and attractors

The framework is a single Python module (`causal_set_bot.py`) with no external
machine-learning dependencies — only NumPy, pandas, and optionally ccxt for live
data.

---

## Theoretical Foundation

### 1. Discrete Spacetime Atoms

Each atomic state change in the limit order book is a **market event**:

```
x = (t_x, b_x, a_x, p_x_mid, V_x)
```

Event types include limit placements, cancellations, market executions, and
aggressor trades.

### 2. The Causal Relation

For two events x, y ∈ C:

```
x ≺ y  iff  (C1) t_x < t_y                  (temporal precedence)
            (C2) im(x) ∩ dom(y) ≠ ∅         (state overlap)
            (C3) ∂Φ_y / ∂Φ_x ≠ 0            (causal susceptibility)
```

Chronological precedence alone is **not** sufficient. Mere coincidence is not
causation.

### 3. Emergent Geometry

The counting measure on causal intervals recovers the volume element:

```
N([x,z]) ≈ ρ · V(x,z)
```

The **causal curvature** at an event x:

```
R(x) = (1/N) · Σ_{z ∈ N(x)} (1 − f(x,z))
f(x,z) = N([x,z]) / N_max(x,z)
```

- **Low curvature** → dense, regular order flow → smooth trending regime
- **High curvature** → fragmented, disordered order flow → high volatility / breakout

### 4. Horizons & Attractors

A large institutional order or liquidity vacuum generates a **market causal
horizon** H — a subset of events that truncates the causal past available to
future participants. The surviving future paths form a **causal attractor**:

```
Ω(p) = Σ_{z : p(z)=p} N([x_H, z])
p_attractor = argmax_p Ω(p)
```

Momentum emerges as the entropy-reducing consequence of information loss at the
horizon.

### 5. Bottlenecks & Liquidity Vacuums

A **causal bottleneck** is a price node p* through which every future price path
must pass:

```
B(p) = (1 / V(p)) · Π_{p' ∈ N(p)} (1 + V(p'))
```

High B(p) indicates a liquidity vacuum — a magnet for price.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CausalSetMomentumBot                         │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐     │
│  │  MarketEvent │──▶│ CausalPoset  │──▶│ CausalGeometry   │     │
│  │  (atom)      │   │  (C, ≺)      │   │  (metric, curv.) │     │
│  └──────────────┘   └──────────────┘   └──────────────────┘     │
│         │                  │                    │               │
│         │                  │                    ▼               │
│         │                  │           ┌──────────────────┐     │
│         │                  │           │ HorizonDetector  │     │
│         │                  │           └──────────────────┘     │
│         │                  │                    │               │
│         │                  ▼                    ▼               │
│         │           ┌──────────────┐   ┌──────────────────┐     │
│         │           │BottleneckScan│   │ CausalAttractor  │     │
│         │           └──────────────┘   └──────────────────┘     │
│         │                  │                    │               │
│         └──────────────────┴──────────┬─────────┘               │
│                                       ▼                         │
│                          ┌────────────────────────┐             │
│                          │  Signal Generation     │             │
│                          │  + Risk Management     │             │
│                          └────────────────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Ingest**       → LOB snapshot converted to `MarketEvent`
2. **Poset Insert** → event added to `CausalPoset`, causal predecessors computed
3. **Geometry**     → interval volumes, ordering fractions, curvature computed
4. **Horizon**      → σ-threshold on local depth
5. **Attractor**    → Ω(p) distribution over future price bins
6. **Bottleneck**   → liquidity vacuum identification
7. **Signal**       → direction, target, strength, curvature
8. **Execute**      → position sizing, stop-loss, paper or live

---

## Installation

### Prerequisites

- Python **3.10 or higher**
- pip **21.0+**
- (Optional) A C++ compiler if installing NumPy/SciPy from source
- (Optional) An exchange API key for live trading

### Automated Installation

```bash
git clone https://github.com/your-org/causal-set-momentum.git
cd causal-set-momentum
chmod +x install.sh
./install.sh
```

The installer will:

1. Verify Python version ≥ 3.10
2. Create a virtual environment in `.venv/`
3. Upgrade pip, setuptools, wheel
4. Install core dependencies (NumPy, pandas, SciPy)
5. Optionally install live-trading dependencies (ccxt / ccxt.pro)
6. Optionally install dev dependencies (pytest, black, ruff)
7. Verify the installation with a self-test
8. Generate a demo LOB CSV for smoke testing

### Manual Installation

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Requirements

**Core** (`requirements.txt`):

```
numpy>=1.24,<3.0
pandas>=2.0,<3.0
scipy>=1.11,<2.0
```

**Live trading** (`requirements-live.txt`):

```
ccxt>=4.0
ccxt[pro]>=4.0
websockets>=12.0
aiohttp>=3.9
```

**Development** (`requirements-dev.txt`):

```
-r requirements.txt
pytest>=7.4
pytest-asyncio>=0.23
pytest-cov>=4.1
black>=23.0
ruff>=0.1
mypy>=1.5
ipython>=8.0
matplotlib>=3.7
```

---

## Quick Start

### 1. Backtest on Synthetic Data

```bash
python causal_set_bot.py --mode backtest
```

Expected output:

```
=== BACKTEST RESULTS ===
  n_events: 5000
  n_signals: 47
  final_equity: 10234.71
  total_return: 2.35
  sharpe: 1.42
  max_drawdown: -1.87
```

### 2. Backtest on Your Own Data

CSV schema:

| timestamp    | bid     | ask     | bid_volume | ask_volume |
|--------------|---------|---------|------------|------------|
| 1700000000.0 | 49999.5 | 50000.5 | 12.3       | 9.7        |
| 1700000000.1 | 49999.8 | 50000.2 | 14.1       | 8.2        |

```bash
python causal_set_bot.py --mode backtest --csv /path/to/lob.csv
```

### 3. Live Trading (Paper)

```bash
python causal_set_bot.py --mode live --symbol BTC/USDT --exchange binance
```

> ⚠ Live mode is a **signal emitter** by default. Order execution is
> deliberately decoupled — see `live_trading()` in the source to integrate
> your broker's API.

### 4. Programmatic API

```python
from causal_set_bot import CausalSetMomentumBot

bot = CausalSetMomentumBot(
    symbol="BTC/USDT",
    delta=0.5,               # causal window (s)
    horizon_sigma=3.0,       # horizon detection threshold
    risk_per_trade=0.01,     # 1% of capital per signal
)

ob = {"bids": [[49999.5, 12.3]], "asks": [[50000.5, 9.7]]}
event  = bot.ingest_orderbook(ob, timestamp=1700000000.0)
signal = bot.on_event(event)

if signal:
    print(f"DIR={signal['direction']}  TARGET={signal['target']}")
```

---

## Configuration Reference

| Parameter              | Type  | Default       | Description                                        |
|------------------------|-------|---------------|----------------------------------------------------|
| `symbol`               | str   | `"BTC/USDT"`  | Trading pair identifier                            |
| `delta`                | float | `0.5`         | Causal window δ (s). Bounds search only            |
| `curvature_window`     | int   | `50`          | Events used to estimate local curvature            |
| `horizon_sigma`        | float | `3.0`         | σ-threshold for horizon detection                  |
| `bottleneck_threshold` | float | `5.0`         | Minimum B(p) score to register a bottleneck        |
| `risk_per_trade`       | float | `0.01`        | Fraction of capital risked per signal              |
| `max_position`         | float | `1.0`         | Max position size (base units)                     |
| `tick_size`            | float | `0.1`         | Price bin width for attractor computation          |

### Tuning Notes

- **`delta`** — increase for slower markets (`2.0` daily bars); decrease for
  HFT (`0.05`).
- **`horizon_sigma`** — lower = more horizons (higher signal freq, more noise);
  `4.0` for conservative regimes.
- **`risk_per_trade`** — do not exceed `0.02` without portfolio-level controls.

---

## Module Reference

### `MarketEvent`

Discrete spacetime atom.

```python
MarketEvent(
    id=0,
    timestamp=1700000000.0,
    event_type=EventType.TRADE_BUY,
    price=50000.5, volume=0.5, side=+1,
    bid=49999.5, ask=50000.5,
    bid_depth=120.0, ask_depth=95.0,
)
```

### `CausalPoset`

Rolling-memory DAG of causal relations.

| Method                | Description                                            |
|-----------------------|--------------------------------------------------------|
| `add_event(event)`    | Insert and compute causal predecessors                 |
| `interval(x, z)`      | Return [x, z] = {y : x ≺ y ≺ z} via BFS                |
| `predecessors(event)` | Direct causal ancestors                                |
| `successors(event)`   | Direct causal descendants                              |

### `CausalGeometry`

| Method                       | Returns                              |
|------------------------------|--------------------------------------|
| `interval_volume(x, z)`      | Transacted value in [x, z]           |
| `ordering_fraction(x, z)`    | f(x, z) ∈ [0, 1]                     |
| `causal_curvature(event)`    | R(x)                                 |
| `local_depth(event)`         | Volume in causal window              |

### `HorizonDetector`

```python
detect(event) -> bool
```

Returns `True` when local depth exceeds `sigma_threshold` σ of recent
distribution.

### `CausalAttractor`

```python
compute(horizon_event, lookahead_seconds=30.0)
    -> (p_attractor: float, omega_max: float, omega_map: dict)
```

### `BottleneckScanner`

```python
scan(mid: float, tick: float, threshold: float)
    -> List[Tuple[price, score]]
```

### `CausalSetMomentumBot`

```python
ingest_orderbook(ob: dict, timestamp: float) -> Optional[MarketEvent]
generate_signal(event: MarketEvent)         -> Optional[dict]
position_size(signal: dict)                 -> float
check_stop_loss(current_price: float)       -> bool
execute(signal: dict, current_price: float) -> None
on_event(event: MarketEvent)                -> Optional[dict]
```

---

## Mathematical Reference

**Definition 1 — Market Event.**
A tuple x = (t_x, b_x, a_x, p_x, V_x) representing an atomic LOB state change.

**Definition 2 — Causal Order.**
x ≺ y iff (C1) t_x < t_y, (C2) im(x) ∩ dom(y) ≠ ∅, (C3) ∂Φ_y / ∂Φ_x ≠ 0.

**Definition 3 — Causal Interval Volume.**
V(x,z) = Σ_{y ∈ [x,z]} V_y · 1[y is a trade]

**Definition 4 — Ordering Fraction.**
f(x,z) = N([x,z]) / N_max(x,z)

**Definition 5 — Causal Curvature.**
R(x) = (1/N) Σ_{z ∈ N(x)} (1 − f(x,z))

**Definition 6 — Causal Horizon.**
H ⊆ C such that N([x,y]) = 0 for all x ∈ H⁻, y ≻ H.

**Definition 7 — Causal Attractor.**
A = argmax_p Ω(p), where Ω(p) = Σ_{z : p(z)=p} N([x_H, z]).

**Definition 8 — Causal Cluster.**
Maximal K ⊆ C such that N([x,y]) ≥ θ for all x, y ∈ K.

**Definition 9 — Causal Bottleneck.**
p* such that ∀ z ≻ x, p(z) ≠ p* ⟹ ∃ y ≺ z : p(y) = p*.

**Proposition — Momentum Forcing.**
E[Δp | x_H] = (Σ_{z ≻ x_H} p(z) N([x_H, z])) / (Σ_{z ≻ x_H} N([x_H, z])) − p(x_H)

---

## Backtesting

### CLI

```bash
python causal_set_bot.py --mode backtest --csv my_data.csv --delta 0.5 --risk 0.01
```

### API

```python
from causal_set_bot import backtest_csv

results = backtest_csv(
    "lob.csv",
    symbol="BTC/USDT",
    delta=0.5,
    horizon_sigma=3.0,
    risk_per_trade=0.01,
)
print(results["total_return"], results["sharpe"])
results["equity_curve"].plot()
```

### Output Metrics

| Metric          | Description                     |
|-----------------|---------------------------------|
| `n_events`      | Total events ingested           |
| `n_signals`     | Signals emitted                 |
| `final_equity`  | Terminal equity                 |
| `total_return`  | Percent return                  |
| `sharpe`        | Annualized Sharpe (√252)        |
| `max_drawdown`  | Peak-to-trough %                |
| `signals`       | Full signal log (list of dicts) |
| `equity_curve`  | `pd.Series` of equity           |

---

## Live Trading

### ccxt.pro Setup

```bash
pip install "ccxt[pro]"
```

### Running Live

```bash
python causal_set_bot.py --mode live --symbol BTC/USDT --exchange binance
```

### Order Execution Hook

`live_trading()` emits signals but does **not** place orders by default.
Insert your broker call at the marked location:

```python
if signal:
    logger.info(f"SIGNAL | dir={signal['direction']} target={signal['target']}")
    # ── EXECUTION HOOK ──
    # await exchange.create_order(
    #     symbol=symbol,
    #     type="limit",
    #     side="buy" if signal["direction"] > 0 else "sell",
    #     amount=size,
    #     price=signal["target"],
    # )
```

### Safety Recommendations

1. Start with **paper trading** for at least 30 days.
2. Enforce a **hard daily loss limit** at the portfolio layer.
3. Run the bot in a **separate process** with a watchdog.
4. Log every signal and execution to a durable store (SQLite, Postgres).
5. Reconcile positions against the exchange hourly.

---

## Performance & Limitations

### Strengths

- **Non-chronological** — no reliance on fixed time windows; adapts to irregular
  event arrival.
- **Order-flow-native** — uses the raw LOB directly, without intermediate
  aggregation.
- **Interpretable** — every signal is traceable to a horizon, attractor, or
  bottleneck.
- **Horizon-aware** — detects liquidity truncations explicitly rather than
  statistically.

### Limitations

- **Computational cost** — interval BFS is O(|C|) per signal; rolling memory
  bounds this.
- **Parameter sensitivity** — `delta` and `horizon_sigma` require tuning per
  instrument.
- **No fundamental clock** — some instruments (e.g. daily bars) have natural
  clocks; CSM may add little value there.
- **Causal approximation** — the exact susceptibility ∂Φ_y / ∂Φ_x is
  approximated by price-level and depth overlap.
- **No portfolio logic** — position sizing is single-instrument; extend for
  multi-asset.

### Complexity

| Operation              | Complexity                       |
|------------------------|----------------------------------|
| `add_event`            | O(δ · r̄) where r̄ = event rate    |
| `interval(x, z)`       | O(|C|) BFS worst-case            |
| `causal_curvature`     | O(k · \|C\|) with k neighborhood |
| Full signal pipeline   | O(\|C\| log \|C\|)               |

---

## Roadmap

- [ ] **v0.2** — Persistent homology for causal cluster detection
- [ ] **v0.3** — Multi-asset poset with cross-asset causal edges
- [ ] **v0.4** — Bayesian hyperparameter optimization for `delta` and `sigma`
- [ ] **v0.5** — Order execution adapters (Binance, Bybit, IBKR)
- [ ] **v0.6** — Real-time dashboard (Streamlit) for curvature visualization
- [ ] **v1.0** — Full paper-trading release with reconciliation and audit logs

---

## Disclaimer

**This software is provided for research and educational purposes only.**

- It is **not** investment advice.
- Past performance (including backtest results) does **not** guarantee future
  results.
- Trading involves substantial risk of loss.
- The authors assume **no liability** for any financial losses incurred through
  the use of this software.
- You are solely responsible for compliance with all applicable laws and
  regulations in your jurisdiction.

By using this software, you acknowledge that you have read, understood, and
accepted this disclaimer.

---

## References

1. Bombelli, L., Lee, J., Meyer, D., & Sorkin, R. D. (1987). *Space-time as a
   causal set*. Physical Review Letters, 59(5), 521.
2. Sorkin, R. D. (1991). *Spacetime and causal sets*. In Relativity and
   Gravitation: Classical and Quantum.
3. Henson, J. (2006). *The causal set approach to quantum gravity*.
   arXiv:gr-qc/0601121.
4. Surya, S. (2019). *The causal set approach to quantum gravity*. Living
   Reviews in Relativity, 22(1), 5.
5. Cont, R., Kukanov, A., & Stoikov, S. (2014). *The price impact of order book
   events*. Journal of Financial Econometrics, 12(1), 47–88.
6. Bouchaud, J.-P., Bonart, J., Donier, J., & Gould, M. (2018). *Trades, Quotes
   and Prices*. Cambridge University Press.

---

## License

MIT License. See [LICENSE](LICENSE) for full text.

---

<p align="center">
  <em>"Without causality, there is no time." — Rafael Sorkin</em>
</p>