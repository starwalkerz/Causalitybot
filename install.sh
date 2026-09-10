#!/usr/bin/env bash
# =============================================================================
#  Causal Set Momentum (CSM) — Installation Script
# =============================================================================
#  Sets up a self-contained virtual environment, installs dependencies,
#  and verifies the installation with a smoke test.
#
#  Usage:
#      chmod +x install.sh
#      ./install.sh                 # interactive
#      ./install.sh --yes           # non-interactive, core only
#      ./install.sh --with-live     # install live-trading deps
#      ./install.sh --with-dev      # install dev deps
#      ./install.sh --all           # everything
# =============================================================================

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_NAME="causal-set-momentum"
VENV_DIR=".venv"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10
REQ_CORE="requirements.txt"
REQ_LIVE="requirements-live.txt"
REQ_DEV="requirements-dev.txt"
BOT_FILE="causal_set_bot.py"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

WITH_LIVE=0
WITH_DEV=0
NON_INTERACTIVE=0
SKIP_TEST=0

# ─────────────────────────────────────────────────────────────────────────────
# Argument Parsing
# ─────────────────────────────────────────────────────────────────────────────

print_usage() {
    cat <<EOF
${PROJECT_NAME} installer

Usage:
    ./install.sh [OPTIONS]

Options:
    --with-live       Install live-trading dependencies (ccxt / ccxt.pro)
    --with-dev        Install development dependencies
    --all             Install everything (core + live + dev)
    --yes, -y         Non-interactive: accept defaults, core only
    --skip-test       Skip the post-install smoke test
    --help, -h        Show this message and exit

Examples:
    ./install.sh --all
    ./install.sh --with-live
    ./install.sh -y
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-live)  WITH_LIVE=1; shift ;;
        --with-dev)   WITH_DEV=1;  shift ;;
        --all)        WITH_LIVE=1; WITH_DEV=1; shift ;;
        --yes|-y)     NON_INTERACTIVE=1; shift ;;
        --skip-test)  SKIP_TEST=1; shift ;;
        --help|-h)    print_usage; exit 0 ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; print_usage; exit 1 ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

log_info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_err()   { echo -e "${RED}[FAIL]${NC}  $*" >&2; }
log_step()  { echo -e "\n${CYAN}▶ $*${NC}"; }

confirm() {
    if [[ "$NON_INTERACTIVE" -eq 1 ]]; then return 0; fi
    read -r -p "$1 [y/N] " reply
    [[ "$reply" =~ ^[Yy]$ ]]
}

# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

echo -e "${CYAN}"
cat <<'EOF'
   ______                 __        _____      __  __
  / ____/___ __   ______ _/ /     / ___/___  / /_/ /____
 / /   / __ `/ | / / __ `/ /_____ \__ \/ _ \/ __/ __/ _ \
/ /___/ /_/ /| |/ / /_/ / /_____/___/ /  __/ /_/ /_/  __/
\____/\__,_/ |___/\__,_/_/     /____/\___/\__/\__/\___/

        Causal Set Momentum (CSM) — Installer
        "Order + Number = Geometry"
EOF
echo -e "${NC}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Locate Project Root
# ─────────────────────────────────────────────────────────────────────────────

log_step "Step 1/7 — Locating project root"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ ! -f "$BOT_FILE" ]]; then
    log_err "$BOT_FILE not found in $SCRIPT_DIR"
    log_err "Please run this script from the project root."
    exit 1
fi
log_ok "Project root: $SCRIPT_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Verify Python
# ─────────────────────────────────────────────────────────────────────────────

log_step "Step 2/7 — Verifying Python interpreter"

PYTHON_BIN=""
for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        version=$("$candidate" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        major=${version%%.*}
        minor=${version##*.}
        if [[ "$major" -gt "$PYTHON_MIN_MAJOR" ]] || \
           { [[ "$major" -eq "$PYTHON_MIN_MAJOR" ]] && [[ "$minor" -ge "$PYTHON_MIN_MINOR" ]]; }; then
            PYTHON_BIN="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    log_err "Python ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR}+ is required but not found."
    log_err "Install it from https://www.python.org/downloads/ and re-run."
    exit 1
fi

PY_VERSION=$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")')
log_ok "Using $PYTHON_BIN (version $PY_VERSION)"

if ! "$PYTHON_BIN" -c 'import venv' >/dev/null 2>&1; then
    log_err "The 'venv' module is unavailable. Install python3-venv and retry."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Create Virtual Environment
# ─────────────────────────────────────────────────────────────────────────────

log_step "Step 3/7 — Creating virtual environment"

if [[ -d "$VENV_DIR" ]]; then
    if confirm "Virtual environment '$VENV_DIR' already exists. Recreate?"; then
        log_info "Removing existing $VENV_DIR ..."
        rm -rf "$VENV_DIR"
    else
        log_info "Reusing existing $VENV_DIR"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    log_ok "Created $VENV_DIR"
fi

if [[ -f "$VENV_DIR/bin/activate" ]]; then
    ACTIVATE="$VENV_DIR/bin/activate"
    VENV_PY="$VENV_DIR/bin/python"
elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    ACTIVATE="$VENV_DIR/Scripts/activate"
    VENV_PY="$VENV_DIR/Scripts/python"
else
    log_err "Could not find activation script in $VENV_DIR"
    exit 1
fi

# shellcheck disable=SC1090
source "$ACTIVATE"
log_ok "Activated virtual environment"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Upgrade Packaging Tools
# ─────────────────────────────────────────────────────────────────────────────

log_step "Step 4/7 — Upgrading pip, setuptools, wheel"
"$VENV_PY" -m pip install --upgrade pip setuptools wheel --quiet
log_ok "Packaging tools up to date"

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Install Dependencies
# ─────────────────────────────────────────────────────────────────────────────

log_step "Step 5/7 — Installing dependencies"

# Core
if [[ -f "$REQ_CORE" ]]; then
    log_info "Installing core requirements from $REQ_CORE ..."
    "$VENV_PY" -m pip install -r "$REQ_CORE"
    log_ok "Core dependencies installed"
else
    log_warn "$REQ_CORE not found — installing defaults"
    "$VENV_PY" -m pip install "numpy>=1.24" "pandas>=2.0" "scipy>=1.11"
    log_ok "Core dependencies installed"
fi

# Live (optional)
if [[ "$WITH_LIVE" -eq 1 ]] || { [[ "$NON_INTERACTIVE" -eq 0 ]] && confirm "Install live-trading deps (ccxt)?"; }; then
    if [[ -f "$REQ_LIVE" ]]; then
        log_info "Installing live-trading requirements from $REQ_LIVE ..."
        if "$VENV_PY" -m pip install -r "$REQ_LIVE"; then
            log_ok "Live-trading dependencies installed"
        else
            log_warn "ccxt[pro] install failed — continuing without WebSocket support"
        fi
    else
        log_warn "$REQ_LIVE not found — attempting direct install"
        if "$VENV_PY" -m pip install "ccxt>=4.0" "ccxt[pro]>=4.0"; then
            log_ok "Live-trading dependencies installed"
        else
            log_warn "Live install failed — continuing"
        fi
    fi
fi

# Dev (optional)
if [[ "$WITH_DEV" -eq 1 ]] || { [[ "$NON_INTERACTIVE" -eq 0 ]] && confirm "Install dev deps (pytest, black, ruff)?"; }; then
    if [[ -f "$REQ_DEV" ]]; then
        log_info "Installing dev requirements from $REQ_DEV ..."
        "$VENV_PY" -m pip install -r "$REQ_DEV"
        log_ok "Dev dependencies installed"
    else
        log_warn "$REQ_DEV not found — installing defaults"
        "$VENV_PY" -m pip install "pytest>=7.4" "black>=23.0" "ruff>=0.1" "mypy>=1.5"
        log_ok "Dev dependencies installed"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Smoke Test
# ─────────────────────────────────────────────────────────────────────────────

log_step "Step 6/7 — Running smoke test"

if [[ "$SKIP_TEST" -eq 1 ]]; then
    log_info "Smoke test skipped (--skip-test)"
else
    if "$VENV_PY" - <<'PYEOF'
import sys
try:
    import numpy, pandas, scipy
except ImportError as e:
    print(f'Missing dependency: {e}', file=sys.stderr)
    sys.exit(1)

sys.path.insert(0, '.')
try:
    import causal_set_bot
except Exception as e:
    print(f'Failed to import causal_set_bot: {e}', file=sys.stderr)
    sys.exit(1)

from causal_set_bot import CausalSetMomentumBot

bot = CausalSetMomentumBot(symbol='TEST/USDT', delta=0.5)
ob  = {'bids': [[99.5, 10.0]], 'asks': [[100.5, 10.0]]}
ev  = bot.ingest_orderbook(ob, timestamp=0.0)
assert ev is not None, 'Failed to ingest order book'
assert len(bot.poset.events) == 1, 'Poset should contain 1 event'

ob2 = {'bids': [[99.6, 12.0]], 'asks': [[100.4, 9.0]]}
ev2 = bot.ingest_orderbook(ob2, timestamp=0.1)
assert ev2 is not None
assert len(bot.poset.events) == 2

print('SMOKE_TEST_OK')
PYEOF
    then
        log_ok "Smoke test passed"
    else
        log_err "Smoke test failed — see output above"
        exit 1
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Generate Demo Data & Final Instructions
# ─────────────────────────────────────────────────────────────────────────────

log_step "Step 7/7 — Generating demo data & finalizing"

DEMO_CSV="demo_lob.csv"
if [[ ! -f "$DEMO_CSV" ]]; then
    "$VENV_PY" - <<'PYEOF'
import numpy as np
import pandas as pd

np.random.seed(42)
n = 5000
price = 50000.0
prices = [price]
for _ in range(n):
    price += np.random.randn() * 2.0
    prices.append(max(price, 1.0))

df = pd.DataFrame({
    "timestamp":   np.arange(n + 1) * 0.1,
    "bid":         np.array(prices) - 0.5,
    "ask":         np.array(prices) + 0.5,
    "bid_volume":  np.random.exponential(10, n + 1),
    "ask_volume":  np.random.exponential(10, n + 1),
})
df.to_csv("demo_lob.csv", index=False)
print(f"Wrote {len(df)} rows to demo_lob.csv")
PYEOF
    log_ok "Generated $DEMO_CSV"
else
    log_info "$DEMO_CSV already exists — skipping"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Success Banner
# ─────────────────────────────────────────────────────────────────────────────

echo -e "\n${GREEN}══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✓ Installation complete${NC}"
echo -e "${GREEN}══════════════════════════════════════════════════════════════${NC}\n"

cat <<EOF
  ${CYAN}Activate the environment:${NC}
      source $ACTIVATE

  ${CYAN}Run a quick backtest:${NC}
      python causal_set_bot.py --mode backtest

  ${CYAN}Backtest on the generated demo data:${NC}
      python causal_set_bot.py --mode backtest --csv demo_lob.csv

  ${CYAN}Start live (paper) trading:${NC}
      python causal_set_bot.py --mode live --symbol BTC/USDT --exchange binance

  ${CYAN}Read the documentation:${NC}
      cat README.md

EOF

log_info "To deactivate the environment, run: deactivate"
log_info "To uninstall, remove the '$VENV_DIR' directory."

exit 0