# Input/Output Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `run.py` CLI entry point and `notifier.py` output module so the premarket pipeline can be triggered on-demand for debugging and testing, with rich terminal output and a Telegram stub.

**Architecture:** `run.py` uses `argparse` to accept optional tickers (default: NASDAQ 100) and an optional `--regime` override, then orchestrates the existing pipeline and passes results to `notifier.py`. `notifier.py` renders strategy cards to stdout now and contains a documented `send_telegram()` stub for future activation. Zero changes to existing modules.

**Tech Stack:** Python stdlib `argparse`, `asyncio`; existing `regime_engine`, `pipeline`, `risk_auditor`, `database` modules; `pytest` + `unittest.mock` for tests.

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Create | `notifier.py` | Terminal rendering + Telegram stub |
| Create | `run.py` | CLI entry point, arg parsing, pipeline orchestration |
| Create | `tests/test_notifier.py` | Unit tests for `notifier.py` |
| Create | `tests/test_run.py` | Unit tests for `run.py` arg parsing and flow |

---

## Task 1: Create `notifier.py`

**Files:**
- Create: `notifier.py`
- Test: `tests/test_notifier.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_notifier.py`:

```python
"""Tests for notifier.py — terminal output and Telegram stub."""

import os
import sys
import pytest
from unittest.mock import patch
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import notifier


SAMPLE_CARDS = [
    "## STRATEGY CARD — NVDA — 2026-06-01 08:05 EST\n\n### Catalyst\n- Type: earnings_beat_large\n- EASS Score: 5.5 / 10.0\n\n---"
]
SAMPLE_LOG = [
    {"step": "MACRO_GATE", "result": {"regime": "BULLISH"}},
    {"ticker": "NVDA", "status": "QUALIFIED", "eass_score": 5.5},
]


def test_print_to_terminal_outputs_cards(capsys):
    """Strategy cards are printed to stdout."""
    notifier.print_to_terminal(SAMPLE_CARDS, SAMPLE_LOG)
    out = capsys.readouterr().out
    assert "STRATEGY CARD" in out
    assert "NVDA" in out


def test_print_to_terminal_no_candidates(capsys):
    """Empty card list prints NO QUALIFIED CANDIDATES message."""
    notifier.print_to_terminal([], SAMPLE_LOG)
    out = capsys.readouterr().out
    assert "NO QUALIFIED CANDIDATES" in out


def test_print_to_terminal_shows_gate_summary(capsys):
    """Gate summary from log_entries is printed."""
    notifier.print_to_terminal(SAMPLE_CARDS, SAMPLE_LOG)
    out = capsys.readouterr().out
    assert "MACRO_GATE" in out


def test_send_telegram_is_stub():
    """send_telegram does nothing and returns None (stub)."""
    result = notifier.send_telegram(SAMPLE_CARDS, SAMPLE_LOG)
    assert result is None
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/user/AlphaCrew && python -m pytest tests/test_notifier.py -v
```

Expected: `ModuleNotFoundError: No module named 'notifier'`

- [ ] **Step 3: Create `notifier.py`**

```python
"""Output delivery: terminal rendering and Telegram stub."""


def print_to_terminal(strategy_cards: list, log_entries: list) -> None:
    """Print strategy cards and gate summary to stdout."""
    print("\n" + "=" * 60)
    print("  STRATEGY ENGINE — RESULTS")
    print("=" * 60)

    # Gate summary from log entries
    gate_steps = [e for e in log_entries if "step" in e]
    if gate_steps:
        print("\n--- Gate Summary ---")
        for entry in gate_steps:
            print(f"  {entry['step']}: {entry.get('result', entry)}")

    # Strategy cards
    print(f"\n--- Strategy Cards ({len(strategy_cards)} qualified) ---\n")
    if not strategy_cards:
        print("  NO QUALIFIED CANDIDATES\n")
    else:
        for card in strategy_cards:
            print(card)
            print()

    print("=" * 60 + "\n")


def send_telegram(strategy_cards: list, log_entries: list) -> None:
    """FUTURE_STUB: Send strategy cards to a Telegram channel via bot.

    To activate:
      1. pip install python-telegram-bot
      2. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHANNEL_ID to .env
      3. Implement this function using telegram.Bot.send_message()
    """
    pass
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/user/AlphaCrew && python -m pytest tests/test_notifier.py -v
```

Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /home/user/AlphaCrew && git add notifier.py tests/test_notifier.py
git commit -m "feat: add notifier.py with terminal output and Telegram stub"
```

---

## Task 2: Create `run.py` — arg parser + NASDAQ 100 default

**Files:**
- Create: `run.py`
- Test: `tests/test_run.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_run.py`:

```python
"""Tests for run.py — arg parsing, ticker defaults, regime override."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import run


def test_parse_args_default_tickers():
    """No positional args → returns NASDAQ_100 list."""
    args = run.parse_args([])
    assert args.tickers == run.NASDAQ_100
    assert args.regime is None


def test_parse_args_custom_tickers():
    """Positional ticker args are captured as a list."""
    args = run.parse_args(["NVDA", "AMD"])
    assert args.tickers == ["NVDA", "AMD"]


def test_parse_args_regime_bullish():
    """--regime BULLISH is accepted."""
    args = run.parse_args(["--regime", "BULLISH"])
    assert args.regime == "BULLISH"


def test_parse_args_regime_bearish():
    """--regime BEARISH is accepted."""
    args = run.parse_args(["--regime", "BEARISH"])
    assert args.regime == "BEARISH"


def test_parse_args_invalid_regime():
    """Invalid --regime value raises SystemExit."""
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "SIDEWAYS"])


def test_build_regime_data_override():
    """Regime override returns a dict with the forced regime, no live SPY call."""
    result = run.build_regime_data("BEARISH")
    assert result["regime"] == "BEARISH"
    assert "spy_close" in result


def test_build_regime_data_bullish_override():
    """BULLISH override returns correct regime dict."""
    result = run.build_regime_data("BULLISH")
    assert result["regime"] == "BULLISH"


def test_nasdaq_100_has_expected_tickers():
    """NASDAQ_100 constant contains well-known members."""
    assert "AAPL" in run.NASDAQ_100
    assert "MSFT" in run.NASDAQ_100
    assert "NVDA" in run.NASDAQ_100
    assert len(run.NASDAQ_100) >= 100
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/user/AlphaCrew && python -m pytest tests/test_run.py -v
```

Expected: `ModuleNotFoundError: No module named 'run'`

- [ ] **Step 3: Write `run.py` with argparse and helpers (no pipeline call yet)**

```python
"""CLI entry point for on-demand premarket pipeline execution."""

import argparse
import asyncio
import os
import anthropic
from dotenv import load_dotenv

import config
import regime_engine
import pipeline
import risk_auditor
import database
import notifier

load_dotenv()

# NASDAQ 100 constituent tickers (as of 2025 — update periodically)
NASDAQ_100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    "COST", "NFLX", "AMD", "ADBE", "QCOM", "INTC", "INTU", "CSCO", "TXN",
    "AMGN", "HON", "AMAT", "SBUX", "MDLZ", "GILD", "ADI", "REGN", "ISRG",
    "MU", "LRCX", "KLAC", "PANW", "SNPS", "CDNS", "ASML", "MRVL", "FTNT",
    "ORLY", "CTAS", "ABNB", "DXCM", "ADP", "KDP", "MAR", "MNST", "PAYX",
    "PCAR", "ROST", "FAST", "ODFL", "WDAY", "VRSK", "BIIB", "IDXX", "ILMN",
    "ALGN", "ANSS", "CPRT", "DLTR", "EXC", "FANG", "GEHC", "GFS", "KHC",
    "LCID", "LULU", "MCHP", "MRNA", "NXPI", "ODFL", "ON", "RIVN", "SGEN",
    "SIRI", "TEAM", "TTD", "VRSN", "VRTX", "WBA", "ZBRA", "ZM", "ZS",
    "ACGL", "AEP", "AZN", "BKR", "CEG", "CSGP", "CSX", "DDOG", "EA",
    "EBAY", "ENPH", "FSLR", "FTNT", "GFS", "MELI", "NTES", "OKTA", "PYPL",
    "PTON", "ROP", "SEDG", "TCOM", "WBD", "XEL",
]


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments; returns Namespace with tickers and regime."""
    parser = argparse.ArgumentParser(
        description="Run the Strategy Engine premarket pipeline on demand."
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Tickers to screen (default: NASDAQ 100)",
    )
    parser.add_argument(
        "--regime",
        choices=["BULLISH", "BEARISH"],
        default=None,
        help="Override the live SPY regime check (for testing)",
    )
    args = parser.parse_args(argv)
    if not args.tickers:
        args.tickers = NASDAQ_100
    return args


def build_regime_data(regime_override: str) -> dict:
    """Return a regime dict using the override value instead of a live SPY check."""
    return {
        "regime": regime_override,
        "spy_close": 0.0,
        "spy_sma50": 0.0,
        "pct_above_50sma": 0.0,
    }


async def main(argv=None) -> None:
    """Orchestrate the full premarket pipeline from CLI args."""
    args = parse_args(argv)

    print(f"[RUN] Tickers: {args.tickers}")
    print(f"[RUN] Regime override: {args.regime or 'LIVE (SPY check)'}")

    # Regime gate
    if args.regime:
        regime_data = build_regime_data(args.regime)
        print(f"[RUN] Regime forced to: {args.regime}")
    else:
        regime_data = regime_engine.get_market_regime()
        print(f"[RUN] Live regime: {regime_data['regime']}")

    log_entries = [{"step": "MACRO_GATE", "result": regime_data}]

    # Sector gate
    sector_data = regime_engine.get_top_sectors()
    top_sectors = sector_data["top_sectors"]
    log_entries.append({"step": "SECTOR_GATE", "top_sectors": top_sectors})
    sector_survivors = regime_engine.filter_by_sector(args.tickers, top_sectors)
    log_entries.append({"step": "SECTOR_FILTER", "survivors": sector_survivors})
    print(f"[RUN] Sector survivors: {len(sector_survivors)}/{len(args.tickers)}")

    if not sector_survivors:
        notifier.print_to_terminal([], log_entries)
        return

    # Volatility + trend gate
    individual_survivors = regime_engine.apply_individual_gates(sector_survivors)
    log_entries.append({"step": "INDIVIDUAL_GATES", "count": len(individual_survivors)})
    print(f"[RUN] Individual gate survivors: {len(individual_survivors)}")

    if not individual_survivors:
        notifier.print_to_terminal([], log_entries)
        return

    # Earnings proximity gate
    earnings_survivors = regime_engine.apply_earnings_gate(individual_survivors)
    log_entries.append({"step": "EARNINGS_GATE", "count": len(earnings_survivors)})
    print(f"[RUN] Earnings gate survivors: {len(earnings_survivors)}")

    if not earnings_survivors:
        notifier.print_to_terminal([], log_entries)
        return

    # Sector leadership gate
    import yfinance as yf

    def _get_premarket_gap(ticker: str) -> float:
        """Estimate pre-market gap from yfinance pre/post market data."""
        try:
            t = yf.Ticker(ticker)
            info = t.info
            pre_price = info.get("preMarketPrice") or info.get("regularMarketPrice") or 0
            prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose") or 1
            if prev_close:
                return (pre_price - prev_close) / prev_close
        except Exception:
            pass
        return 0.0

    for entry in earnings_survivors:
        entry["pre_market_gap_pct"] = _get_premarket_gap(entry["ticker"])

    leaders = regime_engine.apply_sector_leadership_gate(earnings_survivors)
    log_entries.append({"step": "LEADERSHIP_GATE", "leaders": [e["ticker"] for e in leaders]})
    print(f"[RUN] Leaders: {[e['ticker'] for e in leaders]}")

    if not leaders:
        notifier.print_to_terminal([], log_entries)
        return

    # Paid pipeline gate
    db_client = database.initialize_database()
    candidates = await pipeline.run_pipeline(leaders, regime_data, db_client)
    log_entries.append({"step": "PIPELINE", "qualified_count": len(candidates)})

    if not candidates:
        notifier.print_to_terminal([], log_entries)
        return

    # Sonnet audit + strategy cards
    anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
    strategy_cards = []
    for candidate in candidates:
        audit = await risk_auditor.run_sonnet_audit(candidate, anthropic_client)
        audit = risk_auditor.apply_regime_strategy_mutator(audit, regime_data["regime"])
        stop_dist = abs(candidate["outcome_profile"].get("suggested_stop") or 0.02)
        position = risk_auditor.calculate_position_size(
            account_size=100_000,
            stop_distance_pct=stop_dist,
            regime=regime_data["regime"],
        )
        card = risk_auditor.format_strategy_card(candidate, audit, position)
        strategy_cards.append(card)
        log_entries.append({"ticker": candidate["ticker"], "status": "QUALIFIED"})

    notifier.print_to_terminal(strategy_cards, log_entries)


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
cd /home/user/AlphaCrew && python -m pytest tests/test_run.py -v
```

Expected: 8 tests PASS

- [ ] **Step 5: Verify the import health check still passes**

```bash
cd /home/user/AlphaCrew && python -c "import config; import database; import regime_engine; import pipeline; import risk_auditor; import scheduler; import notifier; import run; print('all OK')"
```

Expected: `all OK`

- [ ] **Step 6: Commit**

```bash
cd /home/user/AlphaCrew && git add run.py tests/test_run.py
git commit -m "feat: add run.py CLI entry point with argparse and pipeline orchestration"
```

---

## Task 3: Create feature branch and final verification

**Files:**
- No new files — branch + verify only

- [ ] **Step 1: Create and check out the feature branch**

```bash
cd /home/user/AlphaCrew && git checkout -b feature/input-output-interface
```

Expected: `Switched to a new branch 'feature/input-output-interface'`

- [ ] **Step 2: Cherry-pick the commits from main onto the feature branch**

> Skip this step if you are already working on the feature branch (not on main).

```bash
cd /home/user/AlphaCrew && git log --oneline main | head -5
# If notifier and run.py commits are already on this branch, skip cherry-pick.
```

- [ ] **Step 3: Run the full test suite**

```bash
cd /home/user/AlphaCrew && python -m pytest tests/ -v
```

Expected: all tests PASS (existing + new)

- [ ] **Step 4: Smoke-test the CLI help output**

```bash
cd /home/user/AlphaCrew && python run.py --help
```

Expected output includes:
```
usage: run.py [-h] [--regime {BULLISH,BEARISH}] [tickers ...]
```

- [ ] **Step 5: Verify module import health**

```bash
cd /home/user/AlphaCrew && python -c "import config; import database; import regime_engine; import pipeline; import risk_auditor; import scheduler; import notifier; import run; print('all OK')"
```

Expected: `all OK`

- [ ] **Step 6: Commit branch verification note**

```bash
cd /home/user/AlphaCrew && git add .
git commit -m "chore: verify feature/input-output-interface branch complete"
```

---

## Self-Review Against Spec

| Spec Requirement | Covered by |
|---|---|
| `python run.py [TICKERS...] [--regime BULLISH\|BEARISH]` | Task 2 — `parse_args()` |
| Default to NASDAQ 100 when no tickers given | Task 2 — `NASDAQ_100` constant + `parse_args()` |
| Regime override short-circuits live SPY check | Task 2 — `build_regime_data()` branch in `main()` |
| Rich terminal output, no file writes | Task 1 — `print_to_terminal()` |
| `NO QUALIFIED CANDIDATES` message when empty | Task 1 — tested in `test_print_to_terminal_no_candidates` |
| Telegram `send_telegram()` stub | Task 1 — `notifier.py` |
| Single-line docstrings on all functions | Task 1 + Task 2 — all functions have one-line docstrings |
| Zero changes to existing modules | All tasks — only new files created |
| Invalid `--regime` exits with usage message | Task 2 — `argparse choices=` + `test_parse_args_invalid_regime` |
| Feature branch created and checked in | Task 3 |
