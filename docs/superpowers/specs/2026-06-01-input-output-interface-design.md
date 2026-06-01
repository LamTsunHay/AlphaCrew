# Input/Output Interface Design
**Date:** 2026-06-01
**Feature:** CLI entry point + terminal output + Telegram stub

---

## Overview

Add a `run.py` CLI entry point for on-demand debugging and testing of the premarket pipeline. Output is rendered to the terminal (no file writes). A `notifier.py` module handles output delivery and includes a documented Telegram stub for future activation.

Zero changes to existing modules (`scheduler.py`, `pipeline.py`, `risk_auditor.py`, etc.).

---

## New Files

| File | Purpose |
|---|---|
| `run.py` | CLI entry point — parses args, orchestrates pipeline, calls notifier |
| `notifier.py` | Output delivery — terminal render now, Telegram stub for later |

---

## CLI Interface

```
python run.py [TICKERS...] [--regime BULLISH|BEARISH]
```

| Argument | Type | Default | Description |
|---|---|---|---|
| `TICKERS` | positional, zero or more | NASDAQ 100 list | Tickers to screen |
| `--regime` | optional, choices: BULLISH/BEARISH | None (live SPY check) | Override regime gate for testing |

### Examples

```bash
python run.py                          # Full NASDAQ 100, live regime from SPY
python run.py NVDA AMD AAPL            # Custom tickers, live regime
python run.py NVDA --regime BULLISH    # Custom ticker, forced regime override
python run.py --regime BEARISH         # Full NASDAQ 100, bearish regime forced
```

---

## Data Flow

```
argparse
  → ticker list (positional args OR NASDAQ_100 default)
  → regime (--regime override OR live regime_engine.get_market_regime())
  → regime_engine free gates (1.1–1.5)
  → pipeline.run_pipeline() paid gate (1.6–1.8)
  → risk_auditor.run_sonnet_audit() per candidate
  → notifier.print_to_terminal(strategy_cards, log_entries)
```

The regime override short-circuits `regime_engine.get_market_regime()` — all other gates run normally.

---

## `notifier.py` Structure

```python
def print_to_terminal(strategy_cards, log_entries):
    # Render strategy cards and gate summary to stdout

def send_telegram(strategy_cards, log_entries):
    # FUTURE_STUB: send strategy cards to Telegram channel via bot
    # Activate by: pip install python-telegram-bot, add TELEGRAM_BOT_TOKEN
    # and TELEGRAM_CHANNEL_ID to .env, then implement here.
    pass
```

The stub follows the same pattern as existing FUTURE_STUBs in `pipeline.py` (whisper surprise, options component).

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Invalid `--regime` value | `argparse` rejects with usage message, exit 1 |
| Zero tickers qualify | Print `NO QUALIFIED CANDIDATES` summary, exit 0 |
| Polygon/LLM API error | Existing pipeline error handling applies (ticker skipped, logged) |
| Missing API keys | Existing `.env` loading applies — same as scheduler |

---

## Function Comments

Each function in `run.py` and `notifier.py` gets a single-line docstring describing what it does (no multi-paragraph blocks). Existing modules are not modified.

---

## Out of Scope

- File output (`final_strategy_blueprint.md`, `active_monitoring_queue.json`) — not written by `run.py`
- Account size parameter — stays hardcoded at $100,000
- `scheduler.py` changes — none
- Telegram implementation — stub only, not wired up
