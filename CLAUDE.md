# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Strategy Engine v5.1 — a production-ready Python quantitative trading strategy suggestion engine. The full implementation spec lives in `spec.txt`. Build every file from that spec before modifying anything.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in API keys
python scheduler.py
```

**Required environment variables** (in `.env`):
- `ANTHROPIC_API_KEY` ← used when `TESTING_MODE=False`
- `GEMINI_API_KEY` ← used when `TESTING_MODE=True` (default)
- `FINNHUB_API_KEY` ← active news provider (set `NEWS_PROVIDER="finnhub"` in `config.py`)
- `POLYGON_API_KEY` ← kept for future use; not active

## Running the System

The scheduler fires automatically at two times (US/Eastern):
- **8:05 AM** — `run_premarket_pipeline()`: full 9-gate screening + Polygon + LLM pipeline
- **9:45 AM** — `run_live_volume_check()`: RVOL confirmation gate

To verify all modules import cleanly after any change:
```bash
python -c "import config; import database; import regime_engine; import pipeline; import risk_auditor; import scheduler; print('all OK')"
```

## Architecture

Six-module system with a strict cost-minimization principle: **all free data gates must pass before any paid API call is made**.

| Module | Responsibility |
|---|---|
| `config.py` | All constants and thresholds — single source of truth |
| `scheduler.py` | Cron loop orchestrator; hardcoded `CANDIDATE_TICKERS` list for prototype |
| `regime_engine.py` | Free-only gates: macro regime, sector rotation, volatility/trend, earnings proximity, sector leadership |
| `database.py` | ChromaDB local persistence; vector building and similarity queries |
| `pipeline.py` | News gate: Finnhub news (active) / Polygon.io (inactive) → EASS calculation → Haiku summarization → ChromaDB query |
| `risk_auditor.py` | Sonnet structural audit → regime mutator → position sizing → strategy card output |

### Pipeline gate order (never reorder)
1. Macro regime (SPY vs 50 SMA) — `regime_engine`
2. Sector rotation filter (top 3 of 11 sector ETFs) — `regime_engine`
3. Volatility + trend gates (ATR14/price ≤ 10%, price > EMA21 and EMA200) — `regime_engine`
4. Earnings proximity exclusion (±5 calendar days) — `regime_engine`
5. Sector leadership dedup (one ticker per sub-industry by pre-market gap) — `regime_engine`
6. **NEWS GATE START**: Finnhub news fetch (active provider) — `pipeline`
7. Catalyst classification + EASS scoring — `pipeline`
8. ChromaDB historical similarity query — `pipeline` → `database`
9. LLM structural audit (Sonnet or Gemini depending on `TESTING_MODE`) — `risk_auditor`

## LLM Models

Two provider modes controlled by `TESTING_MODE` in `config.py`:

| Stage | Production (`TESTING_MODE=False`) | Testing (`TESTING_MODE=True`) | Purpose |
|---|---|---|---|
| Stage 3 | `claude-haiku-4-5-20251001` | `gemini-3.1-flash-lite` | Fast catalyst classification + summarization |
| Stage 4 | `claude-sonnet-4-6` | `gemini-3.1-flash-lite` | Structural risk audit |

LLM provider switching is fully encapsulated in `llm_client.py` — callers never import `anthropic` or `google.genai` directly.

## Critical Rules

- **Never** call the news API (Finnhub or Polygon) before Step 1.6 (all free gates must pass first).
- **Never** call any LLM before Step 1.6.
- ChromaDB collections are strictly partitioned by catalyst type — **never cross-query** between collections.
- If ChromaDB returns fewer than 40 similar samples, output `INSUFFICIENT_CONFIDENCE` and skip the trade.
- EASS uses 3 components only: `analyst_surprise` (weight 0.45), `whisper_surprise` (weight 0.00, FUTURE_STUB), `guidance_delta` (weight 0.55).
- In BEARISH regime: position size is halved (`BEARISH_REGIME_MUTATOR = 0.5`) and `MARKET_OPEN` entry is downgraded to `PULLBACK_LIMIT_21EMA`.
Add meaningful comment for every new functions added

## Data Conventions

- All ChromaDB collection names: `lowercase_underscore` catalyst type names (see `CATALYST_COLLECTIONS` in `config.py`).
- All vector dimensions: normalized 0.0–1.0.
- All returns stored as decimals (0.05 = 5%).
- Regime vectors are 8-dimensional; catalyst vectors are 4-dimensional; stored vectors are 12-dimensional (concatenated).

## Outputs

| File | Content |
|---|---|
| `final_strategy_blueprint.md` | Formatted Markdown strategy cards, one per qualified candidate |
| `active_monitoring_queue.json` | JSON log of all pipeline decisions at every gate |

## FUTURE_STUBs

Two components are intentionally left as stubs and documented in code:
- **Whisper surprise** (`pipeline.py` EASS Component 2): weight is 0.00 until a whisper feed is added.
- **Options component** (EASS Component 4): skipped entirely until Polygon options feed is activated.

Do not remove or implement these stubs without updating `config.EASS_WEIGHTS` and the corresponding ChromaDB vector dimensions simultaneously.

## Known Limitations

- **Finnhub news ordering**: Finnhub returns articles sorted by publication date (newest first), not by impact or relevance. The most breaking catalyst is rarely `articles[0]` — it is often buried in `articles[1+]`. The pipeline must scan all returned articles to find the highest-priority catalyst type, not just classify the first one. See `pipeline.py:_process_ticker` Step 3.
