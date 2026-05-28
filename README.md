# Strategy Engine v5.1

A production-ready Python quantitative trading strategy suggestion engine that screens pre-market catalysts through a 9-gate pipeline and produces formatted trade setup cards.

## Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and fill in API keys
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY and POLYGON_API_KEY

# 4. Update earnings_calendar.csv with real upcoming dates
#    Format: ticker,next_earnings_date (YYYY-MM-DD)
```

## Running

```bash
# Start the production scheduler (fires at 8:05 AM and 9:45 AM EST)
python scheduler.py
```

## Outputs

| File | Description |
|---|---|
| `final_strategy_blueprint.md` | Formatted Markdown strategy cards for qualified setups |
| `active_monitoring_queue.json` | Full decision log of all pipeline gate outcomes |

## Database Population

The pipeline requires at least 40 similar historical samples per catalyst collection before it will emit a trade card (`INSUFFICIENT_CONFIDENCE` is returned otherwise).

### Option 1 — Synthetic seeder (for initial testing)

```bash
python seed_database.py                  # loads ~120 synthetic samples per collection
python seed_database.py --samples 200    # override samples per collection
```

### Option 2 — Real historical data importer (yfinance, no API key needed)

```bash
# Use or edit the provided template
cp sample_events.csv my_events.csv
# Edit my_events.csv with real historical catalyst events

python import_history.py --input my_events.csv
```

The importer fetches real price history around each event date from yfinance and computes genuine outcome metrics (day1/day3 returns, max adverse move, EMA alignment).

## FUTURE_STUBs

Two components are intentionally unimplemented and documented in code:

- **Whisper surprise** (`pipeline.py`, EASS Component 2): weight is `0.00` in `config.EASS_WEIGHTS` until a whisper feed is added.
- **Options component** (EASS Component 4): skipped entirely until Polygon options feed is activated.

Activating either requires updating `config.EASS_WEIGHTS` and potentially the ChromaDB vector dimensions.

## Architecture

See `CLAUDE.md` for the full architecture reference.
