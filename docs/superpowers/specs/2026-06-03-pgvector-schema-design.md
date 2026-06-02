# pgvector Schema Design

**Date:** 2026-06-03  
**Status:** Approved

## Context

The system currently uses ChromaDB with 15 separate collections (one per catalyst type) and a 12-dimensional embedding split into an 8-dim regime vector and a 4-dim catalyst vector. The catalyst vector dimensions change meaning by catalyst type (e.g., dim 1 is `eps_surprise_pct` for earnings but `was_expected` for FDA), making the vector space incoherent across types and preventing cross-catalyst generalization.

The migration to PostgreSQL pgvector provides:
- Full SQL filtering on any metadata column
- Single data source (no separate `./chroma_db` directory)
- A redesigned universal catalyst vector where dimensions mean the same thing across all 15 catalyst types, enabling cross-catalyst similarity queries

## Architecture

One `setups` table replaces 15 ChromaDB collections. `catalyst_type` becomes a SQL column — use `WHERE catalyst_type = $1` for per-type queries, or omit the filter to query across all types. The 12-dimensional embedding is stored as a `vector(12)` column. News text and outcome metrics are typed SQL columns alongside the vector.

The public interface of `database.py` is preserved: `initialize_database()`, `store_setup()`, `query_similar_setups()`, `calculate_outcome_profile()`. Callers (`pipeline.py`, `scheduler.py`, `seed_database.py`, `import_history.py`) require no changes.

## Schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS setups (
    id                SERIAL PRIMARY KEY,
    doc_id            TEXT UNIQUE NOT NULL,
    ticker            TEXT NOT NULL,
    trade_date        TEXT NOT NULL,
    catalyst_type     TEXT NOT NULL,
    embedding         vector(12) NOT NULL,

    -- Reference text: nullable for historical seed data
    news_header       TEXT,
    news_summary      TEXT,

    -- Outcome metrics
    day1_return       DOUBLE PRECISION NOT NULL,
    day3_return       DOUBLE PRECISION NOT NULL,
    max_adverse_move  DOUBLE PRECISION NOT NULL,
    held_above_21ema  DOUBLE PRECISION NOT NULL,
    regime_at_exit    TEXT NOT NULL,
    exit_trigger      TEXT NOT NULL,

    created_at        TIMESTAMP DEFAULT NOW()
);

-- Per-type queries
CREATE INDEX IF NOT EXISTS setups_catalyst_type_idx ON setups (catalyst_type);

-- Approximate nearest-neighbor: L2 distance matches ChromaDB default
CREATE INDEX IF NOT EXISTS setups_embedding_hnsw_idx
    ON setups USING hnsw (embedding vector_l2_ops);
```

## Embedding Structure

**Total: 12 dimensions = regime(8) + catalyst(4)**

### Regime vector — dims 0–7 (unchanged from current system)

| Index | Field | Normalization |
|-------|-------|--------------|
| 0 | `spy_vs_50sma` | pct above 50SMA → [0, 1] |
| 1 | `sector_5d_vs_spy` | 5d relative return → [0, 1] |
| 2 | `vix_inv` | VIX inverted (lower VIX = higher score) → [0, 1] |
| 3 | `sector_rank_inv` | Rank inverted (rank 1 = 1.0) → [0, 1] |
| 4 | `pct_above_200ema` | % above 200 EMA → [0, 1] |
| 5 | `premarket_gap` | Pre-market gap % → [0, 1] |
| 6 | `rvol_945` | Relative volume at 9:45 → [0, 1] |
| 7 | `quarter_norm` | Fiscal quarter (1–4) → [0, 1] |

### Universal catalyst vector — dims 8–11

All four dimensions carry the same semantic meaning regardless of catalyst type. This allows similarity queries across catalyst types without incoherence.

| Index | Field | Meaning |
|-------|-------|---------|
| 8 | `surprise_magnitude` | How much did this event shift the market's forward cash flow model, normalized [0, 1] |
| 9 | `surprise_direction` | 1.0 = positive surprise, 0.0 = negative surprise |
| 10 | `forward_signal` | What does this event imply about future cash flows beyond the immediate announcement, normalized [0, 1] |
| 11 | `business_impact` | Size of the event relative to total company valuation, normalized [0, 1] |

## Per-Catalyst Mapping

### `surprise_magnitude` — priority: signals that directly update valuation model

Guidance delta outweighs revenue which outweighs EPS, because guidance directly updates the forward DCF inputs; EPS is last (susceptible to buybacks and one-time charges).

| Catalyst group | Formula |
|---------------|---------|
| Earnings (beat/miss) | `normalize(0.50×\|guidance_delta_pct\| + 0.35×\|revenue_surprise_pct\| + 0.15×\|eps_surprise_pct\|)` |
| Guidance (raise/partial/cut) | `normalize(\|eps_guidance_delta_pct\| × 0.6 + \|revenue_guidance_delta_pct\| × 0.4)` |
| FDA (approval/rejection) | `(1 − was_expected) × normalize(1 / market_cap_billions, [0.02, 10])` — surprise × relative TAM impact |
| M&A target | `normalize(premium_pct, [0, 0.60])` — premium IS the valuation change |
| M&A acquirer | `normalize(deal_size_billions / acquirer_market_cap, [0, 1])` |
| Buyback | `normalize(buyback_pct_of_float, [0, 0.20])` |
| Contract (gov/commercial) | `normalize(contract_value_millions / revenue_proxy, [0, 5])` |

### `forward_signal` — future cash flow implication

| Catalyst group | Formula |
|---------------|---------|
| Earnings | `normalize(guidance_delta_pct, [-0.20, 0.20])` |
| Guidance | `normalize(eps_guidance_delta_pct, [-0.30, 0.30])` |
| FDA | `normalize(pipeline_depth, [1, 20])` — more drugs in pipeline = stronger follow-through |
| M&A target | `cash_deal × 0.7 + (1 − hostile_bid) × 0.3` — certainty of capturing the premium |
| M&A acquirer | `normalize(1 − deal_size / acquirer_market_cap, [0, 1])` — lower dilution = better |
| Buyback | `normalize(buyback_pct_of_float, [0, 0.20])` |
| Contract | `multi_year × 0.5 + sole_source × 0.5` — recurring locked-in revenue |

### `business_impact` — event size relative to company valuation

| Catalyst group | Formula |
|---------------|---------|
| Earnings | `normalize(\|eps_surprise_pct\| × inv_market_cap_tier)` — same EPS surprise hits small cap harder |
| Guidance | `normalize(\|eps_guidance_delta_pct\| × inv_market_cap_tier)` |
| FDA | `normalize(1 / market_cap_billions, [0.02, 10])` — smaller company = larger relative impact |
| M&A target | `normalize(deal_size_billions, [0.1, 100])` |
| M&A acquirer | `normalize(deal_size_billions / acquirer_market_cap, [0, 1])` |
| Buyback | `normalize(buyback_pct_of_float, [0, 0.20])` |
| Contract | `normalize(margin_impact_pct, [0, 0.05])` |

### `surprise_direction`

| Value | Meaning |
|-------|---------|
| `1.0` | Positive: earnings beat, FDA approval, M&A target, contract win, buyback |
| `0.0` | Negative: earnings miss, FDA rejection, guidance cut |

## Query Patterns

**Per-type query (current behavior):**
```sql
SELECT ticker, trade_date, catalyst_type, news_header, news_summary,
       day1_return, day3_return, max_adverse_move,
       held_above_21ema, regime_at_exit, exit_trigger,
       embedding <-> %s::vector AS distance
FROM setups
WHERE catalyst_type = %s
ORDER BY distance
LIMIT 80
```
Then filter in Python: keep rows where `distance < VECTOR_DB_MAX_DISTANCE (0.35)`.

**Cross-type query (new capability — omit WHERE clause):**
```sql
SELECT ticker, trade_date, catalyst_type, news_header, news_summary,
       day1_return, day3_return, max_adverse_move,
       held_above_21ema, regime_at_exit, exit_trigger,
       embedding <-> %s::vector AS distance
FROM setups
ORDER BY distance
LIMIT 80
```

## Interface

All existing public functions are preserved. `store_setup()` gains two optional kwargs (backward compatible — existing callers need no changes):

```python
def initialize_database(dsn: str = None) -> connection: ...

def store_setup(
    conn, catalyst_type: str,
    regime_data: dict, catalyst_data: dict, outcome: dict,
    news_header: str = None,      # new optional
    news_summary: str = None,     # new optional
) -> None: ...

def query_similar_setups(
    conn, catalyst_type: str,
    regime_data: dict, catalyst_data: dict,
) -> dict: ...

def calculate_outcome_profile(matches: list) -> dict: ...  # unchanged
```

`catalyst_data` still carries raw type-specific fields (e.g., `eps_surprise_pct`, `guidance_delta_pct`). `_build_catalyst_vector()` applies the per-catalyst mapping internally to produce the 4 universal dims. Callers are unaware of the mapping.

## Files Changed

| File | Change |
|------|--------|
| `migrations/001_create_setups.sql` | New: DDL for schema + indexes |
| `config.py` | Add `POSTGRES_DSN`, remove `VECTOR_DB_PATH` |
| `.env.example` | Add `DATABASE_URL` |
| `requirements.txt` | Swap `chromadb` → `psycopg2-binary` + `pgvector` |
| `database.py` | Full rewrite: same public interface, psycopg2 internals, new vector builders |
| `tests/test_database.py` | Rewrite: tests against real Postgres via `TEST_DATABASE_URL` |

`pipeline.py`, `scheduler.py`, `seed_database.py`, `import_history.py` — no changes required (optional kwargs are backward compatible).

## Data Migration

- `seed_database.py` regenerates synthetic data — re-run after migration, no data porting needed.
- `import_history.py` re-imports from CSV + yfinance — re-run against new schema.
- The `./chroma_db` directory can be deleted once the new database is seeded and verified.
- `news_header` and `news_summary` are nullable — historical records without news text are valid.

## Testing

```bash
# Apply schema
psql "$DATABASE_URL" -f migrations/001_create_setups.sql

# Run tests against isolated test database
TEST_DATABASE_URL="postgresql://localhost/alphacrew_test" pytest tests/test_database.py -v

# Smoke test all module imports
python -c "import config; import database; import regime_engine; import pipeline; import risk_auditor; import scheduler; print('all OK')"

# Verify seeded data
psql "$DATABASE_URL" -c "SELECT catalyst_type, COUNT(*) FROM setups GROUP BY catalyst_type ORDER BY catalyst_type;"

# Dry pipeline query (no Polygon calls) — raw catalyst fields, same as pipeline.py passes
python -c "
import database
conn = database.initialize_database()
regime = {'spy_pct_above_50sma': 0.6, 'sector_5d_vs_spy': 0.02, 'vix_level': 18.0,
          'sector_rank': 2, 'pct_above_200ema': 0.05, 'premarket_gap_pct': 0.03,
          'rvol_945': 1.8, 'quarter': 2}
catalyst = {'eps_surprise_pct': 0.12, 'revenue_surprise_pct': 0.06,
            'guidance_delta_pct': 0.08, 'analyst_revision_count': 3}
result = database.query_similar_setups(conn, 'earnings_beat_large', regime, catalyst)
print('confidence:', result['confidence'], '| sample_size:', result['sample_size'])
"
```
