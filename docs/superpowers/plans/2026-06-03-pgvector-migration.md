# ChromaDB → PostgreSQL pgvector Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace ChromaDB with PostgreSQL pgvector, rewriting `database.py` with a universal 4-dim catalyst vector (surprise_magnitude, surprise_direction, forward_signal, business_impact) while keeping the public interface unchanged so no callers need editing.

**Architecture:** One `setups` table replaces 15 ChromaDB collections. `catalyst_type` becomes a SQL `WHERE` filter. The 12-dim embedding is stored as a `vector(12)` column. `initialize_database()` returns a psycopg2 connection; all other public functions accept that connection as their first arg — callers are unaware of the change. The universal `build_catalyst_vector()` maps raw type-specific fields to 4 valuation-priority dims, so `pipeline.py`, `seed_database.py`, and `import_history.py` need zero changes.

**Tech Stack:** psycopg2-binary, pgvector (Python package), PostgreSQL 14+, pytest

**Spec:** `docs/superpowers/specs/2026-06-03-pgvector-schema-design.md`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `migrations/001_create_setups.sql` | Create | Schema DDL — idempotent, run at startup |
| `config.py` | Modify | Replace `VECTOR_DB_PATH` with `POSTGRES_DSN` |
| `.env.example` | Modify | Add `DATABASE_URL` |
| `requirements.txt` | Modify | Swap `chromadb` → `psycopg2-binary` + `pgvector` |
| `database.py` | Rewrite | Same public interface, psycopg2 internals + universal vector |
| `tests/test_database.py` | Modify | Replace ChromaDB round-trip test; pure-Python tests unchanged |

`pipeline.py`, `scheduler.py`, `seed_database.py`, `import_history.py` — **no changes**.

---

## Task 1: Update Config, Requirements, and .env

**Files:**
- Modify: `config.py`
- Modify: `requirements.txt`
- Modify: `.env.example`

- [ ] **Step 1: Write a failing test confirming POSTGRES_DSN is expected**

Append to `tests/test_database.py` (keep all existing tests):

```python
# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------

def test_postgres_dsn_in_config():
    """config.POSTGRES_DSN must exist and be non-empty."""
    import config as _config
    assert hasattr(_config, "POSTGRES_DSN"), "POSTGRES_DSN missing from config"
    assert _config.POSTGRES_DSN, "POSTGRES_DSN is empty — set DATABASE_URL in .env"

def test_vector_db_path_removed():
    """VECTOR_DB_PATH must be removed — chroma_db directory is no longer used."""
    import config as _config
    assert not hasattr(_config, "VECTOR_DB_PATH"), "VECTOR_DB_PATH must be removed from config"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /home/user/AlphaCrew
pytest tests/test_database.py::test_postgres_dsn_in_config tests/test_database.py::test_vector_db_path_removed -v
```

Expected: FAIL — `AssertionError: POSTGRES_DSN missing from config`

- [ ] **Step 3: Update config.py**

In `config.py`, find the `# ChromaDB` block (lines 61–84) and replace it:

```python
# PostgreSQL pgvector
POSTGRES_DSN = ""           # set via DATABASE_URL in .env; loaded in database.py
VECTOR_DB_MIN_SAMPLES = 40
VECTOR_DB_MAX_DISTANCE = 0.35

# Valid catalyst type names — used for validation only (no longer one-per-collection)
CATALYST_COLLECTIONS = [
    "earnings_beat_large",
    "earnings_beat_small",
    "earnings_miss",
    "revenue_beat_only",
    "guidance_raise_full",
    "guidance_raise_partial",
    "guidance_cut",
    "fda_approval_nda",
    "fda_approval_fast_track",
    "fda_rejection",
    "ma_acquirer",
    "ma_target",
    "buyback_initiation",
    "government_contract",
    "commercial_contract",
]
```

Note: `POSTGRES_DSN` is intentionally empty here — `database.initialize_database()` reads `DATABASE_URL` from the environment directly, so the value is set at runtime rather than at import time.

- [ ] **Step 4: Update requirements.txt**

```
anthropic
yfinance
pandas
numpy
psycopg2-binary
pgvector
aiohttp
schedule
python-dotenv
pytz
openai
```

- [ ] **Step 5: Update .env.example**

Add one line after `GROQ_API_KEY`:

```
DATABASE_URL=postgresql://user:password@localhost:5432/alphacrew
```

- [ ] **Step 6: Install new dependencies**

```bash
pip install psycopg2-binary pgvector
```

Expected: installs without errors

- [ ] **Step 7: Run tests to confirm they pass**

```bash
pytest tests/test_database.py::test_postgres_dsn_in_config tests/test_database.py::test_vector_db_path_removed -v
```

Expected: PASS, PASS

- [ ] **Step 8: Confirm full import still works**

```bash
python -c "import config; print('POSTGRES_DSN attr:', hasattr(config, 'POSTGRES_DSN')); print('VECTOR_DB_PATH removed:', not hasattr(config, 'VECTOR_DB_PATH'))"
```

Expected: both True

- [ ] **Step 9: Commit**

```bash
git add config.py requirements.txt .env.example tests/test_database.py
git commit -m "feat: replace VECTOR_DB_PATH with POSTGRES_DSN, swap chromadb dependency"
```

---

## Task 2: SQL Migration File

**Files:**
- Create: `migrations/001_create_setups.sql`

- [ ] **Step 1: Create the migrations directory and SQL file**

```bash
mkdir -p /home/user/AlphaCrew/migrations
```

Write `migrations/001_create_setups.sql`:

```sql
-- migrations/001_create_setups.sql
-- Idempotent: safe to run multiple times (IF NOT EXISTS throughout)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS setups (
    id                SERIAL PRIMARY KEY,
    doc_id            TEXT UNIQUE NOT NULL,
    ticker            TEXT NOT NULL,
    trade_date        TEXT NOT NULL,
    catalyst_type     TEXT NOT NULL,
    embedding         vector(12) NOT NULL,

    -- Reference text — nullable for historical seed data without news
    news_header       TEXT,
    news_summary      TEXT,

    -- Outcome metrics used by calculate_outcome_profile()
    day1_return       DOUBLE PRECISION NOT NULL,
    day3_return       DOUBLE PRECISION NOT NULL,
    max_adverse_move  DOUBLE PRECISION NOT NULL,
    held_above_21ema  DOUBLE PRECISION NOT NULL,
    regime_at_exit    TEXT NOT NULL,
    exit_trigger      TEXT NOT NULL,

    created_at        TIMESTAMP DEFAULT NOW()
);

-- Fast WHERE catalyst_type = $1 filter (replaces 15 separate collections)
CREATE INDEX IF NOT EXISTS setups_catalyst_type_idx
    ON setups (catalyst_type);

-- HNSW approximate nearest-neighbor on the 12-dim embedding
-- Uses L2 distance (<->) to match ChromaDB's default metric
CREATE INDEX IF NOT EXISTS setups_embedding_hnsw_idx
    ON setups USING hnsw (embedding vector_l2_ops);
```

- [ ] **Step 2: Apply migration to a test database to verify SQL**

```bash
createdb alphacrew_test 2>/dev/null || true
psql alphacrew_test -f migrations/001_create_setups.sql
```

Expected output (in order):
```
CREATE EXTENSION
CREATE TABLE
CREATE INDEX
CREATE INDEX
```

- [ ] **Step 3: Verify the table structure**

```bash
psql alphacrew_test -c "\d setups"
```

Expected: 14 columns including `embedding vector(12)`

- [ ] **Step 4: Commit**

```bash
git add migrations/001_create_setups.sql
git commit -m "feat: add pgvector schema migration for setups table"
```

---

## Task 3: Rewrite initialize_database()

**Files:**
- Modify: `database.py`
- Modify: `tests/test_database.py`

- [ ] **Step 1: Write failing tests for initialize_database()**

Append to `tests/test_database.py`:

```python
import os
import psycopg2

# ---------------------------------------------------------------------------
# Helpers shared by Tasks 3-6
# ---------------------------------------------------------------------------

TEST_DSN = os.getenv("TEST_DATABASE_URL", "")

def _get_conn():
    """Return an initialized test connection, or skip the test if no DSN set."""
    if not TEST_DSN:
        pytest.skip("TEST_DATABASE_URL not set — skipping PostgreSQL tests")
    return database.initialize_database(dsn=TEST_DSN)

def _drop_setups(conn):
    """Tear down the setups table between test runs."""
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS setups")
        cur.execute("DROP EXTENSION IF EXISTS vector CASCADE")
    conn.commit()

# ---------------------------------------------------------------------------
# initialize_database()
# ---------------------------------------------------------------------------

def test_initialize_returns_open_connection():
    """initialize_database() must return an open psycopg2 connection."""
    conn = _get_conn()
    try:
        assert not conn.closed
    finally:
        _drop_setups(conn)
        conn.close()

def test_initialize_creates_setups_table():
    """initialize_database() must create the setups table."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.setups')")
            result = cur.fetchone()[0]
        assert result == "setups"
    finally:
        _drop_setups(conn)
        conn.close()

def test_initialize_is_idempotent():
    """Calling initialize_database() twice must not raise."""
    conn = _get_conn()
    try:
        conn2 = database.initialize_database(dsn=TEST_DSN)
        conn2.close()
    finally:
        _drop_setups(conn)
        conn.close()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
TEST_DATABASE_URL="postgresql://localhost/alphacrew_test" pytest tests/test_database.py::test_initialize_returns_open_connection -v
```

Expected: FAIL — `TypeError` (wrong signature) or import error from chromadb

- [ ] **Step 3: Rewrite the top of database.py**

Replace the entire file from line 1 through the end of `initialize_database()` with:

```python
"""PostgreSQL pgvector setup for Strategy Engine v5.1.

Public interface (unchanged from ChromaDB version):
  initialize_database(dsn=None) -> psycopg2 connection
  store_setup(conn, catalyst_type, regime_data, catalyst_data, outcome,
              news_header=None, news_summary=None)
  query_similar_setups(conn, catalyst_type, regime_data, catalyst_data) -> dict
  calculate_outcome_profile(matches) -> dict

The connection returned by initialize_database() is passed as the first
argument to store_setup() and query_similar_setups() — callers treat it
as an opaque handle, identical to how the old ChromaDB client was used.
"""

import os
import datetime
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
import numpy as np
import config

# SQL DDL embedded here so initialize_database() is self-contained.
# All statements use IF NOT EXISTS — safe to run on every startup.
_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS setups (
    id                SERIAL PRIMARY KEY,
    doc_id            TEXT UNIQUE NOT NULL,
    ticker            TEXT NOT NULL,
    trade_date        TEXT NOT NULL,
    catalyst_type     TEXT NOT NULL,
    embedding         vector(12) NOT NULL,
    news_header       TEXT,
    news_summary      TEXT,
    day1_return       DOUBLE PRECISION NOT NULL,
    day3_return       DOUBLE PRECISION NOT NULL,
    max_adverse_move  DOUBLE PRECISION NOT NULL,
    held_above_21ema  DOUBLE PRECISION NOT NULL,
    regime_at_exit    TEXT NOT NULL,
    exit_trigger      TEXT NOT NULL,
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS setups_catalyst_type_idx
    ON setups (catalyst_type);

CREATE INDEX IF NOT EXISTS setups_embedding_hnsw_idx
    ON setups USING hnsw (embedding vector_l2_ops);
"""


def normalize(value, min_val, max_val):
    """Clip-normalize value to [0.0, 1.0]."""
    if max_val == min_val:
        return 0.0
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


def initialize_database(dsn: str = None):
    """Connect to PostgreSQL, apply schema, and return an open connection.

    Reads DATABASE_URL from the environment when dsn is not supplied.
    Registers the pgvector type so Python lists are accepted as vector(12).
    """
    dsn = dsn or os.getenv("DATABASE_URL", config.POSTGRES_DSN)
    conn = psycopg2.connect(dsn)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    conn.commit()
    print("[DB] PostgreSQL connected; setups table ready")
    return conn
```

Leave the rest of database.py (build_regime_vector, build_catalyst_vector, store_setup, query_similar_setups, calculate_outcome_profile) untouched for now — they will be updated in subsequent tasks.

- [ ] **Step 4: Run tests to confirm they pass**

```bash
TEST_DATABASE_URL="postgresql://localhost/alphacrew_test" \
  pytest tests/test_database.py::test_initialize_returns_open_connection \
         tests/test_database.py::test_initialize_creates_setups_table \
         tests/test_database.py::test_initialize_is_idempotent -v
```

Expected: PASS, PASS, PASS

- [ ] **Step 5: Commit**

```bash
git add database.py tests/test_database.py
git commit -m "feat: implement initialize_database() with psycopg2 + pgvector"
```

---

## Task 4: Rewrite build_catalyst_vector() — Universal Dims

**Files:**
- Modify: `database.py`

The new function maps raw type-specific fields to 4 universal dims:
- `[0] surprise_magnitude` — how much the event shifts the forward cash flow model
- `[1] surprise_direction` — 1.0 positive / 0.0 negative
- `[2] forward_signal` — future cash flow implication beyond the announcement
- `[3] business_impact` — event size relative to company valuation

`build_regime_vector()` is unchanged — do not touch it.

- [ ] **Step 1: Run existing catalyst vector tests to establish baseline**

```bash
pytest tests/test_database.py::test_catalyst_vector_length_earnings \
       tests/test_database.py::test_catalyst_vector_length_fda \
       tests/test_database.py::test_catalyst_vector_fallback \
       tests/test_database.py::test_catalyst_vector_range -v
```

Expected: All PASS (the length and range contracts are preserved by the new impl)

- [ ] **Step 2: Replace build_catalyst_vector() in database.py**

Find and replace the entire `build_catalyst_vector()` function body (keep the name and `(catalyst_data, catalyst_type)` signature):

```python
def build_catalyst_vector(catalyst_data: dict, catalyst_type: str) -> list:
    """Build a 4-dimensional universal catalyst vector.

    Dims are catalyst-type-agnostic so vectors are comparable across types:
      [0] surprise_magnitude  — weighted deviation from market expectation [0, 1]
      [1] surprise_direction  — 1.0 = positive surprise, 0.0 = negative [0, 1]
      [2] forward_signal      — implication for future cash flows [0, 1]
      [3] business_impact     — event size relative to company valuation [0, 1]

    Valuation priority for earnings: guidance > revenue > EPS, because guidance
    directly updates the forward DCF model while EPS is susceptible to buybacks
    and one-time charges.
    """
    ct = catalyst_type.lower()

    # --- EARNINGS / REVENUE ---
    if ct.startswith("earnings") or ct.startswith("revenue"):
        eps  = catalyst_data.get("eps_surprise_pct", 0.0)
        rev  = catalyst_data.get("revenue_surprise_pct", 0.0)
        guid = catalyst_data.get("guidance_delta_pct", 0.0)
        # Weighted magnitude: guidance 50%, revenue 35%, EPS 15%
        # Max raw ≈ 0.50×0.20 + 0.35×0.30 + 0.15×0.50 = 0.28
        raw = 0.50 * abs(guid) + 0.35 * abs(rev) + 0.15 * abs(eps)
        return [
            normalize(raw, 0, 0.28),
            0.0 if ct == "earnings_miss" else 1.0,
            normalize(guid, -0.20, 0.20),
            normalize(abs(guid) + abs(eps), 0, 0.70),
        ]

    # --- GUIDANCE ---
    if ct.startswith("guidance"):
        eps_g = catalyst_data.get("eps_guidance_delta_pct", 0.0)
        rev_g = catalyst_data.get("revenue_guidance_delta_pct", 0.0)
        # Max raw ≈ 0.60×0.30 + 0.40×0.20 = 0.26
        raw = 0.60 * abs(eps_g) + 0.40 * abs(rev_g)
        return [
            normalize(raw, 0, 0.26),
            0.0 if ct == "guidance_cut" else 1.0,
            normalize(eps_g, -0.30, 0.30),
            normalize(abs(eps_g) + abs(rev_g) * 0.5, 0, 0.45),
        ]

    # --- FDA ---
    if ct.startswith("fda"):
        was_expected  = float(bool(catalyst_data.get("was_expected", False)))
        market_cap    = catalyst_data.get("market_cap_billions", 10.0)
        pipeline      = catalyst_data.get("pipeline_depth", 1)
        # Smaller company + unexpected event = larger surprise_magnitude
        impact_factor = 1.0 - normalize(market_cap, 0.1, 50)
        return [
            (1.0 - was_expected) * 0.6 + impact_factor * 0.4,
            0.0 if ct == "fda_rejection" else 1.0,
            normalize(pipeline, 1, 20),
            impact_factor,
        ]

    # --- M&A TARGET ---
    if ct == "ma_target":
        premium   = catalyst_data.get("premium_pct", 0.0)
        cash_deal = float(bool(catalyst_data.get("cash_deal", False)))
        hostile   = float(bool(catalyst_data.get("hostile_bid", False)))
        # Premium IS the valuation change for the target
        mag = normalize(premium, 0, 0.60)
        return [
            mag,
            1.0,                                        # target is always positive
            cash_deal * 0.7 + (1.0 - hostile) * 0.3,   # certainty of capturing premium
            mag,                                        # business_impact == premium
        ]

    # --- M&A ACQUIRER ---
    if ct == "ma_acquirer":
        deal_size = catalyst_data.get("deal_size_billions", 1.0)
        cash_deal = float(bool(catalyst_data.get("cash_deal", False)))
        hostile   = float(bool(catalyst_data.get("hostile_bid", False)))
        mag = normalize(deal_size, 0.1, 100)
        return [
            mag,
            0.5,                                    # acquirer impact is mixed
            cash_deal * 0.5 + (1.0 - hostile) * 0.5,
            mag,
        ]

    # --- BUYBACK ---
    if ct.startswith("buyback"):
        deal_size = catalyst_data.get("deal_size_billions", 1.0)
        mag = normalize(deal_size, 0.1, 50)
        return [mag, 1.0, mag, mag]

    # --- GOVERNMENT / COMMERCIAL CONTRACT ---
    if ct.startswith("government") or ct.startswith("commercial"):
        value      = catalyst_data.get("contract_value_millions", 10.0)
        multi_year = float(bool(catalyst_data.get("multi_year", False)))
        sole_src   = float(bool(catalyst_data.get("sole_source", False)))
        margin     = catalyst_data.get("margin_impact_pct", 0.0)
        return [
            normalize(value, 1, 5000),
            1.0,                                  # contract win is always positive
            multi_year * 0.5 + sole_src * 0.5,   # recurring / locked revenue
            normalize(margin, 0, 0.05),
        ]

    # Fallback for unknown catalyst types
    return [0.5, 0.5, 0.5, 0.5]
```

- [ ] **Step 3: Run existing catalyst tests to confirm they still pass**

```bash
pytest tests/test_database.py::test_catalyst_vector_length_earnings \
       tests/test_database.py::test_catalyst_vector_length_fda \
       tests/test_database.py::test_catalyst_vector_fallback \
       tests/test_database.py::test_catalyst_vector_range -v
```

Expected: All PASS (length=4, range [0,1], fallback=[0.5,0.5,0.5,0.5] — all preserved)

- [ ] **Step 4: Add a semantic test for the guidance priority**

Append to `tests/test_database.py`:

```python
def test_catalyst_vector_guidance_outweighs_eps():
    """For earnings, large guidance_delta must produce higher surprise_magnitude than large eps_surprise."""
    high_guidance = database.build_catalyst_vector(
        {"eps_surprise_pct": 0.0, "revenue_surprise_pct": 0.0, "guidance_delta_pct": 0.15},
        "earnings_beat_large",
    )
    high_eps = database.build_catalyst_vector(
        {"eps_surprise_pct": 0.40, "revenue_surprise_pct": 0.0, "guidance_delta_pct": 0.0},
        "earnings_beat_large",
    )
    # dim[0] = surprise_magnitude; guidance weight=0.50, eps weight=0.15
    assert high_guidance[0] > high_eps[0], (
        f"guidance-driven magnitude {high_guidance[0]:.3f} should exceed "
        f"eps-driven {high_eps[0]:.3f}"
    )

def test_catalyst_vector_direction_earnings_miss():
    """earnings_miss must produce surprise_direction = 0.0."""
    vec = database.build_catalyst_vector(
        {"eps_surprise_pct": -0.10, "revenue_surprise_pct": -0.05, "guidance_delta_pct": -0.03},
        "earnings_miss",
    )
    assert vec[1] == 0.0, f"surprise_direction should be 0.0 for miss, got {vec[1]}"

def test_catalyst_vector_fda_small_cap_surprise():
    """Unexpected FDA approval for small-cap must have higher surprise_magnitude than expected large-cap."""
    small_surprise = database.build_catalyst_vector(
        {"was_expected": False, "market_cap_billions": 0.5, "pipeline_depth": 3},
        "fda_approval_nda",
    )
    large_expected = database.build_catalyst_vector(
        {"was_expected": True, "market_cap_billions": 40.0, "pipeline_depth": 3},
        "fda_approval_nda",
    )
    assert small_surprise[0] > large_expected[0]
```

- [ ] **Step 5: Run new semantic tests**

```bash
pytest tests/test_database.py::test_catalyst_vector_guidance_outweighs_eps \
       tests/test_database.py::test_catalyst_vector_direction_earnings_miss \
       tests/test_database.py::test_catalyst_vector_fda_small_cap_surprise -v
```

Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add database.py tests/test_database.py
git commit -m "feat: rewrite build_catalyst_vector() with universal valuation-priority dims"
```

---

## Task 5: Rewrite store_setup()

**Files:**
- Modify: `database.py`
- Modify: `tests/test_database.py`

- [ ] **Step 1: Write failing tests for store_setup()**

Append to `tests/test_database.py`:

```python
# ---------------------------------------------------------------------------
# Shared fixtures for Tasks 5 and 6
# ---------------------------------------------------------------------------

SAMPLE_REGIME = {
    "spy_pct_above_50sma": 0.03, "sector_5d_vs_spy": 0.01, "vix_level": 16.0,
    "sector_rank": 1, "pct_above_200ema": 0.08, "premarket_gap_pct": 0.05,
    "rvol_945": 6.0, "quarter": 2,
}
SAMPLE_CATALYST = {
    "eps_surprise_pct": 0.12, "revenue_surprise_pct": 0.08,
    "guidance_delta_pct": 0.05, "analyst_revision_count": 7,
}
SAMPLE_OUTCOME = {
    "ticker": "AAPL",
    "date": "2024-01-15",
    "day1_return": 0.04,
    "day3_return": 0.07,
    "max_adverse_move": -0.02,
    "held_above_21ema": True,
    "regime_at_exit": "BULLISH",
    "exit_trigger": "TARGET",
}

# ---------------------------------------------------------------------------
# store_setup()
# ---------------------------------------------------------------------------

def test_store_setup_inserts_row():
    """store_setup() must insert exactly one row into setups."""
    conn = _get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM setups")
            before = cur.fetchone()[0]

        database.store_setup(conn, "earnings_beat_large", SAMPLE_REGIME, SAMPLE_CATALYST, SAMPLE_OUTCOME)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM setups")
            after = cur.fetchone()[0]
        assert after == before + 1
    finally:
        _drop_setups(conn)
        conn.close()

def test_store_setup_idempotent():
    """Calling store_setup() twice with identical data must not raise or duplicate."""
    conn = _get_conn()
    try:
        database.store_setup(conn, "earnings_beat_large", SAMPLE_REGIME, SAMPLE_CATALYST, SAMPLE_OUTCOME)
        before_count = None
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM setups")
            before_count = cur.fetchone()[0]

        database.store_setup(conn, "earnings_beat_large", SAMPLE_REGIME, SAMPLE_CATALYST, SAMPLE_OUTCOME)

        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM setups")
            after = cur.fetchone()[0]
        assert after == before_count
    finally:
        _drop_setups(conn)
        conn.close()

def test_store_setup_persists_news_fields():
    """news_header and news_summary must be stored when provided."""
    conn = _get_conn()
    try:
        database.store_setup(
            conn, "earnings_beat_large", SAMPLE_REGIME, SAMPLE_CATALYST, SAMPLE_OUTCOME,
            news_header="AAPL beats Q1 earnings by 12%",
            news_summary="Apple reported strong Q1 results. Revenue and EPS both exceeded consensus. Management raised full-year guidance.",
        )
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT news_header, news_summary FROM setups WHERE ticker = 'AAPL'")
            row = cur.fetchone()
        assert row["news_header"] == "AAPL beats Q1 earnings by 12%"
        assert "Apple" in row["news_summary"]
    finally:
        _drop_setups(conn)
        conn.close()

def test_store_setup_news_fields_nullable():
    """store_setup() without news args must succeed (fields are nullable)."""
    conn = _get_conn()
    try:
        database.store_setup(conn, "earnings_beat_large", SAMPLE_REGIME, SAMPLE_CATALYST, SAMPLE_OUTCOME)
        with conn.cursor() as cur:
            cur.execute("SELECT news_header FROM setups WHERE ticker = 'AAPL'")
            row = cur.fetchone()
        assert row[0] is None
    finally:
        _drop_setups(conn)
        conn.close()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
TEST_DATABASE_URL="postgresql://localhost/alphacrew_test" \
  pytest tests/test_database.py::test_store_setup_inserts_row -v
```

Expected: FAIL — `TypeError: store_setup() takes 5 positional arguments` (old ChromaDB signature)

- [ ] **Step 3: Replace store_setup() in database.py**

```python
def store_setup(
    conn,
    catalyst_type: str,
    regime_data: dict,
    catalyst_data: dict,
    outcome: dict,
    news_header: str = None,
    news_summary: str = None,
):
    """Build a 12-dim vector and upsert one row into setups.

    ticker and date are read from outcome (same convention as the old ChromaDB version).
    ON CONFLICT DO NOTHING makes repeated calls with the same data safe.
    news_header and news_summary are optional — NULL for historical seed data.
    """
    regime_vec = build_regime_vector(regime_data)
    catalyst_vec = build_catalyst_vector(catalyst_data, catalyst_type)
    full_vector = regime_vec + catalyst_vec

    ticker   = str(outcome.get("ticker") or "UNKNOWN")
    date_str = str(outcome.get("date") or datetime.date.today().isoformat())
    doc_id   = f"{ticker}_{date_str}_{hash(str(full_vector)) % 1_000_000:06d}"

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO setups
                (doc_id, ticker, trade_date, catalyst_type, embedding,
                 news_header, news_summary,
                 day1_return, day3_return, max_adverse_move,
                 held_above_21ema, regime_at_exit, exit_trigger)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO NOTHING
            """,
            (
                doc_id, ticker, date_str, catalyst_type, full_vector,
                news_header, news_summary,
                float(outcome.get("day1_return") or 0.0),
                float(outcome.get("day3_return") or 0.0),
                float(outcome.get("max_adverse_move") or 0.0),
                float(bool(outcome.get("held_above_21ema", False))),
                str(outcome.get("regime_at_exit") or "UNKNOWN"),
                str(outcome.get("exit_trigger") or "UNKNOWN"),
            ),
        )
    conn.commit()
```

- [ ] **Step 4: Run all store_setup tests**

```bash
TEST_DATABASE_URL="postgresql://localhost/alphacrew_test" \
  pytest tests/test_database.py::test_store_setup_inserts_row \
         tests/test_database.py::test_store_setup_idempotent \
         tests/test_database.py::test_store_setup_persists_news_fields \
         tests/test_database.py::test_store_setup_news_fields_nullable -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add database.py tests/test_database.py
git commit -m "feat: implement store_setup() with news fields and ON CONFLICT DO NOTHING"
```

---

## Task 6: Rewrite query_similar_setups() and Replace Round-Trip Test

**Files:**
- Modify: `database.py`
- Modify: `tests/test_database.py`

- [ ] **Step 1: Write failing tests for query_similar_setups()**

Append to `tests/test_database.py`:

```python
# ---------------------------------------------------------------------------
# query_similar_setups()
# ---------------------------------------------------------------------------

def _seed_n(conn, catalyst_type, n, regime=None, catalyst=None):
    """Insert n rows with deterministic dates for testing."""
    import datetime as _dt
    r = regime or SAMPLE_REGIME
    c = catalyst or SAMPLE_CATALYST
    for i in range(n):
        outcome = {
            "ticker": "SEED",
            "date": (_dt.date(2020, 1, 1) + _dt.timedelta(days=i)).isoformat(),
            "day1_return": 0.03,
            "day3_return": 0.05,
            "max_adverse_move": 0.02,
            "held_above_21ema": True,
            "regime_at_exit": "BULLISH",
            "exit_trigger": "TARGET",
        }
        database.store_setup(conn, catalyst_type, r, c, outcome)

def test_query_returns_required_keys():
    """query_similar_setups() must return a dict with confidence, sample_size, action."""
    conn = _get_conn()
    try:
        result = database.query_similar_setups(
            conn, "earnings_beat_large", SAMPLE_REGIME, SAMPLE_CATALYST
        )
        for key in ("confidence", "sample_size", "action"):
            assert key in result, f"Missing key: {key}"
    finally:
        _drop_setups(conn)
        conn.close()

def test_query_insufficient_when_empty():
    """With no rows, query must return INSUFFICIENT / SKIP_TRADE."""
    conn = _get_conn()
    try:
        result = database.query_similar_setups(
            conn, "fda_rejection", SAMPLE_REGIME, SAMPLE_CATALYST
        )
        assert result["confidence"] == "INSUFFICIENT"
        assert result["action"] == "SKIP_TRADE"
    finally:
        _drop_setups(conn)
        conn.close()

def test_query_filters_by_catalyst_type():
    """query_similar_setups() must only match rows with the requested catalyst_type."""
    conn = _get_conn()
    try:
        _seed_n(conn, "earnings_miss", 50)
        result = database.query_similar_setups(
            conn, "fda_rejection", SAMPLE_REGIME, SAMPLE_CATALYST
        )
        # fda_rejection has no rows — must be INSUFFICIENT
        assert result["confidence"] == "INSUFFICIENT"
        assert result["sample_size"] == 0
    finally:
        _drop_setups(conn)
        conn.close()

def test_query_round_trip():
    """Store 55 identical setups, then query — must clear the confidence floor."""
    conn = _get_conn()
    try:
        _seed_n(conn, "earnings_beat_large", config.VECTOR_DB_MIN_SAMPLES + 15)
        result = database.query_similar_setups(
            conn, "earnings_beat_large", SAMPLE_REGIME, SAMPLE_CATALYST
        )
        assert result["confidence"] != "INSUFFICIENT"
        assert result["sample_size"] >= config.VECTOR_DB_MIN_SAMPLES
        assert result["action"] in ("PROCEED", "SKIP_TRADE")
    finally:
        _drop_setups(conn)
        conn.close()
```

- [ ] **Step 2: Remove the old ChromaDB round-trip test**

In `tests/test_database.py`, delete the entire `test_store_and_query_roundtrip()` function (it imports chromadb directly and will fail). It is fully replaced by `test_query_round_trip()` above.

- [ ] **Step 3: Run tests to confirm they fail**

```bash
TEST_DATABASE_URL="postgresql://localhost/alphacrew_test" \
  pytest tests/test_database.py::test_query_returns_required_keys -v
```

Expected: FAIL — old `query_similar_setups()` still uses ChromaDB client API

- [ ] **Step 4: Replace query_similar_setups() in database.py**

```python
def query_similar_setups(conn, catalyst_type: str, regime_data: dict, catalyst_data: dict) -> dict:
    """Find up to 80 historically similar setups via L2 distance on the 12-dim embedding.

    Filters by catalyst_type in SQL (index), ranks by vector distance, then keeps
    only rows within VECTOR_DB_MAX_DISTANCE in Python — mirrors the old ChromaDB filter.
    Returns the outcome profile dict from calculate_outcome_profile().
    """
    regime_vec = build_regime_vector(regime_data)
    catalyst_vec = build_catalyst_vector(catalyst_data, catalyst_type)
    query_vector = regime_vec + catalyst_vec

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            SELECT ticker, trade_date AS date, catalyst_type,
                   news_header, news_summary,
                   day1_return, day3_return, max_adverse_move,
                   held_above_21ema, regime_at_exit, exit_trigger,
                   embedding <-> %s::vector AS distance
            FROM setups
            WHERE catalyst_type = %s
            ORDER BY distance
            LIMIT 80
            """,
            (query_vector, catalyst_type),
        )
        rows = cur.fetchall()

    # Apply distance threshold in Python — identical semantics to the old ChromaDB filter
    matches = [dict(r) for r in rows if r["distance"] < config.VECTOR_DB_MAX_DISTANCE]
    return calculate_outcome_profile(matches)
```

- [ ] **Step 5: Run all query tests**

```bash
TEST_DATABASE_URL="postgresql://localhost/alphacrew_test" \
  pytest tests/test_database.py::test_query_returns_required_keys \
         tests/test_database.py::test_query_insufficient_when_empty \
         tests/test_database.py::test_query_filters_by_catalyst_type \
         tests/test_database.py::test_query_round_trip -v
```

Expected: All PASS

- [ ] **Step 6: Run the full test suite to confirm no regressions**

```bash
TEST_DATABASE_URL="postgresql://localhost/alphacrew_test" pytest tests/test_database.py -v
```

Expected: All PASS (pure-Python tests for normalize, regime vector, outcome profile are unchanged)

- [ ] **Step 7: Commit**

```bash
git add database.py tests/test_database.py
git commit -m "feat: implement query_similar_setups() with pgvector L2 distance"
```

---

## Task 7: Remove chromadb Import and End-to-End Verification

**Files:**
- Modify: `database.py` (remove leftover imports)

- [ ] **Step 1: Remove chromadb from database.py imports**

At the top of `database.py`, confirm the import block looks exactly like:

```python
import os
import datetime
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
import numpy as np
import config
```

No `import chromadb`, no `import json`. Remove them if present.

- [ ] **Step 2: Full module import smoke test**

```bash
DATABASE_URL="postgresql://localhost/alphacrew" \
  python -c "import config; import database; import regime_engine; import pipeline; import risk_auditor; import scheduler; print('all OK')"
```

Expected: `all OK`

- [ ] **Step 3: Reseed the database**

```bash
DATABASE_URL="postgresql://localhost/alphacrew" python seed_database.py
```

Expected: completes without errors, prints seeding progress

- [ ] **Step 4: Verify row counts per catalyst type**

```bash
psql "$DATABASE_URL" -c "SELECT catalyst_type, COUNT(*) FROM setups GROUP BY catalyst_type ORDER BY catalyst_type;"
```

Expected: 15 rows in result, each with a non-zero count

- [ ] **Step 5: Dry pipeline query — raw catalyst fields, no Polygon call**

```bash
DATABASE_URL="postgresql://localhost/alphacrew" python -c "
import database, config
conn = database.initialize_database()
regime = {
    'spy_pct_above_50sma': 0.03, 'sector_5d_vs_spy': 0.01, 'vix_level': 16.0,
    'sector_rank': 1, 'pct_above_200ema': 0.08, 'premarket_gap_pct': 0.05,
    'rvol_945': 6.0, 'quarter': 2,
}
catalyst = {
    'eps_surprise_pct': 0.12, 'revenue_surprise_pct': 0.08,
    'guidance_delta_pct': 0.05, 'analyst_revision_count': 7,
}
result = database.query_similar_setups(conn, 'earnings_beat_large', regime, catalyst)
print('confidence :', result['confidence'])
print('sample_size:', result['sample_size'])
print('action     :', result['action'])
print('expected_value:', result.get('expected_value'))
"
```

Expected: prints all four fields without error

- [ ] **Step 6: Delete the chroma_db directory**

```bash
rm -rf /home/user/AlphaCrew/chroma_db
```

- [ ] **Step 7: Final full test run**

```bash
TEST_DATABASE_URL="postgresql://localhost/alphacrew_test" pytest tests/ -v --tb=short
```

Expected: All tests pass (chromadb is no longer imported anywhere in the suite)

- [ ] **Step 8: Final commit**

```bash
git add database.py
git commit -m "chore: complete ChromaDB → pgvector migration, remove chroma_db directory"
```

---

## Verification Checklist

| Check | Command | Expected |
|-------|---------|----------|
| Schema applied | `psql $DATABASE_URL -c "\d setups"` | 14 columns incl. `embedding vector(12)` |
| Imports clean | `python -c "import config; import database; ..."` | `all OK` |
| Seed data loads | `python seed_database.py` | No errors |
| Row counts | `SELECT catalyst_type, COUNT(*) FROM setups GROUP BY 1` | 15 types |
| Dry query | python snippet in Task 7 Step 5 | confidence + sample_size printed |
| Full tests | `pytest tests/ -v` | All pass |
| chroma_db gone | `ls chroma_db` | `No such file or directory` |
