"""Tests for database.py — vector math, outcome profiles, and PostgreSQL pgvector round-trip."""

import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import database
import config


# ---------------------------------------------------------------------------
# normalize()
# ---------------------------------------------------------------------------

def test_normalize_midpoint():
    assert database.normalize(0.5, 0.0, 1.0) == pytest.approx(0.5)

def test_normalize_clips_low():
    assert database.normalize(-999, 0, 10) == 0.0

def test_normalize_clips_high():
    assert database.normalize(999, 0, 10) == 1.0

def test_normalize_same_min_max():
    assert database.normalize(5.0, 5.0, 5.0) == 0.0


# ---------------------------------------------------------------------------
# build_regime_vector()
# ---------------------------------------------------------------------------

def test_regime_vector_length():
    rd = {
        "spy_pct_above_50sma": 0.02, "sector_5d_vs_spy": 0.01, "vix_level": 18.0,
        "sector_rank": 1, "pct_above_200ema": 0.05, "premarket_gap_pct": 0.04,
        "rvol_945": 5.0, "quarter": 2,
    }
    vec = database.build_regime_vector(rd)
    assert len(vec) == 8

def test_regime_vector_range():
    rd = {
        "spy_pct_above_50sma": 0.0, "sector_5d_vs_spy": 0.0, "vix_level": 20.0,
        "sector_rank": 2, "pct_above_200ema": 0.10, "premarket_gap_pct": 0.05,
        "rvol_945": 4.0, "quarter": 1,
    }
    vec = database.build_regime_vector(rd)
    for v in vec:
        assert 0.0 <= v <= 1.0, f"Out of range: {v}"

def test_regime_vector_vix_inverted():
    low_vix = database.build_regime_vector({
        "spy_pct_above_50sma": 0.0, "sector_5d_vs_spy": 0.0, "vix_level": 12.0,
        "sector_rank": 2, "pct_above_200ema": 0.0, "premarket_gap_pct": 0.02,
        "rvol_945": 3.0, "quarter": 1,
    })
    high_vix = database.build_regime_vector({
        "spy_pct_above_50sma": 0.0, "sector_5d_vs_spy": 0.0, "vix_level": 35.0,
        "sector_rank": 2, "pct_above_200ema": 0.0, "premarket_gap_pct": 0.02,
        "rvol_945": 3.0, "quarter": 1,
    })
    assert low_vix[2] > high_vix[2]  # dim[2] = VIX inverted


# ---------------------------------------------------------------------------
# build_catalyst_vector()
# ---------------------------------------------------------------------------

def test_catalyst_vector_length_earnings():
    cd = {"eps_surprise_pct": 0.1, "revenue_surprise_pct": 0.05, "guidance_delta_pct": 0.03, "analyst_revision_count": 5}
    vec = database.build_catalyst_vector(cd, "earnings_beat_large")
    assert len(vec) == 4

def test_catalyst_vector_length_fda():
    cd = {"was_expected": True, "has_competitor": False, "market_cap_billions": 5.0, "pipeline_depth": 3}
    vec = database.build_catalyst_vector(cd, "fda_approval_nda")
    assert len(vec) == 4

def test_catalyst_vector_fallback():
    vec = database.build_catalyst_vector({}, "unknown_type_xyz")
    assert vec == [0.5, 0.5, 0.5, 0.5]

def test_catalyst_vector_range():
    cd = {"eps_surprise_pct": 0.15, "revenue_surprise_pct": 0.10, "guidance_delta_pct": 0.05, "analyst_revision_count": 8}
    for v in database.build_catalyst_vector(cd, "earnings_beat_large"):
        assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# calculate_outcome_profile()
# ---------------------------------------------------------------------------

def test_outcome_profile_insufficient():
    result = database.calculate_outcome_profile([])
    assert result["confidence"] == "INSUFFICIENT"
    assert result["action"] == "SKIP_TRADE"
    assert result["win_rate_day1"] is None

def test_outcome_profile_insufficient_exact_threshold():
    matches = [
        {"day1_return": 0.02, "day3_return": 0.03, "max_adverse_move": 0.01, "held_above_21ema": 1.0}
        for _ in range(config.VECTOR_DB_MIN_SAMPLES - 1)
    ]
    result = database.calculate_outcome_profile(matches)
    assert result["confidence"] == "INSUFFICIENT"

def _make_matches(n, win_rate=0.6, avg_win=0.05, avg_loss=-0.03):
    """Generate n deterministic mock match dicts."""
    import random
    rng = random.Random(0)
    matches = []
    for _ in range(n):
        win = rng.random() < win_rate
        d3 = avg_win if win else avg_loss
        d1 = d3 * 0.7
        adverse = abs(d3) * 0.5
        matches.append({
            "day1_return": d1,
            "day3_return": d3,
            "max_adverse_move": adverse,
            "held_above_21ema": 1.0 if win else 0.0,
        })
    return matches

def test_outcome_profile_proceed():
    matches = _make_matches(60, win_rate=0.70, avg_win=0.06, avg_loss=-0.02)
    result = database.calculate_outcome_profile(matches)
    assert result["action"] == "PROCEED"
    assert result["expected_value"] > 0
    assert result["sample_size"] == 60
    assert 0.0 <= result["win_rate_day3"] <= 1.0

def test_outcome_profile_low_confidence():
    matches = _make_matches(45)
    result = database.calculate_outcome_profile(matches)
    assert result["confidence"] == "LOW"

def test_outcome_profile_moderate_confidence():
    matches = _make_matches(80)
    result = database.calculate_outcome_profile(matches)
    assert result["confidence"] == "MODERATE"

def test_outcome_profile_suggested_stop_negative():
    matches = _make_matches(50)
    result = database.calculate_outcome_profile(matches)
    assert result["suggested_stop"] <= 0.0


# ---------------------------------------------------------------------------
# Config sanity
# ---------------------------------------------------------------------------

def test_postgres_dsn_in_config():
    """config.POSTGRES_DSN must exist and be non-empty string attribute."""
    import config as _config
    assert hasattr(_config, "POSTGRES_DSN"), "POSTGRES_DSN missing from config"
    assert _config.POSTGRES_DSN is not None, "POSTGRES_DSN must not be None — set DATABASE_URL in .env"

def test_vector_db_path_removed():
    """VECTOR_DB_PATH must be removed — chroma_db directory is no longer used."""
    import config as _config
    assert not hasattr(_config, "VECTOR_DB_PATH"), "VECTOR_DB_PATH must be removed from config"


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

# ---------------------------------------------------------------------------
# build_catalyst_vector() — semantic tests
# ---------------------------------------------------------------------------

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
