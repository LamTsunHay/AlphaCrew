"""Tests for database.py — vector math, outcome profiles, and ChromaDB round-trip."""

import pytest
import tempfile
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
# ChromaDB round-trip: store_setup → query_similar_setups
# ---------------------------------------------------------------------------

def test_store_and_query_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        import chromadb
        client = chromadb.PersistentClient(path=tmpdir)
        for name in config.CATALYST_COLLECTIONS:
            client.get_or_create_collection(name)

        catalyst_type = "earnings_beat_large"
        regime = {
            "spy_pct_above_50sma": 0.03, "sector_5d_vs_spy": 0.01, "vix_level": 16.0,
            "sector_rank": 1, "pct_above_200ema": 0.08, "premarket_gap_pct": 0.05,
            "rvol_945": 6.0, "quarter": 2,
        }
        catalyst = {
            "eps_surprise_pct": 0.12, "revenue_surprise_pct": 0.08,
            "guidance_delta_pct": 0.05, "analyst_revision_count": 7,
        }

        # Store enough samples to clear the confidence floor
        import datetime as _dt
        for i in range(config.VECTOR_DB_MIN_SAMPLES + 10):
            outcome = {
                "ticker": "TEST",
                "date": (_dt.date(2020, 1, 1) + _dt.timedelta(days=i)).isoformat(),
                "day1_return": 0.03,
                "day3_return": 0.05,
                "max_adverse_move": 0.02,
                "held_above_21ema": True,
                "regime_at_exit": "BULLISH",
                "exit_trigger": "TARGET",
            }
            database.store_setup(client, catalyst_type, regime, catalyst, outcome)

        profile = database.query_similar_setups(client, catalyst_type, regime, catalyst)
        assert profile["confidence"] != "INSUFFICIENT"
        assert profile["sample_size"] >= config.VECTOR_DB_MIN_SAMPLES
