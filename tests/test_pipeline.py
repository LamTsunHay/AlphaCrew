"""Tests for pipeline.py — catalyst classification, EASS math, run_pipeline skip logic."""

import pytest
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pipeline
import config


# ---------------------------------------------------------------------------
# classify_catalyst_type()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title,expected", [
    ("Company XYZ to acquire Rival Corp", "ma_acquirer"),
    ("Rival Corp acquired by XYZ in $5B deal", "ma_target"),
    ("FDA approved new drug NDA for XYZ", "fda_approval_nda"),
    ("FDA grants fast track designation", "fda_approval_fast_track"),
    ("FDA rejects XYZ drug application", "fda_rejection"),
    ("Company announces $2B buyback program", "buyback_initiation"),
    ("XYZ wins Department of Defense contract", "government_contract"),
    ("XYZ wins commercial contract with airline", "commercial_contract"),
    ("Company raises full-year guidance above expectations", "guidance_raise_full"),
    ("Company cuts guidance below consensus", "guidance_cut"),
    ("Company reports strong earnings beat", "earnings_beat_large"),
    ("Stock surges on heavy volume", "market_movers"),
])
def test_classify_catalyst_type(title, expected):
    article = {"title": title, "description": ""}
    assert pipeline.classify_catalyst_type(article) == expected

def test_classify_priority_ma_over_earnings():
    """M&A keywords take priority over earnings keywords."""
    article = {"title": "Company acquires rival in earnings beat quarter", "description": ""}
    assert pipeline.classify_catalyst_type(article) == "ma_acquirer"


# ---------------------------------------------------------------------------
# extract_eass_inputs()
# ---------------------------------------------------------------------------

def test_extract_eps_values():
    articles = [{"title": "NVDA earns $5.16 vs $4.60 expected", "description": ""}]
    result = pipeline.extract_eass_inputs(articles, "NVDA")
    assert result["actual_eps"] == pytest.approx(5.16)
    assert result["consensus_eps"] == pytest.approx(4.60)

def test_extract_revenue_billions():
    articles = [{"title": "Revenue $13.5B versus $12.1B", "description": ""}]
    result = pipeline.extract_eass_inputs(articles, "NVDA")
    assert result["actual_revenue"] == pytest.approx(13.5)
    assert result["consensus_revenue"] == pytest.approx(12.1)

def test_extract_guidance_raised():
    articles = [{"title": "Company raises full-year guidance to $22.5", "description": ""}]
    result = pipeline.extract_eass_inputs(articles, "X")
    assert result["guidance_raised"] is True

def test_extract_ceo_positive():
    articles = [{"title": "CEO says results exceeded all expectations", "description": ""}]
    result = pipeline.extract_eass_inputs(articles, "X")
    assert result["ceo_statement_positive"] is True

def test_extract_no_data():
    articles = [{"title": "Stock moves higher", "description": ""}]
    result = pipeline.extract_eass_inputs(articles, "X")
    assert result["actual_eps"] is None
    assert result["guidance_raised"] is False


# ---------------------------------------------------------------------------
# calculate_eass()
# ---------------------------------------------------------------------------

def test_eass_confirmed_bullish():
    inputs = {
        "actual_eps": 6.00, "consensus_eps": 4.60,  # 30% beat → score clearly > 2.0
        "guidance_raised": True, "guidance_cut": False, "guidance_delta_pct": 0.08,
        "raw_text_combined": "",
    }
    result = pipeline.calculate_eass(inputs, "earnings_beat_large")
    assert result["signal_quality"] == "CONFIRMED_BULLISH"
    assert result["eass_score"] > 2.0

def test_eass_guidance_destruction_risk():
    inputs = {
        "actual_eps": 5.0, "consensus_eps": 4.0,
        "guidance_raised": False, "guidance_cut": True, "guidance_delta_pct": -0.10,
        "raw_text_combined": "",
    }
    result = pipeline.calculate_eass(inputs, "earnings_beat_large")
    assert result["signal_quality"] == "GUIDANCE_DESTRUCTION_RISK"

def test_eass_analyst_beat_only():
    inputs = {
        "actual_eps": 3.5, "consensus_eps": 3.0,
        "guidance_raised": False, "guidance_cut": False, "guidance_delta_pct": 0.0,
        "raw_text_combined": "",
    }
    result = pipeline.calculate_eass(inputs, "earnings_beat_large")
    assert result["signal_quality"] == "ANALYST_BEAT_ONLY"

def test_eass_score_normalized_range():
    """EASS score must always be within [-10, +10]."""
    inputs = {
        "actual_eps": 100.0, "consensus_eps": 0.01,
        "guidance_raised": True, "guidance_cut": False, "guidance_delta_pct": 1.0,
        "raw_text_combined": "",
    }
    result = pipeline.calculate_eass(inputs, "earnings_beat_large")
    assert -10.0 <= result["eass_score"] <= 10.0

def test_eass_negative_guidance_amplification():
    """Negative guidance component should be multiplied by 2.5."""
    inputs = {
        "actual_eps": None, "consensus_eps": None,
        "guidance_raised": False, "guidance_cut": True, "guidance_delta_pct": -0.10,
        "raw_text_combined": "",
    }
    result = pipeline.calculate_eass(inputs, "guidance_cut")
    # guidance_component = -0.10 * 2.5 = -0.25
    # raw_eass = 0.55 * -0.25 = -0.1375 → normalized = -0.1375 * 20 = -2.75
    assert result["guidance_component"] == pytest.approx(-0.25, abs=1e-4)
    assert result["eass_score"] == pytest.approx(-2.75, abs=1e-3)

def test_eass_no_eps_no_guidance_mixed_signal():
    inputs = {
        "actual_eps": None, "consensus_eps": None,
        "guidance_raised": False, "guidance_cut": False, "guidance_delta_pct": 0.0,
        "raw_text_combined": "",
    }
    result = pipeline.calculate_eass(inputs, "market_movers")
    assert result["signal_quality"] == "MIXED_SIGNAL"
    assert result["eass_score"] == 0.0


# ---------------------------------------------------------------------------
# run_pipeline() — skip logic with mocked paid APIs
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_pipeline_no_articles_skipped():
    """Tickers with no news articles are skipped with NO_CATALYST_FOUND."""
    mock_client = MagicMock()
    with patch("pipeline.fetch_news", new_callable=AsyncMock, return_value=[]), \
         patch("pipeline.llm_client.create_client", return_value=(mock_client, "anthropic")):
        result = await pipeline.run_pipeline(
            [{"ticker": "XYZ", "price": 100, "ema200": 90, "pre_market_gap_pct": 0.03, "rvol_945": 3.0}],
            {"regime": "BULLISH", "spy_pct_above_50sma": 0.02},
            MagicMock(),
        )
    assert result == []


@pytest.mark.asyncio
async def test_run_pipeline_market_mover_skipped():
    """Tickers classified as market_movers are skipped with LOW_WEIGHT_CATALYST."""
    article = {"title": "Stock surges on heavy volume", "description": ""}
    mock_client = MagicMock()
    with patch("pipeline.fetch_news", new_callable=AsyncMock, return_value=[article]), \
         patch("pipeline.llm_client.create_client", return_value=(mock_client, "anthropic")):
        result = await pipeline.run_pipeline(
            [{"ticker": "XYZ", "price": 100, "ema200": 90, "pre_market_gap_pct": 0.03, "rvol_945": 3.0}],
            {"regime": "BULLISH", "spy_pct_above_50sma": 0.02},
            MagicMock(),
        )
    assert result == []


@pytest.mark.asyncio
async def test_run_pipeline_low_eass_skipped():
    """Tickers with EASS < 2.0 are skipped."""
    article = {"title": "Company earns $1.00 vs $1.00 expected", "description": ""}
    mock_client = MagicMock()
    with patch("pipeline.fetch_news", new_callable=AsyncMock, return_value=[article]), \
         patch("pipeline.llm_client.create_client", return_value=(mock_client, "anthropic")):
        result = await pipeline.run_pipeline(
            [{"ticker": "XYZ", "price": 100, "ema200": 90, "pre_market_gap_pct": 0.03, "rvol_945": 3.0}],
            {"regime": "BULLISH", "spy_pct_above_50sma": 0.02},
            MagicMock(),
        )
    assert result == []


# ---------------------------------------------------------------------------
# select_best_catalyst()
# ---------------------------------------------------------------------------

def test_select_best_catalyst_finds_buried_ma():
    """articles[0] is generic noise; the catalyst is in articles[1]."""
    articles = [
        {"title": "Stock surges on heavy volume", "description": ""},
        {"title": "Company XYZ to acquire Rival Corp", "description": ""},
    ]
    assert pipeline.select_best_catalyst(articles) == "ma_acquirer"


def test_select_best_catalyst_all_generic_returns_market_movers():
    articles = [
        {"title": "Stock surges on heavy volume", "description": ""},
        {"title": "Trading session recap", "description": ""},
    ]
    assert pipeline.select_best_catalyst(articles) == "market_movers"


def test_select_best_catalyst_prefers_ma_over_earnings():
    """M&A is higher priority than earnings even if earnings article comes first."""
    articles = [
        {"title": "Company reports strong earnings beat", "description": ""},
        {"title": "Company XYZ to acquire Rival Corp", "description": ""},
    ]
    assert pipeline.select_best_catalyst(articles) == "ma_acquirer"


def test_select_best_catalyst_single_fda_article():
    articles = [{"title": "FDA approved new drug NDA for XYZ", "description": ""}]
    assert pipeline.select_best_catalyst(articles) == "fda_approval_nda"


def test_select_best_catalyst_catalyst_in_description_field():
    """Catalyst keyword in description (not title) must still be found."""
    articles = [
        {"title": "Market update", "description": "Company acquires rival in $2B deal"},
    ]
    assert pipeline.select_best_catalyst(articles) == "ma_acquirer"


def test_select_best_catalyst_duplicate_catalyst_type():
    """Two articles with the same catalyst type should still return that type."""
    articles = [
        {"title": "Company XYZ to acquire Rival Corp", "description": ""},
        {"title": "XYZ acquisition deal confirmed", "description": "acquiring target for $3B"},
    ]
    assert pipeline.select_best_catalyst(articles) == "ma_acquirer"


def test_select_best_catalyst_prefers_fda_over_guidance():
    articles = [
        {"title": "Company raises full-year guidance above expectations", "description": ""},
        {"title": "FDA approved new drug NDA for XYZ", "description": ""},
    ]
    assert pipeline.select_best_catalyst(articles) == "fda_approval_nda"


def test_select_best_catalyst_empty_list_returns_market_movers():
    assert pipeline.select_best_catalyst([]) == "market_movers"
