"""Tests for risk_auditor.py — mutator, position sizing, card format, audit fallback."""

import pytest
import os
import sys
import json
import tempfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import risk_auditor
import config


# ---------------------------------------------------------------------------
# apply_regime_strategy_mutator()
# ---------------------------------------------------------------------------

def test_bearish_mutator_downgrades_market_open():
    audit = {
        "entry_strategy": "MARKET_OPEN",
        "threats_identified": [],
    }
    result = risk_auditor.apply_regime_strategy_mutator(audit, "BEARISH")
    assert result["entry_strategy"] == "PULLBACK_LIMIT_21EMA"
    assert any("BEARISH REGIME" in t for t in result["threats_identified"])

def test_bearish_mutator_preserves_pullback():
    audit = {
        "entry_strategy": "PULLBACK_LIMIT_21EMA",
        "threats_identified": [],
    }
    result = risk_auditor.apply_regime_strategy_mutator(audit, "BEARISH")
    assert result["entry_strategy"] == "PULLBACK_LIMIT_21EMA"

def test_bullish_mutator_noop():
    audit = {
        "entry_strategy": "MARKET_OPEN",
        "threats_identified": [],
    }
    result = risk_auditor.apply_regime_strategy_mutator(audit, "BULLISH")
    assert result["entry_strategy"] == "MARKET_OPEN"
    assert result["threats_identified"] == []


# ---------------------------------------------------------------------------
# calculate_position_size()
# ---------------------------------------------------------------------------

def test_position_size_bullish():
    result = risk_auditor.calculate_position_size(100_000, 0.02, "BULLISH")
    assert result["risk_amount_dollars"] == pytest.approx(1000.0)
    assert result["regime_applied"] == "BULLISH"
    assert isinstance(result["suggested_shares"], int)

def test_position_size_bearish_halved():
    bull = risk_auditor.calculate_position_size(100_000, 0.02, "BULLISH")
    bear = risk_auditor.calculate_position_size(100_000, 0.02, "BEARISH")
    assert bear["risk_amount_dollars"] == pytest.approx(bull["risk_amount_dollars"] * 0.5)

def test_position_size_zero_stop():
    result = risk_auditor.calculate_position_size(100_000, 0.0, "BULLISH")
    assert result["suggested_shares"] == 0

def test_position_size_stop_distance_stored():
    result = risk_auditor.calculate_position_size(100_000, 0.025, "BULLISH")
    assert result["stop_distance_pct"] == pytest.approx(0.025)


# ---------------------------------------------------------------------------
# format_strategy_card()
# ---------------------------------------------------------------------------

def _make_candidate():
    return {
        "ticker": "NVDA",
        "catalyst_type": "earnings_beat_large",
        "eass": {"eass_score": 6.5, "signal_quality": "CONFIRMED_BULLISH"},
        "haiku_summary": "NVDA beat estimates. Margins expanded. Guidance raised.",
        "outcome_profile": {
            "confidence": "LOW", "sample_size": 55,
            "win_rate_day1": 0.72, "win_rate_day3": 0.69,
            "expected_value": 0.031, "avg_max_adverse": 0.018,
            "suggested_stop": -0.025, "suggested_target": 0.06,
        },
    }

def test_format_card_contains_ticker():
    card = risk_auditor.format_strategy_card(
        _make_candidate(),
        {"threat_level": "LOW", "threats_identified": [], "durability": "DURABLE",
         "regime_support": "CONFIRMED", "entry_strategy": "MARKET_OPEN", "audit_note": "OK"},
        {"risk_amount_dollars": 1000, "stop_distance_pct": 0.025, "suggested_shares": 10, "regime_applied": "BULLISH"},
    )
    assert "NVDA" in card
    assert "CONFIRMED_BULLISH" in card
    assert "LOW" in card  # confidence or threat_level
    assert "MARKET_OPEN" in card

def test_format_card_no_threats():
    card = risk_auditor.format_strategy_card(
        _make_candidate(),
        {"threat_level": "LOW", "threats_identified": [], "durability": "DURABLE",
         "regime_support": "CONFIRMED", "entry_strategy": "MARKET_OPEN", "audit_note": "OK"},
        {"risk_amount_dollars": 1000, "stop_distance_pct": 0.025, "suggested_shares": 10, "regime_applied": "BULLISH"},
    )
    assert "None identified" in card


# ---------------------------------------------------------------------------
# write_outputs()
# ---------------------------------------------------------------------------

def test_write_outputs_creates_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_strat = config.STRATEGY_OUTPUT_PATH
        orig_log = config.LOG_QUEUE_PATH
        config.STRATEGY_OUTPUT_PATH = os.path.join(tmpdir, "cards.md")
        config.LOG_QUEUE_PATH = os.path.join(tmpdir, "log.json")

        try:
            risk_auditor.write_outputs(["## Card 1\n---", "## Card 2\n---"], [{"step": "TEST"}])
            assert os.path.exists(config.STRATEGY_OUTPUT_PATH)
            assert os.path.exists(config.LOG_QUEUE_PATH)
            with open(config.LOG_QUEUE_PATH) as f:
                data = json.load(f)
            assert data[0]["step"] == "TEST"
        finally:
            config.STRATEGY_OUTPUT_PATH = orig_strat
            config.LOG_QUEUE_PATH = orig_log


# ---------------------------------------------------------------------------
# run_sonnet_audit() — fallback on parse failure
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_sonnet_audit_fallback_on_bad_json():
    """If the LLM returns non-JSON, safe default is returned."""
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="Not valid JSON at all")]
    mock_client.messages.create.return_value = mock_msg

    result = await risk_auditor.run_sonnet_audit({"ticker": "X"}, mock_client, "anthropic")
    assert result["entry_strategy"] == "DO_NOT_ENTER"
    assert result["threat_level"] == "HIGH"

@pytest.mark.asyncio
async def test_sonnet_audit_valid_json():
    """If the LLM returns valid JSON, it is parsed and returned."""
    valid = {
        "threat_level": "LOW",
        "threats_identified": [],
        "durability": "DURABLE",
        "regime_support": "CONFIRMED",
        "entry_strategy": "MARKET_OPEN",
        "audit_note": "Setup looks clean.",
    }
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text=json.dumps(valid))]
    mock_client.messages.create.return_value = mock_msg

    result = await risk_auditor.run_sonnet_audit({"ticker": "X"}, mock_client, "anthropic")
    assert result["entry_strategy"] == "MARKET_OPEN"
    assert result["threat_level"] == "LOW"
