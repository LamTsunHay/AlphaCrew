"""Tests that sector leadership gate runs AFTER the paid pipeline in both
scheduler.py and run.py.

Core scenario: two tickers share a sub-industry. TICKER_A has a higher
pre-market gap but no news catalyst; TICKER_B has a lower gap but a
qualifying catalyst. With the old gate order (leadership before pipeline),
TICKER_A wins the industry slot and enters the pipeline with no news —
the whole industry is lost. With the fix, both reach the pipeline; TICKER_B
qualifies, then wins its industry slot in the post-pipeline dedup.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

REGIME_BULLISH = {
    "regime": "BULLISH",
    "spy_close": 500.0,
    "spy_sma50": 480.0,
    "pct_above_50sma": 0.04,
}

SECTOR_DATA = {
    "top_sectors": ["XLK", "XLV"],
    "sector_rankings": {},
    "spy_5d_return": 0.01,
}

# Higher gap, no news — should be eliminated by the pipeline
TICKER_A_ENTRY = {
    "ticker": "TICKER_A",
    "price": 100.0,
    "atr14": 5.0,
    "volatility_ratio": 0.05,
    "ema21": 95.0,
    "ema200": 80.0,
    "pre_market_gap_pct": 0.08,
}

# Lower gap, has catalyst — should survive the pipeline
TICKER_B_ENTRY = {
    "ticker": "TICKER_B",
    "price": 100.0,
    "atr14": 5.0,
    "volatility_ratio": 0.05,
    "ema21": 95.0,
    "ema200": 80.0,
    "pre_market_gap_pct": 0.03,
}

# Pipeline only qualifies TICKER_B
PIPELINE_CANDIDATES = [
    {
        "ticker": "TICKER_B",
        "catalyst_type": "earnings_beat_large",
        "eass": {"eass_score": 5.0},
        "haiku_summary": "Strong beat.",
        "outcome_profile": {
            "action": "TRADE",
            "confidence": 50,
            "suggested_stop": 0.02,
        },
        "pre_market_gap_pct": 0.03,
        "metrics": {**TICKER_B_ENTRY},
    }
]

AUDIT_RESULT = {
    "entry_strategy": "MARKET_OPEN",
    "risk_level": "LOW",
    "stop_loss": 0.02,
    "suggested_stop": 0.02,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_leadership_gate_that_records(call_log):
    """Return a fake apply_sector_leadership_gate that records its input."""
    def fake(entries):
        call_log.append([e["ticker"] for e in entries])
        return entries  # pass all through; ordering test only cares about input
    return fake


def _make_pipeline_that_records(call_log):
    """Return a fake pipeline that records its input and returns candidates."""
    async def fake(tickers_with_metrics, regime_data, db_client, test_mode=None):
        call_log.append([e["ticker"] for e in tickers_with_metrics])
        return PIPELINE_CANDIDATES
    return fake


# ---------------------------------------------------------------------------
# scheduler.py tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_scheduler_pipeline_receives_all_earnings_survivors():
    """scheduler feeds ALL earnings survivors into the pipeline, not just the
    gap leader, so every ticker has a chance to contribute a news catalyst."""
    pipeline_inputs = []

    with (
        patch("scheduler.regime_engine.get_market_regime", return_value=REGIME_BULLISH),
        patch("scheduler.regime_engine.get_top_sectors", return_value=SECTOR_DATA),
        patch("scheduler.regime_engine.filter_by_sector", return_value=["TICKER_A", "TICKER_B"]),
        patch("scheduler.regime_engine.apply_individual_gates", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("scheduler.regime_engine.apply_earnings_gate", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("scheduler._get_premarket_gap", return_value=0.05),
        patch("scheduler.pipeline.run_pipeline", new=_make_pipeline_that_records(pipeline_inputs)),
        patch("scheduler.regime_engine.apply_sector_leadership_gate", return_value=PIPELINE_CANDIDATES),
        patch("scheduler.risk_auditor.run_sonnet_audit", new=AsyncMock(return_value=AUDIT_RESULT)),
        patch("scheduler.risk_auditor.apply_regime_strategy_mutator", return_value=AUDIT_RESULT),
        patch("scheduler.risk_auditor.calculate_position_size", return_value={"shares": 10}),
        patch("scheduler.risk_auditor.format_strategy_card", return_value="card"),
        patch("scheduler.risk_auditor.write_outputs"),
        patch("scheduler.llm_client.create_client", return_value=(MagicMock(), "anthropic")),
    ):
        import scheduler
        await scheduler.run_premarket_pipeline(db_client=MagicMock())

    assert pipeline_inputs, "pipeline was never called"
    assert set(pipeline_inputs[0]) == {"TICKER_A", "TICKER_B"}, (
        "pipeline should receive all earnings survivors, not just the gap leader"
    )


@pytest.mark.asyncio
async def test_scheduler_leadership_gate_receives_pipeline_output():
    """scheduler calls the leadership gate with pipeline-qualified candidates,
    not with the full earnings-survivor list."""
    leadership_inputs = []

    with (
        patch("scheduler.regime_engine.get_market_regime", return_value=REGIME_BULLISH),
        patch("scheduler.regime_engine.get_top_sectors", return_value=SECTOR_DATA),
        patch("scheduler.regime_engine.filter_by_sector", return_value=["TICKER_A", "TICKER_B"]),
        patch("scheduler.regime_engine.apply_individual_gates", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("scheduler.regime_engine.apply_earnings_gate", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("scheduler._get_premarket_gap", return_value=0.05),
        patch("scheduler.pipeline.run_pipeline", new=AsyncMock(return_value=PIPELINE_CANDIDATES)),
        patch("scheduler.regime_engine.apply_sector_leadership_gate",
              side_effect=_make_leadership_gate_that_records(leadership_inputs)),
        patch("scheduler.risk_auditor.run_sonnet_audit", new=AsyncMock(return_value=AUDIT_RESULT)),
        patch("scheduler.risk_auditor.apply_regime_strategy_mutator", return_value=AUDIT_RESULT),
        patch("scheduler.risk_auditor.calculate_position_size", return_value={"shares": 10}),
        patch("scheduler.risk_auditor.format_strategy_card", return_value="card"),
        patch("scheduler.risk_auditor.write_outputs"),
        patch("scheduler.llm_client.create_client", return_value=(MagicMock(), "anthropic")),
    ):
        import scheduler
        await scheduler.run_premarket_pipeline(db_client=MagicMock())

    assert leadership_inputs, "apply_sector_leadership_gate was never called"
    assert leadership_inputs[0] == ["TICKER_B"], (
        "leadership gate should receive pipeline candidates (only TICKER_B qualified), "
        "not the full earnings-survivor list"
    )


@pytest.mark.asyncio
async def test_scheduler_lower_gap_ticker_qualifies_when_higher_gap_has_no_news():
    """End-to-end: TICKER_B (lower gap, has catalyst) makes it to strategy cards
    even though TICKER_A (higher gap, no news) shares its sub-industry."""
    formatted_cards = []

    def fake_format_card(candidate, audit, position):
        formatted_cards.append(candidate["ticker"])
        return f"card_{candidate['ticker']}"

    with (
        patch("scheduler.regime_engine.get_market_regime", return_value=REGIME_BULLISH),
        patch("scheduler.regime_engine.get_top_sectors", return_value=SECTOR_DATA),
        patch("scheduler.regime_engine.filter_by_sector", return_value=["TICKER_A", "TICKER_B"]),
        patch("scheduler.regime_engine.apply_individual_gates", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("scheduler.regime_engine.apply_earnings_gate", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("scheduler._get_premarket_gap", return_value=0.05),
        patch("scheduler.pipeline.run_pipeline", new=AsyncMock(return_value=PIPELINE_CANDIDATES)),
        patch("scheduler.regime_engine.apply_sector_leadership_gate", return_value=PIPELINE_CANDIDATES),
        patch("scheduler.risk_auditor.run_sonnet_audit", new=AsyncMock(return_value=AUDIT_RESULT)),
        patch("scheduler.risk_auditor.apply_regime_strategy_mutator", return_value=AUDIT_RESULT),
        patch("scheduler.risk_auditor.calculate_position_size", return_value={"shares": 10}),
        patch("scheduler.risk_auditor.format_strategy_card", side_effect=fake_format_card),
        patch("scheduler.risk_auditor.write_outputs"),
        patch("scheduler.llm_client.create_client", return_value=(MagicMock(), "anthropic")),
    ):
        import scheduler
        await scheduler.run_premarket_pipeline(db_client=MagicMock())

    assert "TICKER_B" in formatted_cards, (
        "TICKER_B should produce a strategy card even though TICKER_A had a higher gap"
    )


# ---------------------------------------------------------------------------
# run.py tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_pipeline_receives_all_earnings_survivors():
    """run.main feeds ALL earnings survivors into the pipeline, not just the
    gap leader."""
    pipeline_inputs = []

    with (
        patch("run.regime_engine.get_market_regime", return_value=REGIME_BULLISH),
        patch("run.regime_engine.get_top_sectors", return_value=SECTOR_DATA),
        patch("run.regime_engine.filter_by_sector", return_value=["TICKER_A", "TICKER_B"]),
        patch("run.regime_engine.apply_individual_gates", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("run.regime_engine.apply_earnings_gate", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("run._get_premarket_gap", return_value=0.05),
        patch("run.pipeline.run_pipeline", new=_make_pipeline_that_records(pipeline_inputs)),
        patch("run.regime_engine.apply_sector_leadership_gate", return_value=PIPELINE_CANDIDATES),
        patch("run.risk_auditor.run_sonnet_audit", new=AsyncMock(return_value=AUDIT_RESULT)),
        patch("run.risk_auditor.apply_regime_strategy_mutator", return_value=AUDIT_RESULT),
        patch("run.risk_auditor.calculate_position_size", return_value={"shares": 10}),
        patch("run.risk_auditor.format_strategy_card", return_value="card"),
        patch("run.notifier.print_to_terminal"),
        patch("run.database.initialize_database", return_value=MagicMock()),
        patch("run.llm_client.create_client", return_value=(MagicMock(), "anthropic")),
    ):
        import run
        await run.main(["TICKER_A", "TICKER_B", "--regime", "BULLISH"])

    assert pipeline_inputs, "pipeline was never called"
    assert set(pipeline_inputs[0]) == {"TICKER_A", "TICKER_B"}, (
        "pipeline should receive all earnings survivors, not just the gap leader"
    )


@pytest.mark.asyncio
async def test_run_leadership_gate_receives_pipeline_output():
    """run.main calls the leadership gate with pipeline-qualified candidates,
    not with the full earnings-survivor list."""
    leadership_inputs = []

    with (
        patch("run.regime_engine.get_market_regime", return_value=REGIME_BULLISH),
        patch("run.regime_engine.get_top_sectors", return_value=SECTOR_DATA),
        patch("run.regime_engine.filter_by_sector", return_value=["TICKER_A", "TICKER_B"]),
        patch("run.regime_engine.apply_individual_gates", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("run.regime_engine.apply_earnings_gate", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("run._get_premarket_gap", return_value=0.05),
        patch("run.pipeline.run_pipeline", new=AsyncMock(return_value=PIPELINE_CANDIDATES)),
        patch("run.regime_engine.apply_sector_leadership_gate",
              side_effect=_make_leadership_gate_that_records(leadership_inputs)),
        patch("run.risk_auditor.run_sonnet_audit", new=AsyncMock(return_value=AUDIT_RESULT)),
        patch("run.risk_auditor.apply_regime_strategy_mutator", return_value=AUDIT_RESULT),
        patch("run.risk_auditor.calculate_position_size", return_value={"shares": 10}),
        patch("run.risk_auditor.format_strategy_card", return_value="card"),
        patch("run.notifier.print_to_terminal"),
        patch("run.database.initialize_database", return_value=MagicMock()),
        patch("run.llm_client.create_client", return_value=(MagicMock(), "anthropic")),
    ):
        import run
        await run.main(["TICKER_A", "TICKER_B", "--regime", "BULLISH"])

    assert leadership_inputs, "apply_sector_leadership_gate was never called"
    assert leadership_inputs[0] == ["TICKER_B"], (
        "leadership gate should receive pipeline candidates (only TICKER_B qualified), "
        "not the full earnings-survivor list"
    )


@pytest.mark.asyncio
async def test_run_lower_gap_ticker_qualifies_when_higher_gap_has_no_news():
    """End-to-end: TICKER_B (lower gap, has catalyst) makes it to the notifier
    even though TICKER_A (higher gap, no news) shares its sub-industry."""
    notifier_cards = []

    def fake_notifier(cards, log):
        notifier_cards.extend(cards)

    with (
        patch("run.regime_engine.get_market_regime", return_value=REGIME_BULLISH),
        patch("run.regime_engine.get_top_sectors", return_value=SECTOR_DATA),
        patch("run.regime_engine.filter_by_sector", return_value=["TICKER_A", "TICKER_B"]),
        patch("run.regime_engine.apply_individual_gates", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("run.regime_engine.apply_earnings_gate", return_value=[TICKER_A_ENTRY, TICKER_B_ENTRY]),
        patch("run._get_premarket_gap", return_value=0.05),
        patch("run.pipeline.run_pipeline", new=AsyncMock(return_value=PIPELINE_CANDIDATES)),
        patch("run.regime_engine.apply_sector_leadership_gate", return_value=PIPELINE_CANDIDATES),
        patch("run.risk_auditor.run_sonnet_audit", new=AsyncMock(return_value=AUDIT_RESULT)),
        patch("run.risk_auditor.apply_regime_strategy_mutator", return_value=AUDIT_RESULT),
        patch("run.risk_auditor.calculate_position_size", return_value={"shares": 10}),
        patch("run.risk_auditor.format_strategy_card", return_value="TICKER_B_card"),
        patch("run.notifier.print_to_terminal", side_effect=fake_notifier),
        patch("run.database.initialize_database", return_value=MagicMock()),
        patch("run.llm_client.create_client", return_value=(MagicMock(), "anthropic")),
    ):
        import run
        await run.main(["TICKER_A", "TICKER_B", "--regime", "BULLISH"])

    assert notifier_cards, (
        "TICKER_B should produce a strategy card even though TICKER_A had a higher gap"
    )
