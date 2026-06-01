"""Tests for run.py — arg parsing, ticker defaults, regime override."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import run


def test_parse_args_default_tickers():
    """No positional args → returns NASDAQ_100 list."""
    args = run.parse_args([])
    assert args.tickers == run.NASDAQ_100
    assert args.regime is None


def test_parse_args_custom_tickers():
    """Positional ticker args are captured as a list."""
    args = run.parse_args(["NVDA", "AMD"])
    assert args.tickers == ["NVDA", "AMD"]


def test_parse_args_regime_bullish():
    """--regime BULLISH is accepted."""
    args = run.parse_args(["--regime", "BULLISH"])
    assert args.regime == "BULLISH"


def test_parse_args_regime_bearish():
    """--regime BEARISH is accepted."""
    args = run.parse_args(["--regime", "BEARISH"])
    assert args.regime == "BEARISH"


def test_parse_args_invalid_regime():
    """Invalid --regime value raises SystemExit."""
    with pytest.raises(SystemExit):
        run.parse_args(["--regime", "SIDEWAYS"])


def test_build_regime_data_override():
    """Regime override returns a dict with the forced regime, no live SPY call."""
    result = run.build_regime_data("BEARISH")
    assert result["regime"] == "BEARISH"
    assert "spy_close" in result


def test_build_regime_data_bullish_override():
    """BULLISH override returns correct regime dict."""
    result = run.build_regime_data("BULLISH")
    assert result["regime"] == "BULLISH"


def test_nasdaq_100_has_expected_tickers():
    """NASDAQ_100 constant contains well-known members."""
    assert "AAPL" in run.NASDAQ_100
    assert "MSFT" in run.NASDAQ_100
    assert "NVDA" in run.NASDAQ_100
    assert len(run.NASDAQ_100) >= 100


def test_main_regime_override_skips_live_spy(monkeypatch):
    """When --regime is given, regime_engine.get_market_regime is NOT called."""
    calls = []
    monkeypatch.setattr("regime_engine.get_market_regime", lambda: calls.append(1) or {})
    # Patch all downstream calls so main() exits early without network calls
    monkeypatch.setattr("regime_engine.get_top_sectors", lambda: {"top_sectors": []})
    monkeypatch.setattr("regime_engine.filter_by_sector", lambda tickers, sectors: [])
    monkeypatch.setattr("notifier.print_to_terminal", lambda cards, log: None)
    import asyncio
    asyncio.run(run.main(["NVDA", "--regime", "BULLISH"]))
    assert calls == [], "get_market_regime() must not be called when --regime is provided"


def test_main_no_regime_calls_live_spy(monkeypatch):
    """When --regime is not given, regime_engine.get_market_regime IS called."""
    calls = []
    monkeypatch.setattr(
        "regime_engine.get_market_regime",
        lambda: calls.append(1) or {"regime": "BULLISH", "spy_close": 0.0, "spy_sma50": 0.0, "pct_above_50sma": 0.0},
    )
    monkeypatch.setattr("regime_engine.get_top_sectors", lambda: {"top_sectors": []})
    monkeypatch.setattr("regime_engine.filter_by_sector", lambda tickers, sectors: [])
    monkeypatch.setattr("notifier.print_to_terminal", lambda cards, log: None)
    import asyncio
    asyncio.run(run.main(["NVDA"]))
    assert calls == [1], "get_market_regime() must be called when no --regime override"
