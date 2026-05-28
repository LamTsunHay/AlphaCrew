"""Tests for regime_engine.py — gates verified with mocked yfinance."""

import pytest
import os
import sys
import datetime
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import regime_engine
import config


def _make_close_df(prices: list, ticker: str = "SPY") -> pd.DataFrame:
    """Build a minimal yfinance-style DataFrame (flat, not MultiIndex)."""
    dates = pd.date_range("2024-01-01", periods=len(prices), freq="B")
    return pd.DataFrame({"Close": prices}, index=dates)


def _make_ohlcv_df(prices: list, ticker: str = None) -> pd.DataFrame:
    """Build OHLCV DataFrame for apply_individual_gates."""
    dates = pd.date_range("2022-01-01", periods=len(prices), freq="B")
    close = np.array(prices, dtype=float)
    high = close * 1.01
    low = close * 0.99
    vol = np.full(len(prices), 1_000_000)
    return pd.DataFrame({"Close": close, "High": high, "Low": low, "Volume": vol}, index=dates)


# ---------------------------------------------------------------------------
# get_market_regime()
# ---------------------------------------------------------------------------

def test_regime_bullish():
    prices = [100.0] * 50 + [110.0] * 10  # close > SMA50
    df = _make_close_df(prices)
    with patch("yfinance.download", return_value=df):
        result = regime_engine.get_market_regime()
    assert result["regime"] == "BULLISH"
    assert result["spy_close"] > result["spy_sma50"]

def test_regime_bearish():
    prices = [110.0] * 50 + [90.0] * 10  # close < SMA50
    df = _make_close_df(prices)
    with patch("yfinance.download", return_value=df):
        result = regime_engine.get_market_regime()
    assert result["regime"] == "BEARISH"
    assert result["spy_close"] < result["spy_sma50"]


# ---------------------------------------------------------------------------
# get_top_sectors()
# ---------------------------------------------------------------------------

def test_get_top_sectors_returns_three():
    all_tickers = list(config.SECTOR_ETFS.keys()) + ["SPY"]
    n = 10
    prices = {t: list(range(100, 100 + n)) for t in all_tickers}
    prices["SPY"] = [100.0] * n
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    idx = pd.MultiIndex.from_product([["Close"], all_tickers])
    df = pd.DataFrame(
        {("Close", t): prices[t] for t in all_tickers},
        index=dates,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    with patch("yfinance.download", return_value=df):
        result = regime_engine.get_top_sectors()

    assert len(result["top_sectors"]) == config.TOP_SECTOR_COUNT
    assert all(t in config.SECTOR_ETFS for t in result["top_sectors"])


# ---------------------------------------------------------------------------
# filter_by_sector()
# ---------------------------------------------------------------------------

def test_filter_by_sector_keeps_matching():
    mock_ticker = MagicMock()
    mock_ticker.info = {"sector": "Technology"}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = regime_engine.filter_by_sector(["NVDA"], ["XLK"])

    assert "NVDA" in result

def test_filter_by_sector_drops_non_matching():
    mock_ticker = MagicMock()
    mock_ticker.info = {"sector": "Technology"}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = regime_engine.filter_by_sector(["NVDA"], ["XLF"])  # Financials, not Tech

    assert result == []


# ---------------------------------------------------------------------------
# apply_individual_gates()
# ---------------------------------------------------------------------------

def test_individual_gates_passes_stable_trend():
    """Stock at 200, above both EMAs, low ATR — should pass."""
    prices = [100.0 + i * 0.01 for i in range(220)]  # gently trending up
    df = _make_ohlcv_df(prices)

    with patch("yfinance.download", return_value=df):
        result = regime_engine.apply_individual_gates(["NVDA"])

    assert len(result) == 1
    assert result[0]["ticker"] == "NVDA"

def test_individual_gates_drops_volatile():
    """Stock with very high ATR relative to price — should be dropped."""
    # Wide candles: high = price * 1.20, low = price * 0.80 → ~40% daily range
    prices = [100.0] * 220
    dates = pd.date_range("2022-01-01", periods=220, freq="B")
    close = np.array(prices)
    df = pd.DataFrame({
        "Close": close,
        "High": close * 1.20,
        "Low": close * 0.80,
        "Volume": np.full(220, 1_000_000),
    }, index=dates)

    with patch("yfinance.download", return_value=df):
        result = regime_engine.apply_individual_gates(["XYZ"])

    assert result == []


# ---------------------------------------------------------------------------
# apply_earnings_gate()
# ---------------------------------------------------------------------------

def test_earnings_gate_drops_near_earnings():
    soon = (datetime.date.today() + datetime.timedelta(days=3)).isoformat()
    earnings_df = pd.DataFrame({
        "ticker": ["NVDA"],
        "next_earnings_date": pd.to_datetime([soon]),
    })
    tickers = [{"ticker": "NVDA", "price": 500.0}]
    with patch("pandas.read_csv", return_value=earnings_df):
        result = regime_engine.apply_earnings_gate(tickers)
    assert result == []

def test_earnings_gate_keeps_far_earnings():
    far = (datetime.date.today() + datetime.timedelta(days=60)).isoformat()
    earnings_df = pd.DataFrame({
        "ticker": ["NVDA"],
        "next_earnings_date": pd.to_datetime([far]),
    })
    tickers = [{"ticker": "NVDA", "price": 500.0}]
    with patch("pandas.read_csv", return_value=earnings_df):
        result = regime_engine.apply_earnings_gate(tickers)
    assert len(result) == 1

def test_earnings_gate_keeps_unknown_ticker():
    """Tickers not in the CSV should be kept (conservative)."""
    tickers = [{"ticker": "UNKNOWN_CO", "price": 50.0}]
    with patch("pandas.read_csv", return_value=pd.DataFrame({"ticker": [], "next_earnings_date": []})):
        result = regime_engine.apply_earnings_gate(tickers)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# apply_sector_leadership_gate()
# ---------------------------------------------------------------------------

def test_leadership_gate_one_per_industry():
    mock_info = {"industry": "Semiconductors"}
    mock_ticker = MagicMock()
    mock_ticker.info = mock_info

    tickers = [
        {"ticker": "NVDA", "price": 500.0, "pre_market_gap_pct": 0.05},
        {"ticker": "AMD",  "price": 120.0, "pre_market_gap_pct": 0.03},
    ]
    with patch("yfinance.Ticker", return_value=mock_ticker):
        result = regime_engine.apply_sector_leadership_gate(tickers)

    # Both in same industry → only the one with higher gap kept
    assert len(result) == 1
    assert result[0]["ticker"] == "NVDA"

def test_leadership_gate_different_industries():
    def side_effect(ticker):
        m = MagicMock()
        m.info = {"industry": "Semiconductors" if ticker == "NVDA" else "Software"}
        return m

    tickers = [
        {"ticker": "NVDA", "price": 500.0, "pre_market_gap_pct": 0.05},
        {"ticker": "MSFT", "price": 400.0, "pre_market_gap_pct": 0.03},
    ]
    with patch("yfinance.Ticker", side_effect=side_effect):
        result = regime_engine.apply_sector_leadership_gate(tickers)

    assert len(result) == 2
