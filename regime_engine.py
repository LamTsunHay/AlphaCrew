"""Macro and sector alpha filters for Strategy Engine v5.1.

Free data only (yfinance + pandas). No paid API calls anywhere in this module.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import config


def _close_series(df, ticker=None):
    """Extract a 1-D close series from yfinance download output.

    yfinance 1.4+ returns a MultiIndex DataFrame for multi-ticker downloads
    and a flat DataFrame for single-ticker downloads.
    """
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"][ticker].dropna()
    return df["Close"].dropna()


def get_market_regime() -> dict:
    """Return current SPY macro regime vs. 50-day SMA."""
    df = yf.download(config.REGIME_INDICATOR, period="60d", progress=False, auto_adjust=True)
    close = _close_series(df, config.REGIME_INDICATOR)
    sma50 = close.rolling(config.REGIME_SMA_PERIOD).mean()

    latest_close = float(close.iloc[-1])
    latest_sma = float(sma50.iloc[-1])
    pct_above = (latest_close - latest_sma) / latest_sma

    return {
        "regime": "BULLISH" if latest_close > latest_sma else "BEARISH",
        "spy_close": latest_close,
        "spy_sma50": latest_sma,
        "pct_above_50sma": pct_above,
    }


def get_top_sectors() -> dict:
    """Rank sector ETFs by 5-day return relative to SPY."""
    all_tickers = list(config.SECTOR_ETFS.keys()) + ["SPY"]
    df = yf.download(all_tickers, period="10d", progress=False, auto_adjust=True)

    spy_close = _close_series(df, "SPY")
    spy_5d = float((spy_close.iloc[-1] - spy_close.iloc[-6]) / spy_close.iloc[-6])

    sector_rankings = {}
    for etf in config.SECTOR_ETFS:
        try:
            s = _close_series(df, etf)
            ret = float((s.iloc[-1] - s.iloc[-6]) / s.iloc[-6])
            sector_rankings[etf] = round(ret - spy_5d, 6)
        except Exception:
            sector_rankings[etf] = -999.0

    sorted_sectors = sorted(sector_rankings.items(), key=lambda x: x[1], reverse=True)
    top_sectors = [etf for etf, _ in sorted_sectors[: config.TOP_SECTOR_COUNT]]

    return {
        "top_sectors": top_sectors,
        "sector_rankings": dict(sorted_sectors),
        "spy_5d_return": round(spy_5d, 6),
    }


# Maps sector name (from yf.Ticker.info["sector"]) to its ETF ticker
_SECTOR_NAME_TO_ETF = {v: k for k, v in config.SECTOR_ETFS.items()}


def filter_by_sector(tickers: list, top_sectors: list) -> list:
    """Retain only tickers whose sector ETF is in the top-performing sectors."""
    survivors = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            sector_name = info.get("sector", "")
            etf = _SECTOR_NAME_TO_ETF.get(sector_name)
            if etf and etf in top_sectors:
                survivors.append(ticker)
        except Exception:
            pass  # skip on any data error
    return survivors


def apply_individual_gates(tickers: list) -> list:
    """Apply volatility and trend gates using 220 days of daily price data."""
    survivors = []
    for ticker in tickers:
        try:
            df = yf.download(ticker, period="220d", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                high = df["High"][ticker].dropna()
                low = df["Low"][ticker].dropna()
                close = df["Close"][ticker].dropna()
            else:
                high = df["High"].dropna()
                low = df["Low"].dropna()
                close = df["Close"].dropna()

            if len(close) < 210:
                continue

            atr14 = (high - low).rolling(14).mean().iloc[-1]
            price = float(close.iloc[-1])
            volatility_ratio = float(atr14) / price

            ema21 = float(close.ewm(span=config.TREND_EMA_FAST, adjust=False).mean().iloc[-1])
            ema200 = float(close.ewm(span=config.TREND_EMA_SLOW, adjust=False).mean().iloc[-1])

            # Volatility gate
            if volatility_ratio > config.VOLATILITY_CEILING_RATIO:
                continue
            # Trend gate
            if price <= ema200 or price <= ema21:
                continue

            survivors.append({
                "ticker": ticker,
                "price": price,
                "atr14": float(atr14),
                "volatility_ratio": round(volatility_ratio, 6),
                "ema21": round(ema21, 4),
                "ema200": round(ema200, 4),
            })
        except Exception:
            pass
    return survivors


def apply_earnings_gate(tickers_with_metrics: list) -> list:
    """Drop tickers with earnings within EARNINGS_EXCLUSION_DAYS calendar days."""
    try:
        cal = pd.read_csv("earnings_calendar.csv", parse_dates=["next_earnings_date"])
        cal_map = {
            row["ticker"]: row["next_earnings_date"].date()
            for _, row in cal.iterrows()
        }
    except Exception:
        # If CSV missing or unreadable, keep all tickers (conservative)
        return tickers_with_metrics

    today = datetime.date.today()
    survivors = []
    for entry in tickers_with_metrics:
        ticker = entry["ticker"]
        earnings_date = cal_map.get(ticker)
        if earnings_date is None:
            survivors.append(entry)  # not in CSV → keep
            continue
        days_away = (earnings_date - today).days
        if abs(days_away) <= config.EARNINGS_EXCLUSION_DAYS:
            continue
        survivors.append(entry)
    return survivors


def apply_sector_leadership_gate(tickers_with_metrics: list) -> list:
    """Retain the highest pre-market gap ticker per sub-industry."""
    by_industry: dict[str, dict] = {}
    for entry in tickers_with_metrics:
        ticker = entry["ticker"]
        try:
            info = yf.Ticker(ticker).info
            industry = info.get("industry", "Unknown")
        except Exception:
            industry = "Unknown"

        gap = entry.get("pre_market_gap_pct", 0.0) or 0.0
        if industry not in by_industry or gap > by_industry[industry].get("pre_market_gap_pct", 0.0):
            entry["industry"] = industry
            by_industry[industry] = entry

    return list(by_industry.values())
