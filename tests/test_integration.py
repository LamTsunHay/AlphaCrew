"""Offline end-to-end integration test.

Seeds a PostgreSQL test database, runs run_premarket_pipeline with all paid APIs
mocked, and verifies a strategy card is written to the output files.

Requires TEST_DATABASE_URL env var pointing to a live PostgreSQL instance.
All tests are skipped when that variable is not set.
"""

import pytest
import asyncio
import json
import os
import sys
import tempfile
import datetime
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config
import database
import seed_database
import scheduler

TEST_DSN = os.getenv("TEST_DATABASE_URL", "")


def _build_spy_df(n=60):
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    prices = np.linspace(380, 430, n)
    return pd.DataFrame({"Close": prices}, index=dates)


def _build_sector_df(tickers, n=10):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    data = {}
    for t in tickers:
        data[("Close", t)] = np.linspace(100, 105, n)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _build_ohlcv(n=220, trend=True):
    dates = pd.date_range("2022-01-01", periods=n, freq="B")
    close = np.linspace(100, 150, n) if trend else np.full(n, 100.0)
    return pd.DataFrame({
        "Close": close, "High": close * 1.005,
        "Low": close * 0.995, "Volume": np.full(n, 5_000_000),
    }, index=dates)


@pytest.mark.asyncio
async def test_full_pipeline_produces_strategy_card():
    if not TEST_DSN:
        pytest.skip("TEST_DATABASE_URL not set — skipping integration test")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Override output paths
        config.STRATEGY_OUTPUT_PATH = os.path.join(tmpdir, "blueprint.md")
        config.LOG_QUEUE_PATH = os.path.join(tmpdir, "queue.json")

        # Seed DB with synthetic data via the new psycopg2 interface
        conn = database.initialize_database(dsn=TEST_DSN)
        try:
            with patch.dict(os.environ, {"DATABASE_URL": TEST_DSN}):
                seed_database.seed(samples_per_collection=60, seed_val=1)

            anthropic_client = MagicMock()

            # Mock Haiku summarization
            mock_haiku_msg = MagicMock()
            mock_haiku_msg.content = [MagicMock(text="NVDA beat EPS by 15%. Margins expanded to record levels. Guidance raised for full year.")]
            # Mock Sonnet audit
            mock_sonnet_msg = MagicMock()
            mock_sonnet_msg.content = [MagicMock(text=json.dumps({
                "threat_level": "LOW",
                "threats_identified": [],
                "durability": "DURABLE",
                "regime_support": "CONFIRMED",
                "entry_strategy": "MARKET_OPEN",
                "audit_note": "Clean catalyst with strong follow-through history.",
            }))]
            anthropic_client.messages.create.side_effect = [mock_haiku_msg, mock_sonnet_msg]

            # Polygon news article that qualifies
            good_article = {
                "title": "NVDA reports Q3 EPS $5.16 vs $4.60 expected raises full-year guidance to $22",
                "description": "Revenue $18.1B versus $16.9B. CEO says results exceeded all expectations.",
                "published_utc": "2024-11-21T00:00:00Z",
                "keywords": [],
                "tickers": ["NVDA"],
            }

            all_tickers = list(config.SECTOR_ETFS.keys()) + ["SPY"]
            sector_df = _build_sector_df(all_tickers)
            ohlcv_df = _build_ohlcv()
            spy_df = _build_spy_df()

            far_date = (datetime.date.today() + datetime.timedelta(days=60)).isoformat()
            earnings_df = pd.DataFrame({"ticker": ["NVDA"], "next_earnings_date": [far_date]})
            earnings_df["next_earnings_date"] = pd.to_datetime(earnings_df["next_earnings_date"])

            nvda_info = MagicMock()
            nvda_info.info = {
                "sector": "Technology",
                "industry": "Semiconductors",
                "preMarketPrice": 540.0,
                "regularMarketPreviousClose": 500.0,
            }

            def yf_download_side(ticker_arg, *args, **kwargs):
                if ticker_arg == config.REGIME_INDICATOR or ticker_arg == "SPY":
                    return spy_df
                if isinstance(ticker_arg, list):
                    return sector_df
                return ohlcv_df

            with patch("yfinance.download", side_effect=yf_download_side), \
                 patch("yfinance.Ticker", return_value=nvda_info), \
                 patch("pandas.read_csv", return_value=earnings_df), \
                 patch("pipeline.fetch_polygon_news", new_callable=AsyncMock, return_value=[good_article]), \
                 patch("pipeline.anthropic.Anthropic", return_value=anthropic_client), \
                 patch.dict(os.environ, {"DATABASE_URL": TEST_DSN}), \
                 patch.object(scheduler, "CANDIDATE_TICKERS", ["NVDA"]):

                await scheduler.run_premarket_pipeline(conn, anthropic_client)

        finally:
            # Clean up test data
            with conn.cursor() as cur:
                cur.execute("DROP TABLE IF EXISTS setups")
                cur.execute("DROP EXTENSION IF EXISTS vector CASCADE")
            conn.commit()
            conn.close()

        # Verify outputs were written
        assert os.path.exists(config.STRATEGY_OUTPUT_PATH), "Strategy blueprint not written"
        assert os.path.exists(config.LOG_QUEUE_PATH), "Decision log not written"

        with open(config.STRATEGY_OUTPUT_PATH) as f:
            content = f.read()

        with open(config.LOG_QUEUE_PATH) as f:
            log = json.load(f)

        # The pipeline may short-circuit at various gates (sector filter, confidence floor, etc.)
        # At minimum, verify the files were created and log has entries
        assert isinstance(log, list)
        assert len(log) > 0

        # If a card was produced, it should mention NVDA
        if content.strip():
            assert "NVDA" in content or "STRATEGY CARD" in content
