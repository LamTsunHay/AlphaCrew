"""Tests for api.py — POST /run endpoint validation and response shape."""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import api

client = TestClient(api.app)

STUB_CARDS = [{"ticker": "NVDA", "strategy": "LONG"}]
STUB_LOG = [{"step": "MACRO_GATE", "result": {"regime": "BULLISH"}}]


def _mock_main(cards=None, log=None):
    """Return an AsyncMock that resolves to (cards, log)."""
    cards = cards if cards is not None else STUB_CARDS
    log = log if log is not None else STUB_LOG
    return AsyncMock(return_value=(cards, log))


def test_run_valid_ticker_and_regime():
    """Single ticker + regime override returns strategy_cards and log_entries."""
    with patch("run.main", _mock_main()) as mock:
        resp = client.post("/run", json={"tickers": ["NVDA"], "regime": "BULLISH"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy_cards"] == STUB_CARDS
    assert body["log_entries"] == STUB_LOG


def test_run_defaults_to_nasdaq100():
    """Omitting tickers passes None to run.main so NASDAQ 100 is used."""
    with patch("run.main", _mock_main()) as mock:
        resp = client.post("/run", json={})
    assert resp.status_code == 200
    # argv=None is passed when no tickers — verify via mock call args
    mock.assert_awaited_once_with(None, test_mode=True)


def test_run_invalid_regime_returns_422():
    """Invalid regime value is rejected by Pydantic before reaching the handler."""
    resp = client.post("/run", json={"regime": "SIDEWAYS"})
    assert resp.status_code == 422


def test_run_no_candidates_returns_empty_list():
    """Pipeline returning zero candidates still produces HTTP 200 with empty cards."""
    with patch("run.main", _mock_main(cards=[], log=STUB_LOG)):
        resp = client.post("/run", json={"tickers": ["XYZ"]})
    assert resp.status_code == 200
    assert resp.json()["strategy_cards"] == []


def test_run_test_mode_false_passes_through():
    """test_mode=False is forwarded to run.main as test_mode=False (Claude LLM)."""
    with patch("run.main", _mock_main()) as mock:
        resp = client.post("/run", json={"tickers": ["AAPL"], "test_mode": False})
    assert resp.status_code == 200
    mock.assert_awaited_once_with(["AAPL"], test_mode=False)


def test_run_test_mode_defaults_to_true():
    """Omitting test_mode defaults to True (Gemini)."""
    with patch("run.main", _mock_main()) as mock:
        resp = client.post("/run", json={"tickers": ["AAPL"]})
    assert resp.status_code == 200
    mock.assert_awaited_once_with(["AAPL"], test_mode=True)


def test_run_pipeline_exception_returns_500():
    """Unexpected pipeline exception surfaces as HTTP 500."""
    boom = AsyncMock(side_effect=RuntimeError("exploded"))
    with patch("run.main", boom):
        resp = client.post("/run", json={"tickers": ["NVDA"]})
    assert resp.status_code == 500
    assert "exploded" in resp.json()["detail"]
