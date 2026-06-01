"""Tests for notifier.py — terminal output and Telegram stub."""

import os
import sys
import pytest
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import notifier


SAMPLE_CARDS = [
    "## STRATEGY CARD — NVDA — 2026-06-01 08:05 EST\n\n### Catalyst\n- Type: earnings_beat_large\n- EASS Score: 5.5 / 10.0\n\n---"
]
SAMPLE_LOG = [
    {"step": "MACRO_GATE", "result": {"regime": "BULLISH"}},
    {"ticker": "NVDA", "status": "QUALIFIED", "eass_score": 5.5},
]


def test_print_to_terminal_outputs_cards(capsys):
    """Strategy cards are printed to stdout."""
    notifier.print_to_terminal(SAMPLE_CARDS, SAMPLE_LOG)
    out = capsys.readouterr().out
    assert "STRATEGY CARD" in out
    assert "NVDA" in out


def test_print_to_terminal_no_candidates(capsys):
    """Empty card list prints NO QUALIFIED CANDIDATES message."""
    notifier.print_to_terminal([], SAMPLE_LOG)
    out = capsys.readouterr().out
    assert "NO QUALIFIED CANDIDATES" in out


def test_print_to_terminal_shows_gate_summary(capsys):
    """Gate summary from log_entries is printed."""
    notifier.print_to_terminal(SAMPLE_CARDS, SAMPLE_LOG)
    out = capsys.readouterr().out
    assert "MACRO_GATE" in out


def test_send_telegram_is_stub():
    """send_telegram does nothing and returns None (stub)."""
    result = notifier.send_telegram(SAMPLE_CARDS, SAMPLE_LOG)
    assert result is None
