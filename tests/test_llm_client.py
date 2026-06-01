"""Tests for llm_client.py — client factory and unified chat interface."""

import pytest
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import config


# ---------------------------------------------------------------------------
# create_client()
# ---------------------------------------------------------------------------

def test_create_client_returns_anthropic_by_default(monkeypatch):
    """With TESTING_MODE=False, create_client() returns an Anthropic client and 'anthropic'."""
    monkeypatch.setattr(config, "TESTING_MODE", False)
    import importlib, llm_client
    importlib.reload(llm_client)
    client, provider = llm_client.create_client()
    assert provider == "anthropic"
    import anthropic
    assert isinstance(client, anthropic.Anthropic)


def test_create_client_returns_groq_in_testing_mode(monkeypatch):
    """With TESTING_MODE=True, create_client() returns an OpenAI client and 'groq'."""
    monkeypatch.setattr(config, "TESTING_MODE", True)
    monkeypatch.setattr(config, "GROQ_API_KEY", "test-key")
    import importlib, llm_client
    importlib.reload(llm_client)
    client, provider = llm_client.create_client()
    assert provider == "groq"
    from openai import OpenAI
    assert isinstance(client, OpenAI)


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------

def test_chat_anthropic_path():
    """chat() with 'anthropic' calls client.messages.create() and returns text."""
    import llm_client
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.content = [MagicMock(text="anthropic result")]
    mock_client.messages.create.return_value = mock_msg

    result = llm_client.chat(
        mock_client, "anthropic", "claude-haiku-4-5-20251001", "sys prompt", "user input", 100
    )

    assert result == "anthropic result"
    mock_client.messages.create.assert_called_once_with(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        system="sys prompt",
        messages=[{"role": "user", "content": "user input"}],
    )


def test_chat_groq_path():
    """chat() with 'groq' calls client.chat.completions.create() and returns text."""
    import llm_client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="groq result"))]
    mock_client.chat.completions.create.return_value = mock_response

    result = llm_client.chat(
        mock_client, "groq", "llama-3.1-8b-instant", "sys prompt", "user input", 100
    )

    assert result == "groq result"
    mock_client.chat.completions.create.assert_called_once_with(
        model="llama-3.1-8b-instant",
        max_tokens=100,
        messages=[
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "user input"},
        ],
    )
