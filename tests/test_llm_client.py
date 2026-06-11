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


def test_create_client_returns_gemini_in_testing_mode(monkeypatch):
    """With TESTING_MODE=True, create_client() returns a Gemini client and 'gemini'."""
    monkeypatch.setattr(config, "TESTING_MODE", True)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-gemini-key")
    import importlib, llm_client
    importlib.reload(llm_client)
    client, provider = llm_client.create_client()
    assert provider == "gemini"
    from google import genai
    assert isinstance(client, genai.Client)


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


def test_chat_gemini_path():
    """chat() with 'gemini' calls client.models.generate_content() and returns text."""
    import llm_client
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "gemini result"
    mock_client.models.generate_content.return_value = mock_response

    result = llm_client.chat(
        mock_client, "gemini", "gemini-2.0-flash-lite", "sys prompt", "user input", 100
    )

    assert result == "gemini result"
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-2.0-flash-lite"
    assert call_kwargs["contents"] == "user input"
    assert call_kwargs["config"].system_instruction == "sys prompt"
    assert call_kwargs["config"].max_output_tokens == 100
