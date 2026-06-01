"""Unified LLM client factory for Anthropic and Groq (OpenAI-compatible) providers.

All LLM SDK details are contained here. Callers never import anthropic or openai directly.
"""

import os
import anthropic
import config


def create_client():
    """Return (client, provider) for the active LLM provider.

    Provider is 'anthropic' when TESTING_MODE is False, 'groq' when True.
    API keys are read from environment first, then fall back to config values.
    """
    if config.TESTING_MODE:
        # Lazy import: avoids loading openai SDK when testing mode is inactive
        from openai import OpenAI
        api_key = os.environ.get("GROQ_API_KEY", config.GROQ_API_KEY)
        client = OpenAI(base_url=config.GROQ_BASE_URL, api_key=api_key)
        return client, "groq"
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=api_key)
    return client, "anthropic"


def chat(client, provider: str, model: str, system: str, user: str, max_tokens: int) -> str:
    """Send a single-turn chat request and return the response text.

    Translates between anthropic.messages.create() and openai.chat.completions.create()
    so callers are insulated from SDK differences.
    """
    if provider == "groq":
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text
