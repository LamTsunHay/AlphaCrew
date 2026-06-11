"""Unified LLM client factory for Anthropic and Gemini providers.

All LLM SDK details are contained here. Callers never import anthropic or google.genai directly.
"""

import os
import anthropic
import config


def create_client(test_mode: bool | None = None):
    """Return (client, provider) for the active LLM provider.

    Provider is 'anthropic' when test_mode (or config.TESTING_MODE) is False,
    'gemini' when True. Explicit test_mode arg takes precedence over config.
    """
    use_gemini = config.TESTING_MODE if test_mode is None else test_mode
    if use_gemini:
        # Lazy import: avoids loading google-genai SDK when testing mode is inactive
        from google import genai
        client = genai.Client(api_key=config.GEMINI_API_KEY)
        return client, "gemini"
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=api_key)
    return client, "anthropic"


def chat(client, provider: str, model: str, system: str, user: str, max_tokens: int) -> str:
    """Send a single-turn chat request and return the response text.

    Translates between anthropic.messages.create() and google.genai generate_content()
    so callers are insulated from SDK differences.
    """
    if provider == "gemini":
        from google.genai import types
        response = client.models.generate_content(
            model=model,
            contents=user,
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=max_tokens,
            ),
        )
        return response.text
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return message.content[0].text
