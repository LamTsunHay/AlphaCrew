# Groq Testing Mode Design

**Date:** 2026-06-02  
**Status:** Approved  

## Goal

Add a `TESTING_MODE` flag that routes both LLM stages (news summarization + risk audit) to Groq's free-tier API instead of Claude, so the full pipeline can be exercised without incurring Anthropic API costs.

## Scope

Files changed: `config.py`, `pipeline.py`, `risk_auditor.py`, `scheduler.py`  
New file: `llm_client.py`  
No changes to: `regime_engine.py`, `database.py`, ChromaDB collections, news providers, or gate logic.

---

## Section 1: Config (`config.py`)

Add below the existing LLM block:

```python
# Testing mode — set True to route both LLM stages to Groq (free tier)
TESTING_MODE        = False
GROQ_API_KEY        = ""   # set in .env as GROQ_API_KEY
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
GROQ_STAGE_3_MODEL  = "llama-3.1-8b-instant"     # replaces Haiku
GROQ_STAGE_4_MODEL  = "llama-3.3-70b-versatile"  # replaces Sonnet
```

`TESTING_MODE = False` is the default. Flipping it to `True` activates Groq for both stages. All Groq values are co-located here for easy discovery.

---

## Section 2: Client Factory (`llm_client.py`, new file)

A thin module with two responsibilities:

1. **`create_client()`** — returns `(client, provider)` tuple. Provider is `"anthropic"` or `"groq"`. Reads API keys from environment first, falls back to config values.
2. **`chat(client, provider, model, system, user, max_tokens)`** — unified call interface. Translates between `anthropic.messages.create()` and `openai.chat.completions.create()` (Groq's OpenAI-compatible endpoint). Returns plain response text in both cases.

Callers never import `anthropic` or `openai` directly — all SDK details are contained here.

**Dependency added:** `openai` package (already OpenAI-compatible, used only when `TESTING_MODE = True`).

---

## Section 3: Call Site Changes

### `pipeline.py`
- `run_pipeline()`: replace `anthropic.Anthropic(...)` with `llm_client.create_client()` to get `(client, provider)`.
- `summarize_news_haiku(raw_text, ticker, client)`: add `provider` parameter. Replace `client.messages.create(...)` with `llm_client.chat(...)`. Model resolved as:
  - `GROQ_STAGE_3_MODEL` when `provider == "groq"`
  - `LLM_STAGE_3_FAST` otherwise

### `risk_auditor.py`
- `run_sonnet_audit(candidate, client)`: add `provider` parameter. Replace `client.messages.create(...)` with `llm_client.chat(...)`. Model resolved as:
  - `GROQ_STAGE_4_MODEL` when `provider == "groq"`
  - `LLM_STAGE_4_PREMIUM` otherwise

### `scheduler.py`
- Call `llm_client.create_client()` once at pipeline entry point.
- Pass both `client` and `provider` into `run_pipeline()` and `run_sonnet_audit()`.

---

## Data Flow

```
TESTING_MODE=True
    └── llm_client.create_client()
            └── openai.OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
                    ├── pipeline.py → llm_client.chat(..., GROQ_STAGE_3_MODEL, ...)
                    └── risk_auditor.py → llm_client.chat(..., GROQ_STAGE_4_MODEL, ...)

TESTING_MODE=False (default)
    └── llm_client.create_client()
            └── anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                    ├── pipeline.py → llm_client.chat(..., LLM_STAGE_3_FAST, ...)
                    └── risk_auditor.py → llm_client.chat(..., LLM_STAGE_4_PREMIUM, ...)
```

---

## Error Handling

No new error handling required. Both SDKs raise exceptions on API failure; existing try/except blocks in `pipeline.py` and `risk_auditor.py` already catch these and return safe defaults.

---

## Testing

1. Set `TESTING_MODE = True` and `GROQ_API_KEY=<key>` in `.env`.
2. Run the import check: `python -c "import config; import llm_client; print('OK')"`.
3. Run the full pipeline with a single ticker to confirm Groq responses flow through both stages.
4. Set `TESTING_MODE = False` and confirm Claude is restored.

---

## Groq Free Tier Limits

| Limit | Value |
|---|---|
| Requests/day | 14,400 |
| Tokens/minute | 500,000 (llama-3.1-8b), 100,000 (llama-3.3-70b) |
| Cost | Free |

Sufficient for daily pre-market runs over ~8–20 tickers.
