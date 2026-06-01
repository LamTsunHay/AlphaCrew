# Finnhub Switchable News Provider Design
**Date:** 2026-06-01
**Feature:** Add Finnhub as a switchable alternative to Polygon.io for news fetching

---

## Overview

Add `fetch_finnhub_news()` to `pipeline.py` alongside the existing `fetch_polygon_news()`. A new `fetch_news()` dispatcher reads `config.NEWS_PROVIDER` and calls the right fetcher. `run_pipeline()` calls `fetch_news()` instead of `fetch_polygon_news()` directly. The switch is a single constant in `config.py`.

---

## Changes

| File | Action | Detail |
|---|---|---|
| `config.py` | Modify | Add `NEWS_PROVIDER` and `FINNHUB_API_KEY` constants |
| `pipeline.py` | Modify | Add `fetch_finnhub_news()`, `fetch_news()` dispatcher; update `run_pipeline()` call |

No new files. No changes to any other module.

---

## `config.py` additions

```python
# News provider — "polygon" or "finnhub"
NEWS_PROVIDER = "polygon"
FINNHUB_API_KEY = ""  # set in .env as FINNHUB_API_KEY
```

`NEWS_PROVIDER` is the single toggle. `FINNHUB_API_KEY` is read at runtime via `os.environ.get("FINNHUB_API_KEY", config.FINNHUB_API_KEY)`, consistent with how `POLYGON_API_KEY` is handled.

---

## Finnhub Endpoint

```
GET https://finnhub.io/api/v1/company-news
  ?symbol=NVDA
  &from=YYYY-MM-DD    (today - 3 days)
  &to=YYYY-MM-DD      (today)
  &token=<FINNHUB_API_KEY>
```

Free tier: 60 requests/minute. No date range restriction on free plan for company news.

---

## Field Mapping

| Finnhub field | Maps to | Notes |
|---|---|---|
| `headline` | `title` | Used by `classify_catalyst_type` and `extract_eass_inputs` |
| `summary` | `description` | Used by `extract_eass_inputs` EASS regex — must be full summary |
| `datetime` (unix int) | `published_utc` | Converted to ISO string; used for logging only |
| `related` | `tickers` | Used for logging only |
| *(none)* | `keywords` | Empty list — not used downstream |

The normalized article shape is identical to what `fetch_polygon_news` returns, so `classify_catalyst_type` and `extract_eass_inputs` need no changes.

---

## `pipeline.py` additions

### `fetch_finnhub_news(ticker, session)`
Calls the Finnhub company-news endpoint with a 3-day lookback window. Maps response fields to the standard article shape. On error: logs and returns `[]`.

### `fetch_news(ticker, session)` — dispatcher
```python
if config.NEWS_PROVIDER == "polygon":
    return await fetch_polygon_news(ticker, session)
elif config.NEWS_PROVIDER == "finnhub":
    return await fetch_finnhub_news(ticker, session)
else:
    raise ValueError(f"Unknown NEWS_PROVIDER: {config.NEWS_PROVIDER!r}. Use 'polygon' or 'finnhub'.")
```

### `run_pipeline()` — one-line change
Replace `fetch_polygon_news(ticker, session)` with `fetch_news(ticker, session)`.

---

## Error Handling

| Scenario | Behavior |
|---|---|
| Invalid `NEWS_PROVIDER` value | `ValueError` raised immediately at dispatch — fail fast, clear message |
| Finnhub API error / timeout | Print error, return `[]` — ticker skipped as `NO_CATALYST_FOUND` |
| Missing `FINNHUB_API_KEY` | Finnhub returns 401; caught by error handler, returns `[]` |

---

## Function Comments

Each new function gets a single-line docstring per project convention.

---

## Out of Scope

- `run.py` CLI flag for news provider — provider is config-only
- More than two providers
- Caching or fallback (e.g., try Finnhub, fall back to Polygon)
