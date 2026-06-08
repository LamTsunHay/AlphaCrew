"""EASS calculation, Polygon.io news gate, and pipeline coordinator.

This is the paid-API gate. Nothing in this module is called until all free
gates in regime_engine have passed.
"""

import aiohttp
import asyncio
import llm_client
import json
import re
import datetime
import os
import config
import database


async def fetch_polygon_news(ticker: str, session: aiohttp.ClientSession) -> list:
    """Fetch recent news articles from Polygon.io for a ticker."""
    url = "https://api.polygon.io/v2/reference/news"
    params = {
        "ticker": ticker,
        "limit": 10,
        "order": "desc",
        "sort": "published_utc",
        "apiKey": os.environ.get("POLYGON_API_KEY", ""),
    }
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            data = await resp.json()
            articles = data.get("results", [])
            return [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "published_utc": a.get("published_utc", ""),
                    "keywords": a.get("keywords", []),
                    "tickers": a.get("tickers", []),
                }
                for a in articles
            ]
    except Exception as exc:
        print(f"[PIPELINE] Polygon fetch error for {ticker}: {exc}")
        return []


async def fetch_finnhub_news(ticker: str, session: aiohttp.ClientSession) -> list:
    """Fetch recent news articles from Finnhub for a ticker using a 3-day lookback."""
    today = datetime.date.today()
    from_date = (today - datetime.timedelta(days=3)).isoformat()
    to_date = today.isoformat()
    url = config.FINNHUB_NEWS_URL
    params = {
        "symbol": ticker,
        "from": from_date,
        "to": to_date,
        "token": os.environ.get("FINNHUB_API_KEY", config.FINNHUB_API_KEY),
    }
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            articles = await resp.json()
            return [
                {
                    "title": a.get("headline", ""),
                    "description": a.get("summary", ""),
                    "published_utc": datetime.datetime.utcfromtimestamp(a["datetime"]).isoformat() if a.get("datetime") else "",
                    "keywords": [],
                    "tickers": [a.get("related", "")] if a.get("related") else [],
                }
                for a in articles
            ]
    except Exception as exc:
        print(f"[PIPELINE] Finnhub fetch error for {ticker}: {exc}")
        return []


async def fetch_news(ticker: str, session: aiohttp.ClientSession) -> list:
    """Dispatch news fetch to the configured provider (polygon or finnhub)."""
    if config.NEWS_PROVIDER == "polygon":
        return await fetch_polygon_news(ticker, session)
    elif config.NEWS_PROVIDER == "finnhub":
        return await fetch_finnhub_news(ticker, session)
    else:
        raise ValueError(f"Unknown NEWS_PROVIDER: {config.NEWS_PROVIDER!r}. Use 'polygon' or 'finnhub'.")


def classify_catalyst_type(article: dict) -> str:
    """Classify the dominant catalyst type from an article using keyword priority order."""
    text = (article.get("title", "") + " " + article.get("description", "")).lower()

    if "acquisition" in text or "merger" in text or " acquires " in text or " acquiring " in text or "to acquire" in text:
        return "ma_acquirer"
    if "acquired" in text or "takeover" in text:
        return "ma_target"
    if "fda" in text and "approved" in text:
        return "fda_approval_nda"
    if "fda" in text and "fast track" in text:
        return "fda_approval_fast_track"
    if "fda" in text and ("reject" in text or "complete response" in text):
        return "fda_rejection"
    if "buyback" in text or "repurchase" in text:
        return "buyback_initiation"
    if "contract" in text and (
        "department of defense" in text or "government" in text or "federal" in text
    ):
        return "government_contract"
    if "contract" in text:
        return "commercial_contract"
    if "guidance" in text and ("raise" in text or "increase" in text or "above" in text):
        return "guidance_raise_full"
    if "guidance" in text and ("cut" in text or "lower" in text or "below" in text):
        return "guidance_cut"
    if "earnings" in text or "eps" in text or "beat" in text:
        return "earnings_beat_large"

    return "market_movers"


def select_best_catalyst(articles: list) -> str:
    """Return the highest-priority catalyst type found across all articles.

    Classifies each article independently via classify_catalyst_type, then
    selects the winner by CATALYST_PRIORITY rank. For an empty list, returns
    'market_movers' directly; for all-generic articles, the loop returns
    'market_movers' as the last entry in CATALYST_PRIORITY.
    """
    found = {classify_catalyst_type(a) for a in articles}
    for catalyst in config.CATALYST_PRIORITY:
        if catalyst in found:
            return catalyst
    return "market_movers"


def classify_and_summarize(articles: list, ticker: str, client, provider: str) -> dict | None:
    """Classify highest-impact catalyst and generate a 3-sentence summary in one Haiku call.

    Sends all articles with a priority-ranked catalyst list so the LLM selects the
    highest-rank catalyst across all articles and summarizes it in one pass.
    Falls back to select_best_catalyst() if the LLM call raises an exception.
    Returns None if articles is empty or the LLM response cannot be parsed.
    """
    if not articles:
        return None

    ranked_catalysts = "\n".join(
        f"{i + 1:2}. {cat}"
        for i, cat in enumerate(config.CATALYST_PRIORITY)
    )

    articles_text = "\n".join(
        f"[{i}] {a.get('title', '')} | {a.get('description', '')} | {a.get('published_utc', '')}"
        for i, a in enumerate(articles)
    )

    system_prompt = (
        "You are a financial catalyst analyst. Given news articles for a ticker, "
        "identify the single highest-impact catalyst article.\n\n"
        f"Catalyst types ranked by market impact (1 = highest):\n{ranked_catalysts}\n\n"
        "Rules:\n"
        "- Select the catalyst type with the lowest rank number present across any article.\n"
        "- If multiple articles match the same top-rank type, prefer the one with the clearest data.\n"
        "- Return ONLY valid JSON with no markdown or explanation:\n"
        '{"catalyst_type": "<exact type from list>", "article_index": <0-based int>, '
        '"summary": "<3 sentences: what happened, numerical magnitude, forward implication>", '
        '"reasoning": "<one sentence>"}'
    )

    user_prompt = f"Ticker: {ticker}\n\nArticles:\n{articles_text[:4000]}"
    model = config.GROQ_STAGE_3_MODEL if provider == "groq" else config.LLM_STAGE_3_FAST

    try:
        raw = llm_client.chat(client, provider, model, system_prompt, user_prompt, config.LLM_MAX_TOKENS)
    except Exception as exc:
        print(f"[PIPELINE] {ticker}: LLM_CLASSIFY_ERROR ({exc}) — falling back to regex")
        catalyst_type = select_best_catalyst(articles)
        winning = next((a for a in articles if classify_catalyst_type(a) == catalyst_type), articles[0])
        return {"catalyst_type": catalyst_type, "winning_article": winning, "summary": "", "reasoning": ""}

    json_match = re.search(r'\{.*\}', raw, re.DOTALL)
    if not json_match:
        return None
    try:
        result = json.loads(json_match.group())
    except (json.JSONDecodeError, TypeError):
        return None

    if result.get("catalyst_type") not in config.CATALYST_PRIORITY:
        result["catalyst_type"] = "market_movers"

    idx = result.get("article_index", 0)
    if not isinstance(idx, int) or idx < 0 or idx >= len(articles):
        idx = 0
    result["winning_article"] = articles[idx]

    return result


def extract_eass_inputs(articles: list, ticker: str) -> dict:
    """Parse article text to extract numerical EASS inputs via regex."""
    combined = " ".join(
        a.get("title", "") + " " + a.get("description", "")
        for a in articles
    )
    text = combined.lower()

    actual_eps = consensus_eps = actual_revenue = consensus_revenue = None
    guidance_raised = guidance_cut = False
    guidance_delta_pct = 0.0

    # EPS surprise pattern: "$X vs $Y expected/estimate/consensus"
    eps_match = re.search(
        r'\$?([\d.]+)\s*(?:vs\.?|versus)\s*\$?([\d.]+)\s*(?:expected|estimate|consensus)',
        text
    )
    if eps_match:
        actual_eps = float(eps_match.group(1))
        consensus_eps = float(eps_match.group(2))

    # Revenue surprise pattern: "$XB vs $YB"
    rev_match = re.search(
        r'\$([\d.]+)([bm])\s*(?:vs\.?|versus)\s*\$([\d.]+)([bm])',
        text
    )
    if rev_match:
        def to_billions(val, unit):
            return float(val) if unit == "b" else float(val) / 1000
        actual_revenue = to_billions(rev_match.group(1), rev_match.group(2))
        consensus_revenue = to_billions(rev_match.group(3), rev_match.group(4))

    # Guidance raise pattern
    guidance_raise_match = re.search(
        r'(?:raises?|increases?)\s*(?:full.year|fiscal)\s*(?:guidance|outlook)\s*to\s*\$?([\d.]+)',
        text
    )
    if guidance_raise_match:
        guidance_raised = True
        try:
            new_val = float(guidance_raise_match.group(1))
            guidance_delta_pct = 0.05  # default positive delta when exact prior not available
        except ValueError:
            guidance_delta_pct = 0.05

    if "guidance" in text and ("cut" in text or "lower" in text or "reduces" in text):
        guidance_cut = True
        if not guidance_raised:
            guidance_delta_pct = -0.10

    ceo_positive = any(w in text for w in ["record", "strong", "exceeded"])

    return {
        "actual_eps": actual_eps,
        "consensus_eps": consensus_eps,
        "actual_revenue": actual_revenue,
        "consensus_revenue": consensus_revenue,
        "guidance_raised": guidance_raised,
        "guidance_cut": guidance_cut,
        "guidance_delta_pct": guidance_delta_pct,
        "ceo_statement_positive": ceo_positive,
        "raw_text_combined": combined[:5000],
    }


def calculate_eass(eass_inputs: dict, catalyst_type: str) -> dict:
    """Compute the 3-component Earnings Alignment Surprise Score."""
    weights = config.EASS_WEIGHTS
    actual_eps = eass_inputs.get("actual_eps")
    consensus_eps = eass_inputs.get("consensus_eps")

    # Component 1 — analyst_surprise
    if actual_eps is not None and consensus_eps is not None and consensus_eps != 0:
        raw = (actual_eps - consensus_eps) / abs(consensus_eps)
        sandbagging_penalty = 0.0  # no history yet on first deployment
        analyst_component = raw * (1 - sandbagging_penalty)
    else:
        analyst_component = 0.0

    # Component 2 — whisper_surprise (FUTURE_STUB)
    whisper_component = 0.0
    # FUTURE_STUB: When whisper feed is added, implement whisper_surprise() here.
    # Weight in config.EASS_WEIGHTS["whisper_surprise"] is already 0.00 until activated.

    # Component 3 — guidance_delta
    guidance_raised = eass_inputs.get("guidance_raised", False)
    guidance_cut_flag = eass_inputs.get("guidance_cut", False)
    guidance_delta_pct = eass_inputs.get("guidance_delta_pct", 0.0)

    if guidance_raised:
        guidance_component = guidance_delta_pct if guidance_delta_pct > 0 else 0.10
    elif guidance_cut_flag:
        guidance_component = guidance_delta_pct if guidance_delta_pct < 0 else -0.15
        guidance_component *= 2.50  # Amplify negative guidance heavily
    else:
        guidance_component = 0.0

    raw_eass = (
        weights["analyst_surprise"] * analyst_component
        + weights["whisper_surprise"] * whisper_component
        + weights["guidance_delta"] * guidance_component
    )

    normalized = max(-10.0, min(10.0, raw_eass * 20))

    # Signal quality classification
    if analyst_component > 0 and guidance_component > 0:
        signal_quality = "CONFIRMED_BULLISH"
    elif analyst_component > 0 and guidance_component < 0:
        signal_quality = "GUIDANCE_DESTRUCTION_RISK"
    elif analyst_component < 0 and guidance_component < 0:
        signal_quality = "CONFIRMED_BEARISH"
    elif analyst_component > 0 and guidance_component == 0:
        signal_quality = "ANALYST_BEAT_ONLY"
    else:
        signal_quality = "MIXED_SIGNAL"

    return {
        "eass_score": round(normalized, 4),
        "analyst_component": round(analyst_component, 4),
        "whisper_component": 0.0,
        "guidance_component": round(guidance_component, 4),
        "signal_quality": signal_quality,
        "catalyst_type": catalyst_type,
        "raw_text_combined": eass_inputs.get("raw_text_combined", ""),
    }


async def summarize_news_haiku(raw_text: str, ticker: str, client, provider: str) -> str:
    """Summarize news catalyst in 3 sentences using Stage 3 LLM (Haiku or Groq equivalent)."""
    model = config.GROQ_STAGE_3_MODEL if provider == "groq" else config.LLM_STAGE_3_FAST
    return llm_client.chat(
        client,
        provider,
        model,
        (
            "You are a financial analyst. Summarize the key catalyst facts in exactly 3 sentences. "
            "Include: what happened, the numerical magnitude, and the forward implication. Be factual only."
        ),
        f"Ticker: {ticker}\n\nNews text:\n{raw_text[:3000]}",
        config.LLM_MAX_TOKENS,
    )


async def _process_ticker(entry: dict, regime_data: dict, db_client, session: aiohttp.ClientSession,
                          client, provider: str, semaphore: asyncio.Semaphore) -> dict | None:
    """Process a single ticker through the full paid pipeline.

    Returns a qualified candidate dict or None if the ticker is filtered at any step.
    """
    async with semaphore:
        ticker = entry["ticker"]

        # Step 1: Fetch news via configured provider
        articles = await fetch_news(ticker, session)

        # Step 2: No articles → skip
        if not articles:
            print(f"[PIPELINE] {ticker}: NO_CATALYST_FOUND")
            return None

        # Step 3: Classify catalyst type — scan all articles, pick highest-priority
        catalyst_type = select_best_catalyst(articles)
        winning_article = next(
            (a for a in articles if classify_catalyst_type(a) == catalyst_type),
            articles[0],
        )
        print(
            f"[PIPELINE] {ticker}: CATALYST={catalyst_type} | "
            f"scanned={len(articles)} articles | "
            f"match=\"{winning_article['title']}\" ({winning_article.get('published_utc', 'n/a')})"
        )

        # Step 4: Low-weight catalyst → skip
        if catalyst_type == "market_movers":
            print(f"[PIPELINE] {ticker}: LOW_WEIGHT_CATALYST")
            return None

        # Step 5: Extract EASS inputs
        eass_inputs = extract_eass_inputs(articles, ticker)

        # Step 6: Calculate EASS
        eass = calculate_eass(eass_inputs, catalyst_type)

        # Step 7: EASS below threshold → skip
        if eass["eass_score"] < 2.0:
            print(f"[PIPELINE] {ticker}: EASS_BELOW_THRESHOLD ({eass['eass_score']})")
            return None

        # Step 8: Haiku summarization
        haiku_summary = await summarize_news_haiku(
            eass_inputs["raw_text_combined"], ticker, client, provider
        )

        # Step 9: Build catalyst_data dict for ChromaDB vector
        catalyst_data = {
            "eps_surprise_pct": (
                (eass_inputs["actual_eps"] - eass_inputs["consensus_eps"])
                / abs(eass_inputs["consensus_eps"])
            ) if eass_inputs["actual_eps"] and eass_inputs["consensus_eps"] else 0.0,
            "revenue_surprise_pct": (
                (eass_inputs["actual_revenue"] - eass_inputs["consensus_revenue"])
                / abs(eass_inputs["consensus_revenue"])
            ) if eass_inputs["actual_revenue"] and eass_inputs["consensus_revenue"] else 0.0,
            "guidance_delta_pct": eass_inputs.get("guidance_delta_pct", 0.0),
            "analyst_revision_count": 0,
            "ceo_statement_positive": eass_inputs.get("ceo_statement_positive", False),
        }

        # Enrich regime_data with per-ticker metrics
        enriched_regime = {**regime_data}
        enriched_regime["premarket_gap_pct"] = entry.get("pre_market_gap_pct", 0.015)
        enriched_regime["rvol_945"] = entry.get("rvol_945", 1.0)
        enriched_regime["pct_above_200ema"] = (
            (entry["price"] - entry["ema200"]) / entry["ema200"]
            if entry.get("ema200") else 0.0
        )

        # Step 10: Query ChromaDB
        outcome_profile = database.query_similar_setups(
            db_client, catalyst_type, enriched_regime, catalyst_data
        )

        # Step 11: Insufficient confidence → skip
        if outcome_profile.get("action") == "SKIP_TRADE":
            print(f"[PIPELINE] {ticker}: SKIP_TRADE ({outcome_profile.get('confidence')})")
            return None

        print(f"[PIPELINE] {ticker}: QUALIFIED (EASS={eass['eass_score']}, {catalyst_type})")
        return {
            "ticker": ticker,
            "catalyst_type": catalyst_type,
            "eass": eass,
            "haiku_summary": haiku_summary,
            "outcome_profile": outcome_profile,
            # pre_market_gap_pct promoted to top level so sector leadership gate
            # (applied post-pipeline) can compare gaps across sub-industry peers
            "pre_market_gap_pct": entry.get("pre_market_gap_pct", 0.0),
            "metrics": {**entry, **enriched_regime},
        }


async def run_pipeline(tickers_with_metrics: list, regime_data: dict, db_client) -> list:
    """Main pipeline coordinator. Only called after all free gates have passed.

    Processes all tickers concurrently (up to 5 at a time) using asyncio.gather
    to avoid paying sequential latency for Polygon fetches and LLM calls.
    """
    semaphore = asyncio.Semaphore(5)
    async with aiohttp.ClientSession() as session:
        client, provider = llm_client.create_client()
        tasks = [
            _process_ticker(entry, regime_data, db_client, session, client, provider, semaphore)
            for entry in tickers_with_metrics
        ]
        results = await asyncio.gather(*tasks)

    return [r for r in results if r is not None]
