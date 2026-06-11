"""Discord webhook delivery for AlphaCrew premarket signals and RVOL updates.

Uses aiohttp (already installed) to POST embeds to a Discord Incoming Webhook URL.
No new dependencies. All functions are async. Errors are caught and logged —
a Discord failure never crashes the pipeline.
"""

import logging
import re
import datetime
import aiohttp
import config

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(total=10)

# Embed color constants keyed by threat level
_COLOR_GREEN = 0x2ECC71   # LOW threat
_COLOR_AMBER = 0xF39C12   # MEDIUM threat
_COLOR_RED   = 0xE74C3C   # HIGH threat
_COLOR_GREY  = 0x95A5A6   # unknown / no-signals
_COLOR_BLUE  = 0x3498DB   # informational (RVOL update, header)


async def _post_webhook(payload: dict) -> None:
    """POST a single embed payload to the configured Discord webhook URL.

    Silently returns on any network or HTTP error so the pipeline is never blocked.
    """
    if not config.DISCORD_ENABLED:
        return
    try:
        async with aiohttp.ClientSession(timeout=_TIMEOUT) as session:
            async with session.post(config.DISCORD_WEBHOOK_URL, json=payload) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    logger.warning("[DISCORD] Webhook returned %s: %s", resp.status, body[:200])
    except Exception as exc:
        logger.warning("[DISCORD] Webhook delivery failed: %s", exc)


def _threat_color(threat_level: str) -> int:
    """Return a Discord embed color integer for the given threat level string."""
    return {
        "LOW": _COLOR_GREEN,
        "MEDIUM": _COLOR_AMBER,
        "HIGH": _COLOR_RED,
    }.get(str(threat_level).upper(), _COLOR_GREY)


def _fmt_pct(value_str: str) -> str:
    """Convert a raw decimal string (e.g. '-0.025') to a percentage string (e.g. '-2.50%')."""
    try:
        return f"{float(value_str):.2%}"
    except (ValueError, TypeError):
        return str(value_str)


def _fmt_wr(value_str: str) -> str:
    """Convert a win-rate decimal string (e.g. '0.71') to a rounded percentage (e.g. '71%')."""
    try:
        return f"{float(value_str):.0%}"
    except (ValueError, TypeError):
        return str(value_str)


def _parse_card(card_markdown: str) -> dict:
    """Parse a strategy card markdown string into a flat dict of display fields.

    Matches the exact format produced by risk_auditor.format_strategy_card().
    Returns all fields as strings; unmatched fields default to 'N/A'.
    """
    result = {k: "N/A" for k in (
        "ticker", "date", "timestamp", "catalyst_type", "eass_score",
        "signal_quality", "haiku_summary", "confidence", "sample_size",
        "win_rate_day1", "win_rate_day3", "expected_value", "avg_max_adverse",
        "threat_level", "threats", "durability", "regime_support",
        "audit_note", "entry_strategy", "suggested_stop", "suggested_target",
        "risk_amount", "suggested_shares",
    )}

    lines = card_markdown.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        # Header: ## STRATEGY CARD — NVDA — 2026-06-12 08:05 EST
        if stripped.startswith("## STRATEGY CARD"):
            parts = stripped.split(" — ")  # em dash
            if len(parts) >= 3:
                result["ticker"] = parts[1].strip()
                result["timestamp"] = parts[2].strip()
                result["date"] = result["timestamp"].split()[0]

        elif stripped.startswith("- Type:"):
            result["catalyst_type"] = stripped[len("- Type:"):].strip()

        elif stripped.startswith("- EASS Score:"):
            val = stripped[len("- EASS Score:"):].strip()
            result["eass_score"] = val.replace(" / 10.0", "").strip()

        elif stripped.startswith("- Signal Quality:"):
            result["signal_quality"] = stripped[len("- Signal Quality:"):].strip()

        elif stripped.startswith("- Summary:"):
            result["haiku_summary"] = stripped[len("- Summary:"):].strip()

        elif stripped.startswith("- Confidence:"):
            raw = stripped[len("- Confidence:"):].strip()
            # Format: "HIGH (52 similar setups)"
            m = re.match(r"^(\w+)\s+\((\d+)\s+similar setups\)", raw)
            if m:
                result["confidence"] = m.group(1)
                result["sample_size"] = m.group(2)
            else:
                result["confidence"] = raw

        elif stripped.startswith("- Win Rate Day 1 / Day 3:"):
            raw = stripped[len("- Win Rate Day 1 / Day 3:"):].strip()
            halves = raw.split("/")
            if len(halves) == 2:
                result["win_rate_day1"] = halves[0].strip()
                result["win_rate_day3"] = halves[1].strip()

        elif stripped.startswith("- Expected Value per trade:"):
            result["expected_value"] = stripped[len("- Expected Value per trade:"):].strip()

        elif stripped.startswith("- Average max adverse move:"):
            result["avg_max_adverse"] = stripped[len("- Average max adverse move:"):].strip()

        elif stripped.startswith("- Threat Level:"):
            result["threat_level"] = stripped[len("- Threat Level:"):].strip()

        elif stripped.startswith("- Threats:"):
            # Collect the first line value plus any indented continuation lines (  - ...)
            first = stripped[len("- Threats:"):].strip()
            parts = [first] if first else []
            j = i + 1
            while j < len(lines) and lines[j].startswith("  - "):
                parts.append(lines[j].strip()[2:])  # strip leading "- "
                j += 1
            result["threats"] = "\n".join(parts) if parts else "None identified"
            i = j
            continue

        elif stripped.startswith("- Catalyst Durability:"):
            result["durability"] = stripped[len("- Catalyst Durability:"):].strip()

        elif stripped.startswith("- Regime Support:"):
            result["regime_support"] = stripped[len("- Regime Support:"):].strip()

        elif stripped.startswith("- Audit Note:"):
            result["audit_note"] = stripped[len("- Audit Note:"):].strip()

        elif stripped.startswith("- Entry Strategy:"):
            result["entry_strategy"] = stripped[len("- Entry Strategy:"):].strip()

        elif stripped.startswith("- Suggested Stop:"):
            result["suggested_stop"] = stripped[len("- Suggested Stop:"):].strip()

        elif stripped.startswith("- Suggested Target:"):
            result["suggested_target"] = stripped[len("- Suggested Target:"):].strip()

        elif stripped.startswith("- Risk Amount: $"):
            result["risk_amount"] = stripped[len("- Risk Amount: $"):].strip()

        elif stripped.startswith("- Suggested Shares:"):
            result["suggested_shares"] = stripped[len("- Suggested Shares:"):].strip()

        i += 1

    return result


def _build_card_embed(parsed: dict) -> dict:
    """Build a Discord webhook payload dict for a single AlphaCrew strategy card.

    Uses em-dash separators and percentage formatting to match the AlphaCrew card protocol.
    """
    stop_str   = _fmt_pct(parsed["suggested_stop"])
    target_str = _fmt_pct(parsed["suggested_target"])
    ev_str     = _fmt_pct(parsed["expected_value"])
    wr_d1      = _fmt_wr(parsed["win_rate_day1"])
    wr_d3      = _fmt_wr(parsed["win_rate_day3"])

    embed = {
        "title": f"AlphaCrew Signal — {parsed['ticker']}",
        "description": parsed["haiku_summary"],
        "color": _threat_color(parsed["threat_level"]),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "footer": {"text": "AlphaCrew — Strategy Engine v5.1"},
        "fields": [
            {"name": "Date",               "value": parsed["date"],                                              "inline": True},
            {"name": "Ticker",             "value": parsed["ticker"],                                            "inline": True},
            {"name": "Catalyst Type",      "value": parsed["catalyst_type"],                                     "inline": True},
            {"name": "Signal Quality",     "value": parsed["signal_quality"],                                    "inline": True},
            {"name": "EASS Score",         "value": f"{parsed['eass_score']} / 10.0",                           "inline": True},
            {"name": "Suggested Strategy", "value": parsed["entry_strategy"],                                    "inline": True},
            {"name": "Historical Match",   "value": f"{parsed['sample_size']} similar setups",                  "inline": True},
            {"name": "Win Rate D1 / D3",   "value": f"{wr_d1} / {wr_d3}",                                      "inline": True},
            {"name": "Expected Value",     "value": ev_str,                                                      "inline": True},
            {"name": "Take Profit",        "value": target_str,                                                  "inline": True},
            {"name": "Stop Loss",          "value": stop_str,                                                    "inline": True},
            {"name": "Position Size",      "value": f"{parsed['suggested_shares']} shares · ${parsed['risk_amount']} risk", "inline": False},
            {"name": "Audit Note",         "value": parsed["audit_note"],                                        "inline": False},
            {"name": "Threats",            "value": parsed["threats"] if parsed["threats"] != "N/A" else "None identified", "inline": False},
        ],
    }
    return {"embeds": [embed]}


async def send_strategy_cards(strategy_cards: list, log_entries: list) -> None:
    """Send AlphaCrew strategy card embeds to Discord after the premarket pipeline completes.

    Posts a header summary embed first, then one embed per qualifying card.
    Calls send_no_signals_message() when no cards qualify.
    """
    if not strategy_cards:
        await send_no_signals_message()
        return

    count = len(strategy_cards)
    header = {
        "embeds": [{
            "title": "AlphaCrew — Pre-Market Pipeline Complete",
            "description": f"**{count} signal{'s' if count != 1 else ''}** qualified at 08:05 EST.",
            "color": _COLOR_BLUE,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "footer": {"text": "AlphaCrew — Strategy Engine v5.1"},
        }]
    }
    await _post_webhook(header)

    for card_md in strategy_cards:
        parsed = _parse_card(card_md)
        await _post_webhook(_build_card_embed(parsed))


async def send_rvol_update(log_entries: list) -> None:
    """Send the 9:45 AM RVOL confirmation summary embed to Discord."""
    reviewed = [
        e for e in log_entries
        if e.get("status") in ("ACTIVE", "DOWNGRADED") and "ticker" in e and "interval_rvol" in e
    ]

    if not reviewed:
        description = "No active tickers to confirm at open."
        fields = []
    else:
        description = f"Live volume check complete — {len(reviewed)} ticker(s) reviewed."
        fields = [
            {
                "name": e["ticker"],
                "value": (
                    f"RVOL: **{e['interval_rvol']:.2f}x** · "
                    f"Status: **{e['status']}** · "
                    f"Strategy: {e.get('entry_strategy', 'N/A')}"
                ),
                "inline": False,
            }
            for e in reviewed
        ]

    embed = {
        "title": "AlphaCrew — RVOL Confirmation 9:45 AM",
        "description": description,
        "color": _COLOR_BLUE,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "footer": {"text": "AlphaCrew — Strategy Engine v5.1"},
        "fields": fields,
    }
    await _post_webhook({"embeds": [embed]})


async def send_no_signals_message() -> None:
    """Send a brief no-signals embed when the premarket pipeline produces no qualifying cards."""
    embed = {
        "title": "AlphaCrew — No Signals Today",
        "description": "Pre-market pipeline completed — no tickers qualified at 08:05 EST.",
        "color": _COLOR_GREY,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "footer": {"text": "AlphaCrew — Strategy Engine v5.1"},
    }
    await _post_webhook({"embeds": [embed]})
