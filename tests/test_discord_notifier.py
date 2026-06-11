"""Tests for discord_notifier.py — AlphaCrew Discord webhook delivery.

Covers: color mapping, percentage formatting, card parsing, embed construction,
webhook delivery guards, and all three public async send functions.
"""

import asyncio
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import discord_notifier
import config


# ---------------------------------------------------------------------------
# Fixtures — reusable sample cards matching risk_auditor.format_strategy_card()
# ---------------------------------------------------------------------------

CARD_LOW_THREAT = """\
## STRATEGY CARD — NVDA — 2026-06-12 08:05 EST

### Catalyst
- Type: earnings_beat_large
- EASS Score: 7.2 / 10.0
- Signal Quality: CONFIRMED_BULLISH
- Summary: NVDA beat EPS by 18% on AI demand. Guidance raised 12%. Institutional accumulation confirmed.

### Historical Validation
- Confidence: HIGH (52 similar setups)
- Win Rate Day 1 / Day 3: 0.71 / 0.65
- Expected Value per trade: 0.034
- Average max adverse move: 0.021

### Risk Audit
- Threat Level: LOW
- Threats: None identified
- Catalyst Durability: DURABLE
- Regime Support: CONFIRMED
- Audit Note: Clean earnings beat with durable catalyst.

### Trade Parameters
- Entry Strategy: MARKET_OPEN
- Suggested Stop: -0.025
- Suggested Target: 0.065
- Risk Amount: $1000.0
- Suggested Shares: 20

---"""

# Card with multi-line threats (bearish regime mutator applied)
CARD_HIGH_THREAT = """\
## STRATEGY CARD — TSLA — 2026-06-12 08:05 EST

### Catalyst
- Type: guidance_cut
- EASS Score: 2.1 / 10.0
- Signal Quality: GUIDANCE_DESTRUCTION_RISK
- Summary: Tesla cut full-year guidance. Margin compression cited. Competition accelerating.

### Historical Validation
- Confidence: MODERATE (45 similar setups)
- Win Rate Day 1 / Day 3: 0.41 / 0.38
- Expected Value per trade: -0.012
- Average max adverse move: 0.044

### Risk Audit
- Threat Level: HIGH
- Threats: Margin compression ongoing
  - Guidance uncertainty
  - BEARISH REGIME: Position size reduced 50%, entry strategy downgraded.
- Catalyst Durability: UNCERTAIN
- Regime Support: AGAINST
- Audit Note: Multiple structural threats — avoid.

### Trade Parameters
- Entry Strategy: DO_NOT_ENTER
- Suggested Stop: -0.04
- Suggested Target: 0.02
- Risk Amount: $500.0
- Suggested Shares: 5

---"""

RVOL_LOG = [
    {"ticker": "NVDA", "status": "ACTIVE",     "interval_rvol": 4.72, "entry_strategy": "MARKET_OPEN"},
    {"ticker": "MSFT", "status": "DOWNGRADED", "interval_rvol": 1.18, "entry_strategy": "PULLBACK_LIMIT_21EMA"},
    {"step": "MACRO_GATE", "result": "BULLISH"},          # non-ticker gate entry, must be ignored
]


# ---------------------------------------------------------------------------
# _threat_color()
# ---------------------------------------------------------------------------

def test_threat_color_low():
    assert discord_notifier._threat_color("LOW") == 0x2ECC71

def test_threat_color_medium():
    assert discord_notifier._threat_color("MEDIUM") == 0xF39C12

def test_threat_color_high():
    assert discord_notifier._threat_color("HIGH") == 0xE74C3C

def test_threat_color_unknown_returns_grey():
    assert discord_notifier._threat_color("UNKNOWN") == 0x95A5A6

def test_threat_color_case_insensitive():
    assert discord_notifier._threat_color("low") == 0x2ECC71
    assert discord_notifier._threat_color("High") == 0xE74C3C


# ---------------------------------------------------------------------------
# _fmt_pct()
# ---------------------------------------------------------------------------

def test_fmt_pct_negative_stop():
    assert discord_notifier._fmt_pct("-0.025") == "-2.50%"

def test_fmt_pct_positive_target():
    assert discord_notifier._fmt_pct("0.065") == "6.50%"

def test_fmt_pct_small_ev():
    assert discord_notifier._fmt_pct("0.034") == "3.40%"

def test_fmt_pct_invalid_returns_as_string():
    assert discord_notifier._fmt_pct("N/A") == "N/A"
    assert discord_notifier._fmt_pct("") == ""


# ---------------------------------------------------------------------------
# _fmt_wr()
# ---------------------------------------------------------------------------

def test_fmt_wr_typical():
    assert discord_notifier._fmt_wr("0.71") == "71%"
    assert discord_notifier._fmt_wr("0.65") == "65%"

def test_fmt_wr_invalid_returns_as_string():
    assert discord_notifier._fmt_wr("N/A") == "N/A"


# ---------------------------------------------------------------------------
# _parse_card() — field extraction
# ---------------------------------------------------------------------------

def test_parse_card_ticker():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["ticker"] == "NVDA"

def test_parse_card_date_extracted_from_timestamp():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["date"] == "2026-06-12"
    assert p["timestamp"] == "2026-06-12 08:05 EST"

def test_parse_card_catalyst_type():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["catalyst_type"] == "earnings_beat_large"

def test_parse_card_eass_score_strips_suffix():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["eass_score"] == "7.2"   # "/ 10.0" removed

def test_parse_card_signal_quality():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["signal_quality"] == "CONFIRMED_BULLISH"

def test_parse_card_haiku_summary():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert "NVDA beat EPS" in p["haiku_summary"]

def test_parse_card_confidence_and_sample_size():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["confidence"] == "HIGH"
    assert p["sample_size"] == "52"

def test_parse_card_win_rates():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["win_rate_day1"] == "0.71"
    assert p["win_rate_day3"] == "0.65"

def test_parse_card_expected_value():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["expected_value"] == "0.034"

def test_parse_card_threat_level():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["threat_level"] == "LOW"

def test_parse_card_single_threat_line():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["threats"] == "None identified"

def test_parse_card_multiline_threats_collected():
    p = discord_notifier._parse_card(CARD_HIGH_THREAT)
    # All three threat continuation lines must appear
    assert "Margin compression ongoing" in p["threats"]
    assert "Guidance uncertainty" in p["threats"]
    assert "BEARISH REGIME" in p["threats"]

def test_parse_card_durability():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["durability"] == "DURABLE"

def test_parse_card_regime_support():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["regime_support"] == "CONFIRMED"

def test_parse_card_audit_note():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert "durable catalyst" in p["audit_note"]

def test_parse_card_entry_strategy():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["entry_strategy"] == "MARKET_OPEN"

def test_parse_card_stop_and_target_as_raw_strings():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["suggested_stop"] == "-0.025"
    assert p["suggested_target"] == "0.065"

def test_parse_card_risk_amount_strips_dollar():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["risk_amount"] == "1000.0"   # "$" stripped

def test_parse_card_suggested_shares():
    p = discord_notifier._parse_card(CARD_LOW_THREAT)
    assert p["suggested_shares"] == "20"

def test_parse_card_empty_string_returns_defaults():
    p = discord_notifier._parse_card("")
    assert p["ticker"] == "N/A"
    assert p["eass_score"] == "N/A"
    assert p["entry_strategy"] == "N/A"

def test_parse_card_high_threat_ticker():
    p = discord_notifier._parse_card(CARD_HIGH_THREAT)
    assert p["ticker"] == "TSLA"
    assert p["threat_level"] == "HIGH"


# ---------------------------------------------------------------------------
# _build_card_embed()
# ---------------------------------------------------------------------------

def _parsed_low():
    return discord_notifier._parse_card(CARD_LOW_THREAT)

def _parsed_high():
    return discord_notifier._parse_card(CARD_HIGH_THREAT)

def test_build_card_embed_has_embeds_key():
    payload = discord_notifier._build_card_embed(_parsed_low())
    assert "embeds" in payload
    assert len(payload["embeds"]) == 1

def test_build_card_embed_title_contains_ticker():
    embed = discord_notifier._build_card_embed(_parsed_low())["embeds"][0]
    assert "NVDA" in embed["title"]
    assert "AlphaCrew" in embed["title"]

def test_build_card_embed_description_is_haiku():
    embed = discord_notifier._build_card_embed(_parsed_low())["embeds"][0]
    assert "NVDA beat EPS" in embed["description"]

def test_build_card_embed_color_green_for_low():
    embed = discord_notifier._build_card_embed(_parsed_low())["embeds"][0]
    assert embed["color"] == 0x2ECC71

def test_build_card_embed_color_red_for_high():
    embed = discord_notifier._build_card_embed(_parsed_high())["embeds"][0]
    assert embed["color"] == 0xE74C3C

def test_build_card_embed_footer():
    embed = discord_notifier._build_card_embed(_parsed_low())["embeds"][0]
    assert embed["footer"]["text"] == "AlphaCrew — Strategy Engine v5.1"

def test_build_card_embed_take_profit_formatted_as_pct():
    fields = {f["name"]: f["value"] for f in discord_notifier._build_card_embed(_parsed_low())["embeds"][0]["fields"]}
    assert fields["Take Profit"] == "6.50%"

def test_build_card_embed_stop_loss_formatted_as_pct():
    fields = {f["name"]: f["value"] for f in discord_notifier._build_card_embed(_parsed_low())["embeds"][0]["fields"]}
    assert fields["Stop Loss"] == "-2.50%"

def test_build_card_embed_win_rate_formatted_as_pct():
    fields = {f["name"]: f["value"] for f in discord_notifier._build_card_embed(_parsed_low())["embeds"][0]["fields"]}
    assert fields["Win Rate D1 / D3"] == "71% / 65%"

def test_build_card_embed_historical_match_has_sample_size():
    fields = {f["name"]: f["value"] for f in discord_notifier._build_card_embed(_parsed_low())["embeds"][0]["fields"]}
    assert "52" in fields["Historical Match"]

def test_build_card_embed_suggested_strategy_field():
    fields = {f["name"]: f["value"] for f in discord_notifier._build_card_embed(_parsed_low())["embeds"][0]["fields"]}
    assert fields["Suggested Strategy"] == "MARKET_OPEN"

def test_build_card_embed_has_14_fields():
    fields = discord_notifier._build_card_embed(_parsed_low())["embeds"][0]["fields"]
    assert len(fields) == 14

def test_build_card_embed_timestamp_present():
    embed = discord_notifier._build_card_embed(_parsed_low())["embeds"][0]
    assert "timestamp" in embed


# ---------------------------------------------------------------------------
# _post_webhook() — disabled guard and exception swallowing
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_webhook_noop_when_discord_disabled():
    """No HTTP call is made when DISCORD_ENABLED is False."""
    with patch.object(config, "DISCORD_ENABLED", False):
        with patch("aiohttp.ClientSession") as mock_cls:
            await discord_notifier._post_webhook({"embeds": []})
            mock_cls.assert_not_called()

@pytest.mark.asyncio
async def test_post_webhook_swallows_connection_exception():
    """Network exceptions are caught and never re-raised."""
    with patch.object(config, "DISCORD_ENABLED", True), \
         patch.object(config, "DISCORD_WEBHOOK_URL", "https://example.com"):
        with patch("aiohttp.ClientSession") as mock_cls:
            mock_cls.return_value.__aenter__ = AsyncMock(side_effect=Exception("timeout"))
            mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
            # Must not raise
            await discord_notifier._post_webhook({"embeds": []})

@pytest.mark.asyncio
async def test_post_webhook_logs_warning_on_bad_status(caplog):
    """Non-204 HTTP status triggers a warning log entry."""
    import logging

    mock_resp = AsyncMock()
    mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_resp.__aexit__ = AsyncMock(return_value=False)
    mock_resp.status = 400
    mock_resp.text = AsyncMock(return_value="Bad Request")

    mock_session = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.post = MagicMock(return_value=mock_resp)

    with patch.object(config, "DISCORD_ENABLED", True), \
         patch.object(config, "DISCORD_WEBHOOK_URL", "https://example.com"), \
         patch("aiohttp.ClientSession", return_value=mock_session), \
         caplog.at_level(logging.WARNING, logger="discord_notifier"):
        await discord_notifier._post_webhook({"embeds": []})

    assert any("400" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# send_strategy_cards() — higher-level post orchestration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_strategy_cards_empty_posts_no_signals():
    """Empty card list calls send_no_signals_message (1 webhook post)."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_strategy_cards([], [])
        assert mock_post.call_count == 1
        payload = mock_post.call_args[0][0]
        assert "No Signals" in payload["embeds"][0]["title"]

@pytest.mark.asyncio
async def test_send_strategy_cards_single_card_posts_header_plus_card():
    """One card → 2 webhook posts: header embed + card embed."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_strategy_cards([CARD_LOW_THREAT], [])
        assert mock_post.call_count == 2

@pytest.mark.asyncio
async def test_send_strategy_cards_two_cards_posts_header_plus_two():
    """Two cards → 3 webhook posts: header + 2 card embeds."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_strategy_cards([CARD_LOW_THREAT, CARD_HIGH_THREAT], [])
        assert mock_post.call_count == 3

@pytest.mark.asyncio
async def test_send_strategy_cards_header_mentions_signal_count():
    """Header embed description states the correct number of signals."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_strategy_cards([CARD_LOW_THREAT, CARD_HIGH_THREAT], [])
        header_payload = mock_post.call_args_list[0][0][0]
        assert "2" in header_payload["embeds"][0]["description"]

@pytest.mark.asyncio
async def test_send_strategy_cards_card_embed_contains_ticker():
    """Second webhook call contains the card embed with correct ticker."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_strategy_cards([CARD_LOW_THREAT], [])
        card_payload = mock_post.call_args_list[1][0][0]
        assert "NVDA" in card_payload["embeds"][0]["title"]


# ---------------------------------------------------------------------------
# send_rvol_update()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_rvol_update_posts_one_embed():
    """RVOL update always posts exactly one embed."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_rvol_update(RVOL_LOG)
        assert mock_post.call_count == 1

@pytest.mark.asyncio
async def test_send_rvol_update_embed_has_one_field_per_ticker():
    """Each ticker with interval_rvol gets one field in the embed."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_rvol_update(RVOL_LOG)
        embed = mock_post.call_args[0][0]["embeds"][0]
        ticker_fields = [f["name"] for f in embed["fields"]]
        assert "NVDA" in ticker_fields
        assert "MSFT" in ticker_fields

@pytest.mark.asyncio
async def test_send_rvol_update_rvol_value_formatted():
    """RVOL values appear formatted to 2 decimal places."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_rvol_update(RVOL_LOG)
        embed = mock_post.call_args[0][0]["embeds"][0]
        nvda_field = next(f for f in embed["fields"] if f["name"] == "NVDA")
        assert "4.72" in nvda_field["value"]

@pytest.mark.asyncio
async def test_send_rvol_update_gate_entries_excluded():
    """Log entries without 'interval_rvol' (gate steps) are not shown as ticker fields."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_rvol_update(RVOL_LOG)
        embed = mock_post.call_args[0][0]["embeds"][0]
        field_names = [f["name"] for f in embed["fields"]]
        assert "MACRO_GATE" not in field_names

@pytest.mark.asyncio
async def test_send_rvol_update_empty_log_no_active_tickers():
    """Empty log → 'No active tickers' description."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_rvol_update([])
        embed = mock_post.call_args[0][0]["embeds"][0]
        assert "No active tickers" in embed["description"]
        assert embed["fields"] == []

@pytest.mark.asyncio
async def test_send_rvol_update_title_and_footer():
    """Embed title and footer carry the AlphaCrew branding."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_rvol_update(RVOL_LOG)
        embed = mock_post.call_args[0][0]["embeds"][0]
        assert "AlphaCrew" in embed["title"]
        assert embed["footer"]["text"] == "AlphaCrew — Strategy Engine v5.1"


# ---------------------------------------------------------------------------
# send_no_signals_message()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_send_no_signals_posts_grey_embed():
    """No-signals message uses the grey color constant."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_no_signals_message()
        embed = mock_post.call_args[0][0]["embeds"][0]
        assert embed["color"] == 0x95A5A6

@pytest.mark.asyncio
async def test_send_no_signals_title_and_footer():
    """No-signals embed carries correct title and AlphaCrew footer."""
    with patch.object(discord_notifier, "_post_webhook", new_callable=AsyncMock) as mock_post:
        await discord_notifier.send_no_signals_message()
        embed = mock_post.call_args[0][0]["embeds"][0]
        assert "No Signals" in embed["title"]
        assert "AlphaCrew" in embed["title"]
        assert embed["footer"]["text"] == "AlphaCrew — Strategy Engine v5.1"


# ---------------------------------------------------------------------------
# notifier.py async wrappers
# ---------------------------------------------------------------------------

def test_notifier_send_discord_premarket_is_coroutine():
    """send_discord_premarket must be an async function."""
    import inspect
    import notifier
    assert inspect.iscoroutinefunction(notifier.send_discord_premarket)

def test_notifier_send_discord_rvol_is_coroutine():
    """send_discord_rvol must be an async function."""
    import inspect
    import notifier
    assert inspect.iscoroutinefunction(notifier.send_discord_rvol)

@pytest.mark.asyncio
async def test_notifier_send_discord_premarket_delegates_to_discord_notifier():
    """send_discord_premarket calls discord_notifier.send_strategy_cards."""
    import notifier
    with patch.object(discord_notifier, "send_strategy_cards", new_callable=AsyncMock) as mock_fn:
        await notifier.send_discord_premarket(["card"], [{"step": "TEST"}])
        mock_fn.assert_called_once_with(["card"], [{"step": "TEST"}])

@pytest.mark.asyncio
async def test_notifier_send_discord_rvol_delegates_to_discord_notifier():
    """send_discord_rvol calls discord_notifier.send_rvol_update."""
    import notifier
    with patch.object(discord_notifier, "send_rvol_update", new_callable=AsyncMock) as mock_fn:
        await notifier.send_discord_rvol(RVOL_LOG)
        mock_fn.assert_called_once_with(RVOL_LOG)
