"""Claude Sonnet structural audit and final strategy card output."""

import anthropic
import json
import re
import datetime
import os
import config

_AUDIT_SAFE_DEFAULT = {
    "threat_level": "HIGH",
    "threats_identified": ["Audit parsing failed — manual review required"],
    "durability": "UNCERTAIN",
    "regime_support": "NEUTRAL",
    "entry_strategy": "DO_NOT_ENTER",
    "audit_note": "Automated audit failed. Do not trade without manual review.",
}


async def run_sonnet_audit(candidate: dict, client: anthropic.Anthropic) -> dict:
    """Run Claude Sonnet structural risk audit on a qualified candidate."""
    system_prompt = (
        "You are a senior institutional risk analyst. Your job is to identify "
        "structural threats that quantitative scores cannot detect. Analyze the "
        "provided trade setup and identify: (1) any text-based red flags in the "
        "news summary, (2) whether the catalyst is durable or one-time, "
        "(3) whether the market regime supports continuation. "
        "Output ONLY a JSON object with these exact keys:\n"
        "  threat_level: LOW | MEDIUM | HIGH\n"
        "  threats_identified: list of strings (empty list if none)\n"
        "  durability: DURABLE | ONE_TIME | UNCERTAIN\n"
        "  regime_support: CONFIRMED | NEUTRAL | AGAINST\n"
        "  entry_strategy: MARKET_OPEN | PULLBACK_LIMIT_21EMA | DO_NOT_ENTER\n"
        "  audit_note: one sentence summary"
    )

    try:
        message = client.messages.create(
            model=config.LLM_STAGE_4_PREMIUM,
            max_tokens=config.LLM_MAX_TOKENS,
            system=system_prompt,
            messages=[
                {"role": "user", "content": json.dumps(candidate, default=str)}
            ],
        )
        raw = message.content[0].text.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-z]*\n?", "", raw)
            raw = raw.rstrip("`").strip()
        return json.loads(raw)
    except Exception:
        return dict(_AUDIT_SAFE_DEFAULT)


def apply_regime_strategy_mutator(audit_result: dict, regime: str) -> dict:
    """Downgrade entry strategy and halve position size in bearish regime."""
    if regime == "BEARISH":
        if audit_result.get("entry_strategy") == "MARKET_OPEN":
            audit_result["entry_strategy"] = "PULLBACK_LIMIT_21EMA"
        audit_result["threats_identified"] = list(audit_result.get("threats_identified", [])) + [
            "BEARISH REGIME: Position size reduced 50%, entry strategy downgraded."
        ]
    return audit_result


def calculate_position_size(account_size: float, stop_distance_pct: float, regime: str) -> dict:
    """Calculate position size based on account risk and stop distance.

    Note: shares = risk_amount / (account_size * stop_distance_pct) is the spec formula.
    This yields fractional shares (dimensionless ratio), not dollar-risk/price sizing.
    """
    risk_amount = account_size * config.STANDARD_ACCOUNT_RISK
    if regime == "BEARISH":
        risk_amount *= config.BEARISH_REGIME_MUTATOR

    shares = (risk_amount / (account_size * stop_distance_pct)) if stop_distance_pct > 0 else 0

    return {
        "risk_amount_dollars": round(risk_amount, 2),
        "stop_distance_pct": round(stop_distance_pct, 4),
        "suggested_shares": int(shares),
        "regime_applied": regime,
    }


def format_strategy_card(candidate: dict, audit: dict, position: dict) -> str:
    """Format a Markdown strategy card for a qualified trade setup."""
    ticker = candidate["ticker"]
    catalyst_type = candidate["catalyst_type"]
    eass = candidate["eass"]
    profile = candidate["outcome_profile"]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M EST")

    threats = audit.get("threats_identified", [])
    threats_str = "\n  - ".join(threats) if threats else "None identified"

    return f"""## STRATEGY CARD — {ticker} — {timestamp}

### Catalyst
- Type: {catalyst_type}
- EASS Score: {eass['eass_score']} / 10.0
- Signal Quality: {eass['signal_quality']}
- Summary: {candidate.get('haiku_summary', 'N/A')}

### Historical Validation
- Confidence: {profile['confidence']} ({profile['sample_size']} similar setups)
- Win Rate Day 1 / Day 3: {profile['win_rate_day1']} / {profile['win_rate_day3']}
- Expected Value per trade: {profile['expected_value']}
- Average max adverse move: {profile['avg_max_adverse']}

### Risk Audit
- Threat Level: {audit.get('threat_level', 'N/A')}
- Threats: {threats_str}
- Catalyst Durability: {audit.get('durability', 'N/A')}
- Regime Support: {audit.get('regime_support', 'N/A')}
- Audit Note: {audit.get('audit_note', 'N/A')}

### Trade Parameters
- Entry Strategy: {audit.get('entry_strategy', 'N/A')}
- Suggested Stop: {profile['suggested_stop']}
- Suggested Target: {profile['suggested_target']}
- Risk Amount: ${position['risk_amount_dollars']}
- Suggested Shares: {position['suggested_shares']}

---"""


def write_outputs(strategy_cards: list, log_entries: list):
    """Write strategy cards and decision log to their output files."""
    with open(config.STRATEGY_OUTPUT_PATH, "w") as f:
        f.write("\n\n---\n\n".join(strategy_cards))

    with open(config.LOG_QUEUE_PATH, "w") as f:
        json.dump(log_entries, f, indent=2, default=str)

    print(f"[OUTPUT] {len(strategy_cards)} strategy card(s) written to {config.STRATEGY_OUTPUT_PATH}")
    print(f"[OUTPUT] {len(log_entries)} log entries written to {config.LOG_QUEUE_PATH}")
