"""Production scheduler: 8:05 AM pre-market pipeline + 9:45 AM live RVOL check."""

import asyncio
import schedule
import time
import json
import datetime
import pytz
import yfinance as yf
import llm_client
import os
from dotenv import load_dotenv

import config
import regime_engine
import pipeline
import risk_auditor
import database
import notifier

load_dotenv()

# ---------------------------------------------------------------------------
# Prototype candidate universe — replace with live screener feed in production
# ---------------------------------------------------------------------------
CANDIDATE_TICKERS = ["NVDA", "MSFT", "AAPL", "AMD", "META", "AMZN", "GOOGL", "TSLA"]

# ---------------------------------------------------------------------------
# Queue schema for premarket → live handoff
# Each entry written by the premarket phase has:
#   ticker, status ("ACTIVE"|"DOWNGRADED"), entry_strategy, eass_score,
#   catalyst_type, card_index (index into strategy_cards list), timestamp
# ---------------------------------------------------------------------------


def _get_premarket_gap(ticker: str) -> float:
    """Estimate pre-market gap from yfinance pre/post market data."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        pre_price = info.get("preMarketPrice") or info.get("regularMarketPrice") or 0
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose") or 1
        if prev_close:
            return (pre_price - prev_close) / prev_close
    except Exception:
        pass
    return 0.0


async def run_premarket_pipeline(db_client):
    """Full 9-gate pre-market pipeline. Runs at 8:05 AM EST."""
    print(f"\n[SCHEDULER] Pre-market pipeline started — {datetime.datetime.now()}")
    client, provider = llm_client.create_client()
    print(f"[SCHEDULER] LLM provider: {provider}")
    log_entries = []
    strategy_cards = []

    # STEP 1.1 — MACRO GATE
    regime_data = regime_engine.get_market_regime()
    log_entries.append({"step": "MACRO_GATE", "result": regime_data})
    print(f"[1.1] Regime: {regime_data['regime']} (SPY {regime_data['spy_close']:.2f} vs SMA50 {regime_data['spy_sma50']:.2f})")
    if regime_data["regime"] == "BEARISH":
        print("[1.1] WARNING: Bearish regime — bearish mutator will be applied, not a hard stop")

    # STEP 1.2 — SECTOR GATE
    sector_data = regime_engine.get_top_sectors()
    top_sectors = sector_data["top_sectors"]
    log_entries.append({"step": "SECTOR_GATE", "top_sectors": top_sectors})
    print(f"[1.2] Top sectors: {top_sectors}")

    sector_survivors = regime_engine.filter_by_sector(CANDIDATE_TICKERS, top_sectors)
    log_entries.append({"step": "SECTOR_FILTER", "survivors": sector_survivors, "count": len(sector_survivors)})
    print(f"[1.2] Sector survivors: {sector_survivors} ({len(sector_survivors)}/{len(CANDIDATE_TICKERS)})")

    if not sector_survivors:
        log_entries.append({"step": "ABORT", "reason": "NO_SECTOR_SURVIVORS"})
        risk_auditor.write_outputs([], log_entries)
        await notifier.send_discord_premarket([], log_entries)
        print("[SCHEDULER] ABORT: NO_SECTOR_SURVIVORS")
        return

    # STEP 1.3 — VOLATILITY + TREND GATE
    individual_survivors = regime_engine.apply_individual_gates(sector_survivors)
    log_entries.append({"step": "INDIVIDUAL_GATES", "count": len(individual_survivors)})
    print(f"[1.3] Individual gate survivors: {[e['ticker'] for e in individual_survivors]}")

    if not individual_survivors:
        log_entries.append({"step": "ABORT", "reason": "NO_INDIVIDUAL_GATE_SURVIVORS"})
        risk_auditor.write_outputs([], log_entries)
        await notifier.send_discord_premarket([], log_entries)
        print("[SCHEDULER] ABORT: NO_INDIVIDUAL_GATE_SURVIVORS")
        return

    # STEP 1.4 — EARNINGS PROXIMITY GATE
    earnings_survivors = regime_engine.apply_earnings_gate(individual_survivors)
    log_entries.append({"step": "EARNINGS_GATE", "count": len(earnings_survivors)})
    print(f"[1.4] Earnings gate survivors: {[e['ticker'] for e in earnings_survivors]}")

    if not earnings_survivors:
        log_entries.append({"step": "ABORT", "reason": "NO_EARNINGS_GATE_SURVIVORS"})
        risk_auditor.write_outputs([], log_entries)
        await notifier.send_discord_premarket([], log_entries)
        print("[SCHEDULER] ABORT: NO_EARNINGS_GATE_SURVIVORS")
        return

    # STEP 1.5 — PRE-MARKET GAP FETCH (needed by pipeline + leadership gate)
    # Leadership dedup is intentionally deferred until after news fetch so that
    # a high-gap ticker with no catalyst doesn't silently eliminate its entire
    # sub-industry from reaching the paid gate.
    for entry in earnings_survivors:
        entry["pre_market_gap_pct"] = _get_premarket_gap(entry["ticker"])

    # STEP 1.6–1.7 — PAID PIPELINE (all earnings survivors reach here)
    print(f"\n[SCHEDULER] PAID API GATE REACHED — {len(earnings_survivors)} tickers qualifying")
    candidates = await pipeline.run_pipeline(earnings_survivors, regime_data, db_client)
    log_entries.append({"step": "PIPELINE", "qualified_count": len(candidates)})

    if not candidates:
        log_entries.append({"step": "ABORT", "reason": "NO_CANDIDATES_AFTER_PIPELINE"})
        risk_auditor.write_outputs([], log_entries)
        await notifier.send_discord_premarket([], log_entries)
        print("[SCHEDULER] ABORT: NO_CANDIDATES_AFTER_PIPELINE")
        return

    # STEP 1.8 — SECTOR LEADERSHIP GATE (applied post-news so catalyst-bearing
    # tickers compete for leadership, not just gap magnitude)
    leaders = regime_engine.apply_sector_leadership_gate(candidates)
    log_entries.append({"step": "LEADERSHIP_GATE", "count": len(leaders), "leaders": [e["ticker"] for e in leaders]})
    print(f"[1.8] Leaders after pipeline dedup: {[e['ticker'] for e in leaders]}")

    if not leaders:
        log_entries.append({"step": "ABORT", "reason": "NO_LEADERS_IDENTIFIED"})
        risk_auditor.write_outputs([], log_entries)
        await notifier.send_discord_premarket([], log_entries)
        print("[SCHEDULER] ABORT: NO_LEADERS_IDENTIFIED")
        return

    # STEP — SONNET AUDIT + OUTPUT
    monitoring_queue = []
    for i, candidate in enumerate(leaders):
        audit = await risk_auditor.run_sonnet_audit(candidate, client, provider)
        audit = risk_auditor.apply_regime_strategy_mutator(audit, regime_data["regime"])

        stop_dist = abs(candidate["outcome_profile"].get("suggested_stop") or 0.02)
        position = risk_auditor.calculate_position_size(
            account_size=100_000,
            stop_distance_pct=stop_dist,
            regime=regime_data["regime"],
        )

        card = risk_auditor.format_strategy_card(candidate, audit, position)
        strategy_cards.append(card)

        # Write ACTIVE entry to monitoring queue for 9:45 AM live check
        monitoring_queue.append({
            "ticker": candidate["ticker"],
            "status": "ACTIVE",
            "entry_strategy": audit.get("entry_strategy"),
            "eass_score": candidate["eass"]["eass_score"],
            "catalyst_type": candidate["catalyst_type"],
            "card_index": i,
            "timestamp": datetime.datetime.now().isoformat(),
        })

    log_entries.extend(monitoring_queue)
    risk_auditor.write_outputs(strategy_cards, log_entries)
    await notifier.send_discord_premarket(strategy_cards, log_entries)
    print("[SCHEDULER] PRE-MARKET PIPELINE COMPLETE")


async def run_live_volume_check(db_client):
    """9:45 AM live RVOL confirmation gate."""
    print(f"\n[SCHEDULER] Live volume check started — {datetime.datetime.now()}")

    try:
        with open(config.LOG_QUEUE_PATH, "r") as f:
            log_entries = json.load(f)
    except Exception:
        print("[SCHEDULER] No active monitoring queue found — skipping live check")
        return

    active = [e for e in log_entries if e.get("status") == "ACTIVE"]
    if not active:
        print("[SCHEDULER] No ACTIVE tickers in queue")
        return

    for entry in active:
        ticker = entry["ticker"]
        try:
            # Pull 1-minute data for the 9:30–9:45 AM window
            df = yf.download(
                ticker, period="1d", interval="1m", progress=False, auto_adjust=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                vol = df["Volume"][ticker].dropna()
            else:
                vol = df["Volume"].dropna()

            eastern = pytz.timezone("US/Eastern")
            now = datetime.datetime.now(eastern)
            open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
            cutoff = now.replace(hour=9, minute=45, second=0, microsecond=0)

            window_vol = vol[
                (vol.index >= open_time) & (vol.index <= cutoff)
            ]
            interval_volume = float(window_vol.sum()) if not window_vol.empty else 0.0

            # Use 30-day average 15-minute volume as baseline (approximate)
            hist = yf.download(ticker, period="30d", interval="1d", progress=False, auto_adjust=True)
            if isinstance(hist.columns, pd.MultiIndex):
                hist_vol = hist["Volume"][ticker].dropna()
            else:
                hist_vol = hist["Volume"].dropna()
            avg_daily_vol = float(hist_vol.mean()) if not hist_vol.empty else 1.0
            historical_mean_interval = avg_daily_vol * (15 / 390)  # 15 of 390 trading minutes

            interval_rvol = interval_volume / historical_mean_interval if historical_mean_interval > 0 else 0.0

            if interval_rvol < config.INSTITUTIONAL_RVOL_FLOOR:
                entry["status"] = "DOWNGRADED"
                entry["entry_strategy"] = "PULLBACK_LIMIT_21EMA"
                entry["interval_rvol"] = round(interval_rvol, 2)
                print(f"[9:45] {ticker}: RVOL_BELOW_FLOOR ({interval_rvol:.2f}) — downgraded to PULLBACK_LIMIT")
            else:
                entry["rvol_confirmed"] = True
                entry["interval_rvol"] = round(interval_rvol, 2)
                print(f"[9:45] {ticker}: RVOL_CONFIRMED ({interval_rvol:.2f}) — strategy unchanged")

        except Exception as exc:
            print(f"[9:45] {ticker}: volume check error — {exc}")

    with open(config.LOG_QUEUE_PATH, "w") as f:
        json.dump(log_entries, f, indent=2, default=str)

    await notifier.send_discord_rvol(log_entries)
    print("[SCHEDULER] LIVE VOLUME CHECK COMPLETE")


if __name__ == "__main__":
    import pandas as pd  # needed inside run_live_volume_check

    db_client = database.initialize_database()

    schedule.every().day.at("08:05").do(
        lambda: asyncio.run(run_premarket_pipeline(db_client))
    )
    schedule.every().day.at("09:45").do(
        lambda: asyncio.run(run_live_volume_check(db_client))
    )

    print("Scheduler initialized. Waiting for market sessions...")
    while True:
        schedule.run_pending()
        time.sleep(30)
