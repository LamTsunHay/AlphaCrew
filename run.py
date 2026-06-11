"""CLI entry point for on-demand premarket pipeline execution."""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import llm_client
import yfinance as yf
from dotenv import load_dotenv

import config
import regime_engine
import pipeline
import risk_auditor
import database
import notifier

load_dotenv()

# NASDAQ 100 constituent tickers (as of 2025 — update periodically)
NASDAQ_100 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO",
    "COST", "NFLX", "AMD", "ADBE", "QCOM", "INTC", "INTU", "CSCO", "TXN",
    "AMGN", "HON", "AMAT", "SBUX", "MDLZ", "GILD", "ADI", "REGN", "ISRG",
    "MU", "LRCX", "KLAC", "PANW", "SNPS", "CDNS", "ASML", "MRVL", "FTNT",
    "ORLY", "CTAS", "ABNB", "DXCM", "ADP", "KDP", "MAR", "MNST", "PAYX",
    "PCAR", "ROST", "FAST", "ODFL", "WDAY", "VRSK", "BIIB", "IDXX", "ILMN",
    "ALGN", "ANSS", "CPRT", "DLTR", "EXC", "FANG", "GEHC", "GFS", "KHC",
    "LCID", "LULU", "MCHP", "MRNA", "NXPI", "ON", "RIVN", "SGEN",
    "SIRI", "TEAM", "TTD", "VRSN", "VRTX", "WBA", "ZBRA", "ZM", "ZS",
    "ACGL", "AEP", "AZN", "BKR", "CEG", "CSGP", "CSX", "DDOG", "EA",
    "EBAY", "ENPH", "FSLR", "MELI", "NTES", "OKTA", "PYPL",
    "PTON", "ROP", "SEDG", "TCOM", "WBD", "XEL",
    "ADSK", "CRWD", "DASH", "HUBS", "MSTR", "NET", "PLTR", "SNOW", "SPOT", "UBER",
]


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments; returns Namespace with tickers and regime."""
    parser = argparse.ArgumentParser(
        description="Run the Strategy Engine premarket pipeline on demand."
    )
    parser.add_argument(
        "tickers",
        nargs="*",
        help="Tickers to screen (default: NASDAQ 100)",
    )
    parser.add_argument(
        "--regime",
        choices=["BULLISH", "BEARISH"],
        default=None,
        help="Override the live SPY regime check (for testing)",
    )
    args = parser.parse_args(argv)
    if not args.tickers or args.tickers == ["all"]:
        args.tickers = NASDAQ_100
    return args


def build_regime_data(regime_override: str) -> dict:
    """Return a regime dict using the override value instead of a live SPY check."""
    return {
        "regime": regime_override,
        "spy_close": 0.0,
        "spy_sma50": 0.0,
        "pct_above_50sma": 0.0,
    }


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


async def main(argv=None, test_mode: bool | None = None) -> tuple[list, list]:
    """Orchestrate the full premarket pipeline from CLI args.

    test_mode=True uses Gemini, False uses Claude; None falls back to config.TESTING_MODE.
    Returns (strategy_cards, log_entries) for programmatic callers (e.g. API layer).
    """
    args = parse_args(argv)

    print(f"[RUN] Tickers: {args.tickers}")
    print(f"[RUN] Regime override: {args.regime or 'LIVE (SPY check)'}")

    # Regime gate
    if args.regime:
        regime_data = build_regime_data(args.regime)
        print(f"[RUN] Regime forced to: {args.regime}")
    else:
        regime_data = regime_engine.get_market_regime()
        print(f"[RUN] Live regime: {regime_data['regime']}")

    log_entries = [{"step": "MACRO_GATE", "result": regime_data}]

    # Sector gate
    sector_data = regime_engine.get_top_sectors()
    top_sectors = sector_data["top_sectors"]
    log_entries.append({"step": "SECTOR_GATE", "top_sectors": top_sectors})
    sector_survivors = regime_engine.filter_by_sector(args.tickers, top_sectors)
    log_entries.append({"step": "SECTOR_FILTER", "survivors": sector_survivors})
    print(f"[RUN] Sector survivors: {len(sector_survivors)}/{len(args.tickers)}")

    if not sector_survivors:
        notifier.print_to_terminal([], log_entries)
        return [], log_entries

    # Volatility + trend gate
    individual_survivors = regime_engine.apply_individual_gates(sector_survivors)
    log_entries.append({"step": "INDIVIDUAL_GATES", "count": len(individual_survivors)})
    print(f"[RUN] Individual gate survivors: {len(individual_survivors)}")

    if not individual_survivors:
        notifier.print_to_terminal([], log_entries)
        return [], log_entries

    # Earnings proximity gate
    earnings_survivors = regime_engine.apply_earnings_gate(individual_survivors)
    log_entries.append({"step": "EARNINGS_GATE", "count": len(earnings_survivors)})
    print(f"[RUN] Earnings gate survivors: {len(earnings_survivors)}")

    if not earnings_survivors:
        notifier.print_to_terminal([], log_entries)
        return [], log_entries

    # Pre-market gap fetch — needed by pipeline and post-pipeline leadership gate.
    # Leadership dedup is deferred until after news fetch so a high-gap ticker
    # with no catalyst doesn't silently eliminate its sub-industry.
    tickers_for_gap = [e["ticker"] for e in earnings_survivors]
    with ThreadPoolExecutor(max_workers=20) as executor:
        gap_values = list(executor.map(_get_premarket_gap, tickers_for_gap))
    for entry, gap in zip(earnings_survivors, gap_values):
        entry["pre_market_gap_pct"] = gap

    # Paid pipeline gate — all earnings survivors enter
    db_client = database.initialize_database()
    candidates = await pipeline.run_pipeline(earnings_survivors, regime_data, db_client,
                                              test_mode=test_mode)
    log_entries.append({"step": "PIPELINE", "qualified_count": len(candidates)})

    if not candidates:
        notifier.print_to_terminal([], log_entries)
        return [], log_entries

    # Sector leadership gate — applied post-pipeline so only catalyst-bearing
    # tickers compete for each industry slot
    leaders = regime_engine.apply_sector_leadership_gate(candidates)
    log_entries.append({"step": "LEADERSHIP_GATE", "leaders": [e["ticker"] for e in leaders]})
    print(f"[RUN] Leaders after pipeline dedup: {[e['ticker'] for e in leaders]}")

    if not leaders:
        notifier.print_to_terminal([], log_entries)
        return [], log_entries

    # Sonnet audit + strategy cards
    client, provider = llm_client.create_client(test_mode=test_mode)
    strategy_cards = []
    for candidate in leaders:
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
        log_entries.append({"ticker": candidate["ticker"], "status": "QUALIFIED"})

    notifier.print_to_terminal(strategy_cards, log_entries)
    return strategy_cards, log_entries


if __name__ == "__main__":
    asyncio.run(main())
