"""Real historical data importer for Strategy Engine v5.1.

Reads a CSV of past catalyst events, fetches price history from yfinance (free),
computes genuine outcome metrics, and stores them via database.store_setup().

CSV format: see sample_events.csv for the full column template.
Required columns: ticker, date (YYYY-MM-DD), catalyst_type
Optional columns: all catalyst feature columns (see sample_events.csv)

Usage:
    python import_history.py --input sample_events.csv
    python import_history.py --input my_events.csv --dry-run
"""

import argparse
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import database
import config


def _get_price_data(ticker: str, event_date: datetime.date) -> pd.DataFrame | None:
    """Download daily price history around an event date (+/- 30 trading days)."""
    start = event_date - datetime.timedelta(days=60)
    end = event_date + datetime.timedelta(days=30)
    try:
        df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=1)
        return df.dropna()
    except Exception as exc:
        print(f"  [IMPORT] Price fetch failed for {ticker}: {exc}")
        return None


def _compute_outcomes(df: pd.DataFrame, event_date: datetime.date) -> dict | None:
    """Compute day1/day3 returns and adverse move from price data relative to event date."""
    # Find the closest trading day on or after the event date
    dates = pd.to_datetime(df.index).date
    future = [d for d in dates if d >= event_date]
    if not future:
        return None

    event_idx = list(dates).index(future[0])
    close = df["Close"].values

    if event_idx >= len(close):
        return None

    event_close = float(close[event_idx])

    day1_return = None
    day3_return = None
    max_adverse_move = 0.0

    if event_idx + 1 < len(close):
        day1_return = (float(close[event_idx + 1]) - event_close) / event_close

    if event_idx + 3 < len(close):
        day3_return = (float(close[event_idx + 3]) - event_close) / event_close

    # Max adverse move: worst intraday low in 3 days after event
    if "Low" in df.columns:
        window_low = df["Low"].values[event_idx + 1: event_idx + 4]
        if len(window_low) > 0:
            max_adverse_move = (float(np.min(window_low)) - event_close) / event_close

    # EMA21 at event
    ema21_series = df["Close"].ewm(span=21, adjust=False).mean()
    ema21_at_event = float(ema21_series.iloc[event_idx])
    held_above_21ema = bool(event_close > ema21_at_event)

    return {
        "day1_return": round(day1_return, 5) if day1_return is not None else 0.0,
        "day3_return": round(day3_return, 5) if day3_return is not None else 0.0,
        "max_adverse_move": round(abs(max_adverse_move), 5),
        "held_above_21ema": held_above_21ema,
        "regime_at_exit": "UNKNOWN",
        "exit_trigger": "DATA_IMPORT",
    }


def _get_spy_regime(event_date: datetime.date) -> dict:
    """Reconstruct basic SPY regime data around an event date."""
    start = event_date - datetime.timedelta(days=80)
    try:
        df = yf.download("SPY", start=start.isoformat(),
                         end=(event_date + datetime.timedelta(days=2)).isoformat(),
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            close = df["Close"]["SPY"].dropna()
        else:
            close = df["Close"].dropna()

        sma50 = close.rolling(50).mean()
        latest_close = float(close.iloc[-1])
        latest_sma = float(sma50.iloc[-1])
        pct_above = (latest_close - latest_sma) / latest_sma
    except Exception:
        return {
            "spy_pct_above_50sma": 0.0, "sector_5d_vs_spy": 0.0, "vix_level": 20.0,
            "sector_rank": 2, "pct_above_200ema": 0.0, "premarket_gap_pct": 0.03,
            "rvol_945": 3.0, "quarter": ((event_date.month - 1) // 3) + 1,
        }

    return {
        "spy_pct_above_50sma": round(pct_above, 5),
        "sector_5d_vs_spy": 0.0,  # requires sector data; default to neutral
        "vix_level": 18.0,         # approximate neutral VIX
        "sector_rank": 2,
        "pct_above_200ema": 0.0,
        "premarket_gap_pct": 0.03,
        "rvol_945": 3.0,
        "quarter": ((event_date.month - 1) // 3) + 1,
    }


def _row_to_catalyst_data(row: pd.Series) -> dict:
    """Extract catalyst_data dict from a CSV row, safely handling NaN."""
    def f(col, default=0.0):
        v = row.get(col, default)
        return default if pd.isna(v) else float(v)

    def b(col, default=False):
        v = row.get(col, default)
        return default if pd.isna(v) else bool(int(float(v)))

    return {
        "eps_surprise_pct": f("eps_surprise_pct"),
        "revenue_surprise_pct": f("revenue_surprise_pct"),
        "guidance_delta_pct": f("guidance_delta_pct"),
        "analyst_revision_count": int(f("analyst_revision_count")),
        "ceo_statement_positive": b("ceo_statement_positive"),
        "eps_guidance_delta_pct": f("eps_guidance_delta_pct"),
        "revenue_guidance_delta_pct": f("revenue_guidance_delta_pct"),
        "days_to_next_earnings": int(f("days_to_next_earnings", 45)),
        "was_expected": b("was_expected"),
        "has_competitor": b("has_competitor"),
        "market_cap_billions": f("market_cap_billions", 5.0),
        "pipeline_depth": int(f("pipeline_depth", 3)),
        "premium_pct": f("premium_pct"),
        "cash_deal": b("cash_deal"),
        "deal_size_billions": f("deal_size_billions", 1.0),
        "hostile_bid": b("hostile_bid"),
        "contract_value_millions": f("contract_value_millions", 100.0),
        "multi_year": b("multi_year"),
        "sole_source": b("sole_source"),
        "margin_impact_pct": f("margin_impact_pct"),
    }


def import_events(csv_path: str, dry_run: bool = False):
    """Main import routine: read events CSV, fetch prices, compute outcomes, store."""
    df = pd.read_csv(csv_path)
    required = {"ticker", "date", "catalyst_type"}
    missing = required - set(df.columns)
    if missing:
        print(f"[IMPORT] Error: missing required columns: {missing}")
        return

    db_client = database.initialize_database()
    imported = skipped = 0

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip().upper()
        catalyst_type = str(row["catalyst_type"]).strip()
        try:
            event_date = datetime.date.fromisoformat(str(row["date"]).strip())
        except ValueError:
            print(f"  [IMPORT] Bad date for {ticker}: {row['date']} — skipping")
            skipped += 1
            continue

        if catalyst_type not in config.CATALYST_COLLECTIONS:
            print(f"  [IMPORT] Unknown catalyst_type '{catalyst_type}' for {ticker} — skipping")
            skipped += 1
            continue

        print(f"  Processing {ticker} {event_date} ({catalyst_type})...")

        price_df = _get_price_data(ticker, event_date)
        if price_df is None or len(price_df) < 5:
            print(f"  [IMPORT] Insufficient price data for {ticker} — skipping")
            skipped += 1
            continue

        outcomes = _compute_outcomes(price_df, event_date)
        if outcomes is None:
            print(f"  [IMPORT] Could not compute outcomes for {ticker} — skipping")
            skipped += 1
            continue

        regime_data = _get_spy_regime(event_date)
        catalyst_data = _row_to_catalyst_data(row)
        outcome = {**outcomes, "ticker": ticker, "date": event_date.isoformat()}

        if dry_run:
            print(f"  [DRY-RUN] Would store: day1={outcomes['day1_return']:.3f}, "
                  f"day3={outcomes['day3_return']:.3f}, adverse={outcomes['max_adverse_move']:.3f}")
        else:
            database.store_setup(db_client, catalyst_type, regime_data, catalyst_data, outcome)
            print(f"  Stored: day1={outcomes['day1_return']:.3f}, day3={outcomes['day3_return']:.3f}")

        imported += 1

    action = "Would import" if dry_run else "Imported"
    print(f"\n[IMPORT] {action} {imported} records, skipped {skipped}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import real historical catalyst events into ChromaDB")
    parser.add_argument("--input", required=True, help="Path to events CSV file")
    parser.add_argument("--dry-run", action="store_true", help="Parse and compute without writing to DB")
    args = parser.parse_args()

    import_events(args.input, dry_run=args.dry_run)
