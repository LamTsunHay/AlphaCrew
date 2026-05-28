"""Synthetic database seeder for Strategy Engine v5.1.

Populates ChromaDB with deterministic synthetic historical setups so the pipeline
can clear the 40-sample confidence floor and emit real strategy cards during testing.
No network calls — pure local ChromaDB writes.

Usage:
    python seed_database.py                  # 120 samples per collection (default)
    python seed_database.py --samples 200    # override samples per collection
"""

import argparse
import random
import datetime
import database
import config


def _synthetic_regime(rng: random.Random, bullish_bias: float = 0.65) -> dict:
    """Generate a plausible synthetic regime_data dict."""
    is_bull = rng.random() < bullish_bias
    spy_pct = rng.uniform(0.01, 0.08) if is_bull else rng.uniform(-0.08, -0.01)
    return {
        "spy_pct_above_50sma": spy_pct,
        "sector_5d_vs_spy": rng.uniform(-0.03, 0.04),
        "vix_level": rng.uniform(12, 22) if is_bull else rng.uniform(20, 35),
        "sector_rank": rng.choice([1, 1, 2, 2, 3]),
        "pct_above_200ema": rng.uniform(0.0, 0.20) if is_bull else rng.uniform(-0.05, 0.05),
        "premarket_gap_pct": rng.uniform(0.02, 0.12),
        "rvol_945": rng.uniform(2.0, 12.0),
        "quarter": rng.randint(1, 4),
    }


def _synthetic_catalyst(rng: random.Random, catalyst_type: str, win_bias: float = 0.60) -> dict:
    """Generate plausible catalyst_data matching the catalyst_type branch."""
    ct = catalyst_type.lower()
    if ct.startswith("earnings") or ct.startswith("revenue"):
        surprise = rng.uniform(0.02, 0.20) if rng.random() < win_bias else rng.uniform(-0.15, 0.0)
        return {
            "eps_surprise_pct": surprise,
            "revenue_surprise_pct": surprise * rng.uniform(0.5, 1.3),
            "guidance_delta_pct": rng.uniform(0.0, 0.10) if rng.random() < 0.6 else rng.uniform(-0.10, 0.0),
            "analyst_revision_count": rng.randint(0, 15),
        }
    if ct.startswith("guidance"):
        return {
            "eps_guidance_delta_pct": rng.uniform(0.01, 0.15) if rng.random() < win_bias else rng.uniform(-0.15, 0.0),
            "revenue_guidance_delta_pct": rng.uniform(0.01, 0.10),
            "ceo_statement_positive": rng.random() < 0.7,
            "days_to_next_earnings": rng.randint(20, 75),
        }
    if ct.startswith("fda"):
        return {
            "was_expected": rng.random() < 0.5,
            "has_competitor": rng.random() < 0.4,
            "market_cap_billions": rng.uniform(0.5, 30),
            "pipeline_depth": rng.randint(1, 12),
        }
    if ct.startswith("ma") or ct.startswith("buyback"):
        return {
            "premium_pct": rng.uniform(0.10, 0.45),
            "cash_deal": rng.random() < 0.55,
            "deal_size_billions": rng.uniform(0.5, 50),
            "hostile_bid": rng.random() < 0.15,
        }
    # government_contract / commercial_contract / fallback
    return {
        "contract_value_millions": rng.uniform(50, 3000),
        "multi_year": rng.random() < 0.6,
        "sole_source": rng.random() < 0.4,
        "margin_impact_pct": rng.uniform(0.005, 0.04),
    }


def _synthetic_outcome(rng: random.Random, win_bias: float = 0.60) -> dict:
    """Generate a plausible historical outcome."""
    win = rng.random() < win_bias
    day1 = rng.uniform(0.005, 0.08) if win else rng.uniform(-0.06, -0.005)
    day3 = day1 + rng.uniform(-0.03, 0.05)
    adverse = rng.uniform(0.005, abs(day1) * 1.5 + 0.01)
    ticker = rng.choice(["NVDA", "MSFT", "AAPL", "AMD", "META", "AMZN", "GOOGL", "TSLA",
                         "CRM", "NFLX", "UBER", "PYPL", "SHOP", "SNOW", "PLTR", "PANW"])
    days_ago = rng.randint(10, 730)
    date_str = (datetime.date.today() - datetime.timedelta(days=days_ago)).isoformat()
    return {
        "ticker": ticker,
        "date": date_str,
        "day1_return": round(day1, 5),
        "day3_return": round(day3, 5),
        "max_adverse_move": round(adverse, 5),
        "held_above_21ema": win and rng.random() < 0.7,
        "regime_at_exit": "BULLISH" if rng.random() < 0.65 else "BEARISH",
        "exit_trigger": rng.choice(["STOP", "TARGET", "TIME", "REGIME_CHANGE"]),
    }


def seed(samples_per_collection: int = 120, seed_val: int = 42):
    """Seed all catalyst collections with synthetic historical setups."""
    db_client = database.initialize_database()
    rng = random.Random(seed_val)

    total = 0
    for collection_name in config.CATALYST_COLLECTIONS:
        for i in range(samples_per_collection):
            regime = _synthetic_regime(rng)
            catalyst = _synthetic_catalyst(rng, collection_name)
            outcome = _synthetic_outcome(rng)
            # Give outcome the right catalyst_type for metadata
            outcome_with_type = {**outcome, "catalyst_type": collection_name}
            database.store_setup(db_client, collection_name, regime, catalyst, outcome_with_type)
        total += samples_per_collection
        print(f"  Seeded {samples_per_collection} samples → {collection_name}")

    print(f"\n[SEED] Done. {total} total records written across {len(config.CATALYST_COLLECTIONS)} collections.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed ChromaDB with synthetic historical data")
    parser.add_argument("--samples", type=int, default=120, help="Samples per collection (default: 120)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print(f"[SEED] Seeding {args.samples} samples per collection (seed={args.seed})...")
    seed(samples_per_collection=args.samples, seed_val=args.seed)
