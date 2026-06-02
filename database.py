"""PostgreSQL pgvector setup for Strategy Engine v5.1.

Public interface (unchanged from ChromaDB version):
  initialize_database(dsn=None) -> psycopg2 connection
  store_setup(conn, catalyst_type, regime_data, catalyst_data, outcome,
              news_header=None, news_summary=None)
  query_similar_setups(conn, catalyst_type, regime_data, catalyst_data) -> dict
  calculate_outcome_profile(matches) -> dict

The connection returned by initialize_database() is passed as the first
argument to store_setup() and query_similar_setups() — callers treat it
as an opaque handle, identical to how the old ChromaDB client was used.
"""

import os
import datetime
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector
import numpy as np
import config

# SQL DDL embedded here so initialize_database() is self-contained.
# All statements use IF NOT EXISTS — safe to run on every startup.
_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS setups (
    id                SERIAL PRIMARY KEY,
    doc_id            TEXT UNIQUE NOT NULL,
    ticker            TEXT NOT NULL,
    trade_date        TEXT NOT NULL,
    catalyst_type     TEXT NOT NULL,
    embedding         vector(12) NOT NULL,
    news_header       TEXT,
    news_summary      TEXT,
    day1_return       DOUBLE PRECISION NOT NULL,
    day3_return       DOUBLE PRECISION NOT NULL,
    max_adverse_move  DOUBLE PRECISION NOT NULL,
    held_above_21ema  DOUBLE PRECISION NOT NULL,
    regime_at_exit    TEXT NOT NULL,
    exit_trigger      TEXT NOT NULL,
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS setups_catalyst_type_idx
    ON setups (catalyst_type);

CREATE INDEX IF NOT EXISTS setups_embedding_hnsw_idx
    ON setups USING hnsw (embedding vector_l2_ops);
"""


def normalize(value, min_val, max_val):
    """Clip-normalize value to [0.0, 1.0]."""
    if max_val == min_val:
        return 0.0
    return float(np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0))


def initialize_database(dsn: str = None):
    """Connect to PostgreSQL, apply schema, and return an open connection.

    Reads DATABASE_URL from the environment when dsn is not supplied.
    Registers the pgvector type so Python lists are accepted as vector(12).
    """
    dsn = dsn or os.getenv("DATABASE_URL", config.POSTGRES_DSN)
    conn = psycopg2.connect(dsn)
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(_SCHEMA_SQL)
    conn.commit()
    print("[DB] PostgreSQL connected; setups table ready")
    return conn


def build_regime_vector(regime_data: dict) -> list:
    """Build an 8-dimensional normalized regime vector."""
    spy_pct = regime_data.get("spy_pct_above_50sma", 0.0)
    sector_5d = regime_data.get("sector_5d_vs_spy", 0.0)
    vix = regime_data.get("vix_level", 20.0)
    sector_rank = regime_data.get("sector_rank", 2.0)
    pct_above_200 = regime_data.get("pct_above_200ema", 0.0)
    gap_pct = regime_data.get("premarket_gap_pct", 0.015)
    rvol = regime_data.get("rvol_945", 1.0)
    quarter = regime_data.get("quarter", 1)

    return [
        normalize(spy_pct, -0.15, 0.15),
        normalize(sector_5d, -0.05, 0.05),
        1.0 - normalize(vix, 10, 45),          # inverted: lower VIX → higher score
        1.0 - normalize(sector_rank, 1, 3),     # inverted: rank 1 → highest score
        normalize(pct_above_200, 0, 0.30),
        normalize(gap_pct, 0.015, 0.20),
        normalize(rvol, 1.0, 15.0),
        normalize(quarter, 1, 4),
    ]


def build_catalyst_vector(catalyst_data: dict, catalyst_type: str) -> list:
    """Build a 4-dimensional universal catalyst vector.

    Dims are catalyst-type-agnostic so vectors are comparable across types:
      [0] surprise_magnitude  — weighted deviation from market expectation [0, 1]
      [1] surprise_direction  — 1.0 = positive surprise, 0.0 = negative [0, 1]
      [2] forward_signal      — implication for future cash flows [0, 1]
      [3] business_impact     — event size relative to company valuation [0, 1]

    Valuation priority for earnings: guidance > revenue > EPS, because guidance
    directly updates the forward DCF model while EPS is susceptible to buybacks
    and one-time charges.
    """
    ct = catalyst_type.lower()

    # --- EARNINGS / REVENUE ---
    if ct.startswith("earnings") or ct.startswith("revenue"):
        eps  = catalyst_data.get("eps_surprise_pct", 0.0)
        rev  = catalyst_data.get("revenue_surprise_pct", 0.0)
        guid = catalyst_data.get("guidance_delta_pct", 0.0)
        # Weighted magnitude: guidance 50%, revenue 35%, EPS 15%
        # Max raw ≈ 0.50×0.20 + 0.35×0.30 + 0.15×0.50 = 0.28
        raw = 0.50 * abs(guid) + 0.35 * abs(rev) + 0.15 * abs(eps)
        return [
            normalize(raw, 0, 0.28),
            0.0 if ct == "earnings_miss" else 1.0,
            normalize(guid, -0.20, 0.20),
            normalize(abs(guid) + abs(eps), 0, 0.70),
        ]

    # --- GUIDANCE ---
    if ct.startswith("guidance"):
        eps_g = catalyst_data.get("eps_guidance_delta_pct", 0.0)
        rev_g = catalyst_data.get("revenue_guidance_delta_pct", 0.0)
        # Max raw ≈ 0.60×0.30 + 0.40×0.20 = 0.26
        raw = 0.60 * abs(eps_g) + 0.40 * abs(rev_g)
        return [
            normalize(raw, 0, 0.26),
            0.0 if ct == "guidance_cut" else 1.0,
            normalize(eps_g, -0.30, 0.30),
            normalize(abs(eps_g) + abs(rev_g) * 0.5, 0, 0.45),
        ]

    # --- FDA ---
    if ct.startswith("fda"):
        was_expected  = float(bool(catalyst_data.get("was_expected", False)))
        market_cap    = catalyst_data.get("market_cap_billions", 10.0)
        pipeline      = catalyst_data.get("pipeline_depth", 1)
        # Smaller company + unexpected event = larger surprise_magnitude
        impact_factor = 1.0 - normalize(market_cap, 0.1, 50)
        return [
            (1.0 - was_expected) * 0.6 + impact_factor * 0.4,
            0.0 if ct == "fda_rejection" else 1.0,
            normalize(pipeline, 1, 20),
            impact_factor,
        ]

    # --- M&A TARGET ---
    if ct == "ma_target":
        premium   = catalyst_data.get("premium_pct", 0.0)
        cash_deal = float(bool(catalyst_data.get("cash_deal", False)))
        hostile   = float(bool(catalyst_data.get("hostile_bid", False)))
        # Premium IS the valuation change for the target
        mag = normalize(premium, 0, 0.60)
        return [
            mag,
            1.0,                                        # target is always positive
            cash_deal * 0.7 + (1.0 - hostile) * 0.3,   # certainty of capturing premium
            mag,                                        # business_impact == premium
        ]

    # --- M&A ACQUIRER ---
    if ct == "ma_acquirer":
        deal_size = catalyst_data.get("deal_size_billions", 1.0)
        cash_deal = float(bool(catalyst_data.get("cash_deal", False)))
        hostile   = float(bool(catalyst_data.get("hostile_bid", False)))
        mag = normalize(deal_size, 0.1, 100)
        return [
            mag,
            0.5,                                    # acquirer impact is mixed
            cash_deal * 0.5 + (1.0 - hostile) * 0.5,
            mag,
        ]

    # --- BUYBACK ---
    if ct.startswith("buyback"):
        deal_size = catalyst_data.get("deal_size_billions", 1.0)
        mag = normalize(deal_size, 0.1, 50)
        return [mag, 1.0, mag, mag]

    # --- GOVERNMENT / COMMERCIAL CONTRACT ---
    if ct.startswith("government") or ct.startswith("commercial"):
        value      = catalyst_data.get("contract_value_millions", 10.0)
        multi_year = float(bool(catalyst_data.get("multi_year", False)))
        sole_src   = float(bool(catalyst_data.get("sole_source", False)))
        margin     = catalyst_data.get("margin_impact_pct", 0.0)
        return [
            normalize(value, 1, 5000),
            1.0,                                  # contract win is always positive
            multi_year * 0.5 + sole_src * 0.5,   # recurring / locked revenue
            normalize(margin, 0, 0.05),
        ]

    # Fallback for unknown catalyst types
    return [0.5, 0.5, 0.5, 0.5]


def store_setup(
    conn,
    catalyst_type: str,
    regime_data: dict,
    catalyst_data: dict,
    outcome: dict,
    news_header: str = None,
    news_summary: str = None,
):
    """Build a 12-dim vector and upsert one row into setups.

    ticker and date are read from outcome (same convention as the old ChromaDB version).
    ON CONFLICT DO NOTHING makes repeated calls with the same data safe.
    news_header and news_summary are optional — NULL for historical seed data.
    """
    regime_vec = build_regime_vector(regime_data)
    catalyst_vec = build_catalyst_vector(catalyst_data, catalyst_type)
    full_vector = regime_vec + catalyst_vec

    ticker   = str(outcome.get("ticker") or "UNKNOWN")
    date_str = str(outcome.get("date") or datetime.date.today().isoformat())
    doc_id   = f"{ticker}_{date_str}_{hash(str(full_vector)) % 1_000_000:06d}"

    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO setups
                (doc_id, ticker, trade_date, catalyst_type, embedding,
                 news_header, news_summary,
                 day1_return, day3_return, max_adverse_move,
                 held_above_21ema, regime_at_exit, exit_trigger)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (doc_id) DO NOTHING
            """,
            (
                doc_id, ticker, date_str, catalyst_type, full_vector,
                news_header, news_summary,
                float(outcome.get("day1_return") or 0.0),
                float(outcome.get("day3_return") or 0.0),
                float(outcome.get("max_adverse_move") or 0.0),
                float(bool(outcome.get("held_above_21ema", False))),
                str(outcome.get("regime_at_exit") or "UNKNOWN"),
                str(outcome.get("exit_trigger") or "UNKNOWN"),
            ),
        )
    conn.commit()


def query_similar_setups(client, catalyst_type: str, regime_data: dict, catalyst_data: dict) -> dict:
    """Query ChromaDB for similar historical setups and return an outcome profile."""
    regime_vec = build_regime_vector(regime_data)
    catalyst_vec = build_catalyst_vector(catalyst_data, catalyst_type)
    query_vector = regime_vec + catalyst_vec

    collection = client.get_or_create_collection(catalyst_type)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=80,
        include=["metadatas", "distances"],
    )

    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    filtered = [
        m for m, d in zip(metadatas, distances)
        if d < config.VECTOR_DB_MAX_DISTANCE
    ]

    return calculate_outcome_profile(filtered)


def calculate_outcome_profile(matches: list) -> dict:
    """Compute statistical outcome profile from a list of similar historical setups."""
    if len(matches) < config.VECTOR_DB_MIN_SAMPLES:
        return {
            "confidence": "INSUFFICIENT",
            "sample_size": len(matches),
            "action": "SKIP_TRADE",
            "reason": "Insufficient historical samples for reliable probability estimate.",
            "win_rate_day1": None,
            "win_rate_day3": None,
            "expected_value": None,
            "suggested_stop": None,
            "suggested_target": None,
        }

    day1_returns = [float(m["day1_return"]) for m in matches]
    day3_returns = [float(m["day3_return"]) for m in matches]
    adverse_moves = [float(m["max_adverse_move"]) for m in matches]

    winners_d3 = [r for r in day3_returns if r > 0]
    losers_d3 = [r for r in day3_returns if r <= 0]

    win_rate_d1 = float(np.mean([r > 0 for r in day1_returns]))
    win_rate_d3 = float(np.mean([r > 0 for r in day3_returns]))
    avg_gain = float(np.mean(winners_d3)) if winners_d3 else 0.0
    avg_loss = float(abs(np.mean(losers_d3))) if losers_d3 else 0.0
    loss_rate = 1.0 - win_rate_d3
    expected_value = (win_rate_d3 * avg_gain) - (loss_rate * avg_loss)

    n = len(matches)
    # Note: n_results=80 in query_similar_setups means HIGH (>=150) is unreachable
    # via the standard query path. Kept as-is per spec.
    if n >= 150:
        confidence = "HIGH"
    elif n >= 80:
        confidence = "MODERATE"
    else:
        confidence = "LOW"

    held_vals = [float(m.get("held_above_21ema", 0.0)) for m in matches]

    return {
        "confidence": confidence,
        "sample_size": n,
        "action": "PROCEED" if expected_value > 0 else "SKIP_TRADE",
        "win_rate_day1": round(win_rate_d1, 4),
        "win_rate_day3": round(win_rate_d3, 4),
        "avg_gain_winners": round(avg_gain, 4),
        "avg_loss_losers": round(avg_loss, 4),
        "expected_value": round(expected_value, 4),
        "avg_max_adverse": round(float(np.mean(adverse_moves)), 4),
        "adverse_95pct": round(float(np.percentile(adverse_moves, 95)), 4),
        "pct_held_above_21ema": round(float(np.mean(held_vals)), 4),
        "suggested_stop": round(-float(np.percentile(adverse_moves, 75)), 4),
        "suggested_target": round(float(np.mean(winners_d3)), 4) if winners_d3 else None,
    }
