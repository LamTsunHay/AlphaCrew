"""Global constants and thresholds for Strategy Engine v5.1.

Single source of truth. No logic here — values only.
"""

import os

SYSTEM_TIMEZONE = "US/Eastern"
STRATEGY_OUTPUT_PATH = "./final_strategy_blueprint.md"
LOG_QUEUE_PATH = "./active_monitoring_queue.json"

# LLM Models
LLM_STAGE_3_FAST = "claude-haiku-4-5-20251001"
LLM_STAGE_4_PREMIUM = "claude-sonnet-4-6"
LLM_MAX_TOKENS = 1000

# Testing mode — set True to route both LLM stages to Groq (free tier)
TESTING_MODE        = False
GROQ_API_KEY        = os.environ.get("GROQ_API_KEY", "")
GROQ_BASE_URL       = "https://api.groq.com/openai/v1"
GROQ_STAGE_3_MODEL  = "llama-3.1-8b-instant"     # replaces Haiku
GROQ_STAGE_4_MODEL  = "llama-3.3-70b-versatile"  # replaces Sonnet

# Gemini Flash — free-tier catalyst classifier
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
GEMINI_CLASSIFY_MODEL = "gemini-3.1-flash-lite"

# Regime Filter
REGIME_INDICATOR = "SPY"
REGIME_SMA_PERIOD = 50

# Volatility and Trend Gates
VOLATILITY_CEILING_RATIO = 0.10
TREND_EMA_FAST = 21
TREND_EMA_SLOW = 200

# Institutional Volume Gates
INSTITUTIONAL_RVOL_FLOOR = 3.0
LIQUIDITY_FLOOR_ADV_30D = 2_000_000
PRE_MARKET_MIN_VOLUME = 50_000
PRE_MARKET_MIN_GAP_PCT = 0.015

# Earnings Proximity Gate
EARNINGS_EXCLUSION_DAYS = 5

# Risk Parameters
STANDARD_ACCOUNT_RISK = 0.010
BEARISH_REGIME_MUTATOR = 0.500

# EASS Weights (3-component version — options component reserved for future)
EASS_WEIGHTS = {
    "analyst_surprise": 0.45,
    "whisper_surprise": 0.00,   # FUTURE_STUB — weight is 0 until whisper feed added
    "guidance_delta": 0.55,
}

# Shock Weight Matrix (used for catalyst type classification only — not EASS math)
SHOCK_WEIGHT_MATRIX = {
    "corporate_guidance": 1.00,
    "mergers_acquisitions": 0.95,
    "earnings": 0.90,
    "fda_regulatory": 0.90,
    "analyst_color": 0.40,
    "market_movers": 0.30,
}

# PostgreSQL pgvector
POSTGRES_DSN = os.environ.get("DATABASE_URL", "")
VECTOR_DB_MIN_SAMPLES = 40
VECTOR_DB_MAX_DISTANCE = 0.35

# Valid catalyst type names — used for validation only (no longer one-per-collection)
CATALYST_COLLECTIONS = [
    "earnings_beat_large",
    "earnings_beat_small",
    "earnings_miss",
    "revenue_beat_only",
    "guidance_raise_full",
    "guidance_raise_partial",
    "guidance_cut",
    "fda_approval_nda",
    "fda_approval_fast_track",
    "fda_rejection",
    "ma_acquirer",
    "ma_target",
    "buyback_initiation",
    "government_contract",
    "commercial_contract",
]

# Catalyst types ordered from highest to lowest impact priority.
# Mirrors the if-elif chain in pipeline.classify_catalyst_type().
# Note: CATALYST_COLLECTIONS contains additional types (earnings_beat_small,
# earnings_miss, guidance_raise_partial, revenue_beat_only) reserved for future
# classify_catalyst_type() branches. If you add a branch, add the type here too.
CATALYST_PRIORITY = (
    "ma_acquirer",
    "ma_target",
    "fda_approval_nda",
    "fda_approval_fast_track",
    "fda_rejection",
    "buyback_initiation",
    "government_contract",
    "commercial_contract",
    "guidance_raise_full",
    "guidance_cut",
    "earnings_beat_large",
    "market_movers",   # fallback — not a storage collection, never stored in DB
)

# News provider — "polygon" or "finnhub"
NEWS_PROVIDER = "finnhub"
FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY", "")
FINNHUB_NEWS_URL = "https://finnhub.io/api/v1/company-news"

# Sector ETF universe for rotation filter
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLP": "Consumer Staples",
    "XLY": "Consumer Discretionary",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}
TOP_SECTOR_COUNT = 3
SECTOR_LOOKBACK_DAYS = 5

# Discord webhook delivery
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")
DISCORD_ENABLED = bool(DISCORD_WEBHOOK_URL)
