-- migrations/001_create_setups.sql
-- Idempotent: safe to run multiple times (IF NOT EXISTS throughout)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS historical_setups (
    id                SERIAL PRIMARY KEY,
    doc_id            TEXT UNIQUE NOT NULL,
    ticker            TEXT NOT NULL,
    trade_date        TEXT NOT NULL,
    catalyst_type     TEXT NOT NULL,
    embedding         vector(12) NOT NULL,

    -- Reference text — nullable for historical seed data without news
    news_header       TEXT,
    news_summary      TEXT,

    -- Outcome metrics used by calculate_outcome_profile()
    day1_return       DOUBLE PRECISION NOT NULL,
    day3_return       DOUBLE PRECISION NOT NULL,
    max_adverse_move  DOUBLE PRECISION NOT NULL,
    held_above_21ema  DOUBLE PRECISION NOT NULL,
    regime_at_exit    TEXT NOT NULL,
    exit_trigger      TEXT NOT NULL,

    created_at        TIMESTAMP DEFAULT NOW()
);

-- Hash index for O(1) equality lookup on WHERE catalyst_type = $1
-- (never used for ranges or ordering, so btree's log(n) overhead is wasted)
CREATE INDEX IF NOT EXISTS historical_setups_catalyst_type_idx
    ON historical_setups USING hash (catalyst_type);

-- HNSW approximate nearest-neighbor on the 12-dim embedding
-- Uses L2 distance (<->) to match ChromaDB's default metric
CREATE INDEX IF NOT EXISTS historical_setups_embedding_hnsw_idx
    ON historical_setups USING hnsw (embedding vector_l2_ops);
