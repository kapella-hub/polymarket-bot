-- Polymarket Trading Bot - Database Schema
-- Run against PostgreSQL 15+

CREATE TABLE IF NOT EXISTS markets (
    id VARCHAR(128) PRIMARY KEY,
    question TEXT NOT NULL,
    category VARCHAR(64),
    end_date TIMESTAMPTZ,
    status VARCHAR(16) NOT NULL DEFAULT 'active',
    clob_token_id_yes VARCHAR(128) NOT NULL,
    clob_token_id_no VARCHAR(128) NOT NULL,
    outcome_yes VARCHAR(128) NOT NULL DEFAULT 'Yes',
    outcome_no VARCHAR(128) NOT NULL DEFAULT 'No',
    volume FLOAT NOT NULL DEFAULT 0.0,
    liquidity FLOAT NOT NULL DEFAULT 0.0,
    best_bid FLOAT,
    best_ask FLOAT,
    last_price FLOAT,
    description TEXT,
    resolution_source TEXT,
    tags JSONB,
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_markets_status ON markets(status);
CREATE INDEX IF NOT EXISTS ix_markets_category ON markets(category);
CREATE INDEX IF NOT EXISTS ix_markets_volume ON markets(volume);

CREATE TABLE IF NOT EXISTS market_signals (
    id SERIAL PRIMARY KEY,
    market_id VARCHAR(128) NOT NULL,
    probability FLOAT NOT NULL,
    confidence FLOAT NOT NULL,
    edge_over_market FLOAT NOT NULL,
    reasoning TEXT,
    key_factors JSONB,
    market_price_at_eval FLOAT NOT NULL,
    model_version VARCHAR(64) NOT NULL DEFAULT 'claude-cli',
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_signals_market_evaluated ON market_signals(market_id, evaluated_at);
CREATE INDEX IF NOT EXISTS ix_signals_expires ON market_signals(expires_at);

CREATE TABLE IF NOT EXISTS order_intents (
    id SERIAL PRIMARY KEY,
    market_id VARCHAR(128) NOT NULL,
    clob_token_id VARCHAR(128) NOT NULL,
    side VARCHAR(8) NOT NULL,
    price FLOAT NOT NULL,
    size FLOAT NOT NULL,
    strategy VARCHAR(32) NOT NULL,
    signal_id INTEGER,
    edge_at_creation FLOAT NOT NULL,
    confidence_at_creation FLOAT NOT NULL,
    state VARCHAR(16) NOT NULL DEFAULT 'created',
    invalidation_reason VARCHAR(32),
    exchange_order_id VARCHAR(256),
    filled_price FLOAT,
    filled_size FLOAT,
    fill_timestamp TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS ix_intents_state ON order_intents(state);
CREATE INDEX IF NOT EXISTS ix_intents_market_state ON order_intents(market_id, state);

CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    market_id VARCHAR(128) NOT NULL,
    clob_token_id VARCHAR(128) NOT NULL,
    outcome VARCHAR(128) NOT NULL,
    size FLOAT NOT NULL DEFAULT 0.0,
    avg_entry_price FLOAT NOT NULL DEFAULT 0.0,
    cost_basis FLOAT NOT NULL DEFAULT 0.0,
    realized_pnl FLOAT NOT NULL DEFAULT 0.0,
    unrealized_pnl FLOAT NOT NULL DEFAULT 0.0,
    opened_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS ix_positions_token ON positions(clob_token_id);

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    market_id VARCHAR(128) NOT NULL,
    clob_token_id VARCHAR(128) NOT NULL,
    intent_id INTEGER,
    side VARCHAR(8) NOT NULL,
    price FLOAT NOT NULL,
    size FLOAT NOT NULL,
    fee FLOAT NOT NULL DEFAULT 0.0,
    strategy VARCHAR(32) NOT NULL,
    exchange_order_id VARCHAR(256) NOT NULL,
    executed_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_trades_executed ON trades(executed_at);

CREATE TABLE IF NOT EXISTS risk_snapshots (
    id SERIAL PRIMARY KEY,
    portfolio_value FLOAT NOT NULL,
    total_exposure FLOAT NOT NULL,
    drawdown_pct FLOAT NOT NULL,
    daily_pnl FLOAT NOT NULL,
    active_positions INTEGER NOT NULL,
    active_markets INTEGER NOT NULL,
    kill_switch_active BOOLEAN NOT NULL DEFAULT FALSE,
    snapshot_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
