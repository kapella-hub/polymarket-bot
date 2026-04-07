"""SQLAlchemy async models for the Polymarket trading bot."""

import enum
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


# --- Enums ---


class MarketStatus(str, enum.Enum):
    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"


class IntentState(str, enum.Enum):
    CREATED = "created"
    ARMED = "armed"
    EXECUTED = "executed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    INVALIDATED = "invalidated"


class InvalidationReason(str, enum.Enum):
    MARKET_RESOLVED = "market_resolved"
    SIGNAL_STALE = "signal_stale"
    SPREAD_WIDENED = "spread_widened"
    POSITION_LIMIT = "position_limit"
    RISK_LIMIT = "risk_limit"
    KILL_SWITCH = "kill_switch"
    MANUAL = "manual"


class OrderSide(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"


class StrategyMode(str, enum.Enum):
    INFORMATION = "information"
    MARKET_MAKING = "market_making"
    ARBITRAGE = "arbitrage"


# --- Models ---


class Market(Base):
    """A Polymarket prediction market."""

    __tablename__ = "markets"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)  # Polymarket condition_id
    question: Mapped[str] = mapped_column(Text)
    category: Mapped[Optional[str]] = mapped_column(String(64))
    end_date: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    status: Mapped[MarketStatus] = mapped_column(
        Enum(MarketStatus), default=MarketStatus.ACTIVE
    )

    # Token IDs for YES/NO outcomes
    clob_token_id_yes: Mapped[str] = mapped_column(String(128))
    clob_token_id_no: Mapped[str] = mapped_column(String(128))
    outcome_yes: Mapped[str] = mapped_column(String(128), default="Yes")
    outcome_no: Mapped[str] = mapped_column(String(128), default="No")

    # Market data (updated on each discovery scan)
    volume: Mapped[float] = mapped_column(Float, default=0.0)
    liquidity: Mapped[float] = mapped_column(Float, default=0.0)
    best_bid: Mapped[Optional[float]] = mapped_column(Float)
    best_ask: Mapped[Optional[float]] = mapped_column(Float)
    last_price: Mapped[Optional[float]] = mapped_column(Float)

    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text)
    resolution_source: Mapped[Optional[str]] = mapped_column(Text)
    tags: Mapped[Optional[dict]] = mapped_column(JSONB)

    discovered_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_markets_status", "status"),
        Index("ix_markets_category", "category"),
        Index("ix_markets_volume", "volume"),
    )


class MarketSignal(Base):
    """LLM probability estimate for a market (the signal store)."""

    __tablename__ = "market_signals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)

    # LLM output
    probability: Mapped[float] = mapped_column(Float)  # 0.0-1.0
    confidence: Mapped[float] = mapped_column(Float)  # 0.0-1.0
    edge_over_market: Mapped[float] = mapped_column(Float)  # prob - market_price
    reasoning: Mapped[Optional[str]] = mapped_column(Text)
    key_factors: Mapped[Optional[dict]] = mapped_column(JSONB)

    # Context at evaluation time
    market_price_at_eval: Mapped[float] = mapped_column(Float)
    model_version: Mapped[str] = mapped_column(String(64), default="claude-cli")

    evaluated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_signals_market_evaluated", "market_id", "evaluated_at"),
        Index("ix_signals_expires", "expires_at"),
    )


class OrderIntent(Base):
    """Order intent with full lifecycle tracking."""

    __tablename__ = "order_intents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    clob_token_id: Mapped[str] = mapped_column(String(128))

    # Order details
    side: Mapped[OrderSide] = mapped_column(Enum(OrderSide))
    price: Mapped[float] = mapped_column(Float)
    size: Mapped[float] = mapped_column(Float)  # In USDC
    strategy: Mapped[StrategyMode] = mapped_column(Enum(StrategyMode))

    # Signal context
    signal_id: Mapped[Optional[int]] = mapped_column(Integer)
    edge_at_creation: Mapped[float] = mapped_column(Float)
    confidence_at_creation: Mapped[float] = mapped_column(Float)

    # State machine
    state: Mapped[IntentState] = mapped_column(
        Enum(IntentState), default=IntentState.CREATED
    )
    invalidation_reason: Mapped[Optional[InvalidationReason]] = mapped_column(
        Enum(InvalidationReason)
    )

    # Execution details (populated on fill)
    exchange_order_id: Mapped[Optional[str]] = mapped_column(String(256))
    filled_price: Mapped[Optional[float]] = mapped_column(Float)
    filled_size: Mapped[Optional[float]] = mapped_column(Float)
    fill_timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_intents_state", "state"),
        Index("ix_intents_market_state", "market_id", "state"),
    )


class Position(Base):
    """Current position in a market token."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    clob_token_id: Mapped[str] = mapped_column(String(128))
    outcome: Mapped[str] = mapped_column(String(128))  # "Yes" or "No"

    # Position details
    size: Mapped[float] = mapped_column(Float, default=0.0)  # Token quantity
    avg_entry_price: Mapped[float] = mapped_column(Float, default=0.0)
    cost_basis: Mapped[float] = mapped_column(Float, default=0.0)  # Total USDC spent
    realized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)

    opened_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_positions_token", "clob_token_id", unique=True),
    )


class Trade(Base):
    """Record of an executed trade (fill)."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    clob_token_id: Mapped[str] = mapped_column(String(128))
    intent_id: Mapped[Optional[int]] = mapped_column(Integer)

    side: Mapped[OrderSide] = mapped_column(Enum(OrderSide))
    price: Mapped[float] = mapped_column(Float)
    size: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    strategy: Mapped[StrategyMode] = mapped_column(Enum(StrategyMode))

    exchange_order_id: Mapped[str] = mapped_column(String(256))
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_trades_executed", "executed_at"),
    )


class RiskSnapshot(Base):
    """Periodic snapshot of portfolio risk state."""

    __tablename__ = "risk_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    portfolio_value: Mapped[float] = mapped_column(Float)
    total_exposure: Mapped[float] = mapped_column(Float)
    drawdown_pct: Mapped[float] = mapped_column(Float)
    daily_pnl: Mapped[float] = mapped_column(Float)
    active_positions: Mapped[int] = mapped_column(Integer)
    active_markets: Mapped[int] = mapped_column(Integer)
    kill_switch_active: Mapped[bool] = mapped_column(Boolean, default=False)

    snapshot_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
