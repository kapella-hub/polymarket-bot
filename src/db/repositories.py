"""Async data access layer for all database operations."""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models import (
    IntentState,
    InvalidationReason,
    Market,
    MarketSignal,
    MarketStatus,
    OrderIntent,
    Position,
    RiskSnapshot,
    Trade,
)


class MarketRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def upsert(self, market: dict) -> None:
        """Insert or update a market from Gamma API data."""
        stmt = pg_insert(Market).values(**market)
        stmt = stmt.on_conflict_do_update(
            index_elements=["id"],
            set_={
                "volume": stmt.excluded.volume,
                "liquidity": stmt.excluded.liquidity,
                "best_bid": stmt.excluded.best_bid,
                "best_ask": stmt.excluded.best_ask,
                "last_price": stmt.excluded.last_price,
                "status": stmt.excluded.status,
                "updated_at": datetime.now(timezone.utc),
            },
        )
        await self.session.execute(stmt)

    async def get_active(self) -> list[Market]:
        result = await self.session.execute(
            select(Market)
            .where(Market.status == MarketStatus.ACTIVE)
            .order_by(Market.volume.desc())
        )
        return list(result.scalars().all())

    async def get_by_id(self, market_id: str) -> Optional[Market]:
        result = await self.session.execute(
            select(Market).where(Market.id == market_id)
        )
        return result.scalar_one_or_none()


class SignalRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def write_signal(self, signal: dict) -> int:
        """Write a new LLM signal to the store."""
        obj = MarketSignal(**signal)
        self.session.add(obj)
        await self.session.flush()
        return obj.id

    async def get_latest(self, market_id: str) -> Optional[MarketSignal]:
        """Get the most recent signal for a market."""
        result = await self.session.execute(
            select(MarketSignal)
            .where(MarketSignal.market_id == market_id)
            .order_by(MarketSignal.evaluated_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_fresh_signals(self) -> list[MarketSignal]:
        """Get all non-expired signals."""
        now = datetime.now(timezone.utc)
        result = await self.session.execute(
            select(MarketSignal)
            .where(MarketSignal.expires_at > now)
            .order_by(MarketSignal.evaluated_at.desc())
        )
        return list(result.scalars().all())

    async def get_markets_needing_eval(
        self, active_market_ids: list[str], ttl_seconds: int
    ) -> list[str]:
        """Return market IDs that need LLM evaluation (no signal or stale)."""
        now = datetime.now(timezone.utc)
        result = await self.session.execute(
            select(MarketSignal.market_id, MarketSignal.evaluated_at)
            .where(MarketSignal.market_id.in_(active_market_ids))
            .distinct(MarketSignal.market_id)
            .order_by(MarketSignal.market_id, MarketSignal.evaluated_at.desc())
        )
        signal_times = {row.market_id: row.evaluated_at for row in result.all()}

        stale_cutoff = now.timestamp() - ttl_seconds
        needing = []
        for mid in active_market_ids:
            if mid not in signal_times:
                needing.append(mid)  # Never evaluated
            elif signal_times[mid].timestamp() < stale_cutoff:
                needing.append(mid)  # Stale signal
        return needing


class IntentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, intent: dict) -> int:
        obj = OrderIntent(**intent)
        self.session.add(obj)
        await self.session.flush()
        return obj.id

    async def update_state(
        self,
        intent_id: int,
        new_state: IntentState,
        reason: Optional[InvalidationReason] = None,
        **extra_fields,
    ) -> None:
        values = {"state": new_state, "updated_at": datetime.now(timezone.utc)}
        if reason:
            values["invalidation_reason"] = reason
        values.update(extra_fields)
        await self.session.execute(
            update(OrderIntent).where(OrderIntent.id == intent_id).values(**values)
        )

    async def get_state(self, intent_id: int) -> Optional[IntentState]:
        result = await self.session.execute(
            select(OrderIntent.state).where(OrderIntent.id == intent_id)
        )
        row = result.scalar_one_or_none()
        return row

    async def get_active(self) -> list[OrderIntent]:
        result = await self.session.execute(
            select(OrderIntent).where(
                OrderIntent.state.in_([IntentState.CREATED, IntentState.ARMED])
            )
        )
        return list(result.scalars().all())


class PositionRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all(self) -> list[Position]:
        result = await self.session.execute(
            select(Position).where(Position.size > 0)
        )
        return list(result.scalars().all())

    async def get_by_token(self, clob_token_id: str) -> Optional[Position]:
        result = await self.session.execute(
            select(Position).where(Position.clob_token_id == clob_token_id)
        )
        return result.scalar_one_or_none()

    async def upsert_from_fill(
        self, market_id: str, clob_token_id: str, outcome: str,
        side: str, price: float, size: float,
    ) -> None:
        """Update position from a trade fill."""
        existing = await self.get_by_token(clob_token_id)
        if existing is None:
            pos = Position(
                market_id=market_id,
                clob_token_id=clob_token_id,
                outcome=outcome,
                size=size if side == "buy" else -size,
                avg_entry_price=price,
                cost_basis=price * size,
            )
            self.session.add(pos)
        else:
            if side == "buy":
                new_size = existing.size + size
                existing.cost_basis += price * size
                existing.avg_entry_price = (
                    existing.cost_basis / new_size if new_size > 0 else 0
                )
                existing.size = new_size
            else:
                pnl = (price - existing.avg_entry_price) * size
                existing.realized_pnl += pnl
                existing.size -= size
                existing.cost_basis = existing.avg_entry_price * existing.size


class TradeRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def record(self, trade: dict) -> int:
        obj = Trade(**trade)
        self.session.add(obj)
        await self.session.flush()
        return obj.id


class RiskRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def save_snapshot(self, snapshot: dict) -> None:
        self.session.add(RiskSnapshot(**snapshot))

    async def get_latest(self) -> Optional[RiskSnapshot]:
        result = await self.session.execute(
            select(RiskSnapshot).order_by(RiskSnapshot.snapshot_at.desc()).limit(1)
        )
        return result.scalar_one_or_none()
