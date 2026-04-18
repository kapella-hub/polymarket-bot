"""Risk controller with kill switch, position limits, and drawdown protection."""

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import structlog

from src.config import settings
from src.db.database import async_session
from src.db.repositories import PositionRepository, TradeRepository
from src.ensemble.strategies import TradeDecision

logger = structlog.get_logger()


@dataclass
class RiskCheck:
    """Result of a risk check."""

    allowed: bool
    reason: str = ""
    adjusted_size: Optional[float] = None  # If size was reduced


class RiskController:
    """Enforces risk limits before order execution.

    Checks (in order):
    1. Kill switch
    2. Per-market position cap
    3. Portfolio exposure cap
    4. Daily loss limit
    5. Portfolio drawdown
    """

    def __init__(self):
        self._peak_value: float = 0.0
        self._current_value: float = 0.0
        self._daily_pnl: float = 0.0
        self._day_start: datetime = datetime.now(timezone.utc)
        self._lock = asyncio.Lock()

    async def check(self, decision: TradeDecision) -> RiskCheck:
        """Run all risk checks against a trade decision."""
        async with self._lock:
            return await self._check_inner(decision)

    async def _check_inner(self, decision: TradeDecision) -> RiskCheck:
        """Run all risk checks (called under lock)."""

        # 1. Kill switch
        if self._kill_switch_active():
            return RiskCheck(allowed=False, reason="kill_switch_active")

        async with async_session() as session:
            pos_repo = PositionRepository(session)
            positions = await pos_repo.get_all()

        # 2. Per-market position cap
        market_exposure = sum(
            abs(p.size * p.avg_entry_price)
            for p in positions
            if p.market_id == decision.market_id
        )
        remaining_market_cap = settings.max_position_per_market_usd - market_exposure
        if remaining_market_cap <= 0:
            return RiskCheck(
                allowed=False,
                reason=f"market_position_cap ({market_exposure:.2f} >= {settings.max_position_per_market_usd:.2f})",
            )

        # 3. Portfolio exposure cap
        total_exposure = sum(abs(p.size * p.avg_entry_price) for p in positions)
        # Estimate portfolio value (exposure + remaining cash)
        if self._current_value > 0:
            portfolio_value = self._current_value
        elif self._peak_value > 0:
            portfolio_value = max(total_exposure, self._peak_value)
        else:
            portfolio_value = total_exposure + settings.bankroll_usd
        max_exposure = portfolio_value * settings.max_portfolio_exposure_pct
        remaining_portfolio_cap = max_exposure - total_exposure
        if remaining_portfolio_cap <= 0:
            return RiskCheck(
                allowed=False,
                reason=f"portfolio_exposure_cap ({total_exposure:.2f} >= {max_exposure:.2f})",
            )

        # 4. Daily loss limit
        self._reset_daily_if_needed()
        if self._daily_pnl < -settings.max_daily_loss_usd:
            return RiskCheck(
                allowed=False,
                reason=f"daily_loss_limit ({self._daily_pnl:.2f})",
            )

        # 5. Portfolio drawdown
        if self._peak_value > 0:
            drawdown = 1.0 - (portfolio_value / self._peak_value)
            if drawdown > settings.max_portfolio_drawdown_pct:
                return RiskCheck(
                    allowed=False,
                    reason=f"drawdown_limit ({drawdown:.1%} > {settings.max_portfolio_drawdown_pct:.1%})",
                )

        # Cap size to the smaller of remaining caps
        max_size = min(
            remaining_market_cap,
            remaining_portfolio_cap,
            decision.suggested_size,
        )

        if max_size < 1.0:  # Minimum $1 trade
            return RiskCheck(allowed=False, reason="size_too_small")

        return RiskCheck(
            allowed=True,
            adjusted_size=max_size if max_size < decision.suggested_size else None,
        )

    def record_fill(self, pnl: float) -> None:
        """Record a trade's P&L for daily tracking."""
        self._daily_pnl += pnl

    def update_portfolio_value(self, value: float) -> None:
        """Update peak portfolio value for drawdown tracking."""
        self._current_value = value
        if value > self._peak_value:
            self._peak_value = value

    def _kill_switch_active(self) -> bool:
        return os.path.exists(settings.kill_switch_file)

    def _reset_daily_if_needed(self) -> None:
        now = datetime.now(timezone.utc)
        if now.date() > self._day_start.date():
            self._daily_pnl = 0.0
            self._day_start = now
