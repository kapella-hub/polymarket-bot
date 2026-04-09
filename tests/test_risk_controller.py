"""Tests for RiskController bug fixes."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.risk.controller import RiskController, RiskCheck
from src.ensemble.strategies import TradeDecision
from src.db.models import StrategyMode


def _make_decision(market_id="m1", size=100.0, edge=0.10):
    return TradeDecision(
        market_id=market_id,
        strategy=StrategyMode.INFORMATION,
        side="buy",
        token_id="tok_yes",
        edge=edge,
        confidence=0.8,
        suggested_size=size,
        notes="test",
    )


def _make_position(market_id="m1", size=10.0, avg_price=0.50):
    pos = MagicMock()
    pos.market_id = market_id
    pos.size = size
    pos.avg_entry_price = avg_price
    return pos


class TestDailyPnlTracking:
    def test_record_fill_updates_daily_pnl(self):
        rc = RiskController()
        assert rc._daily_pnl == 0.0
        rc.record_fill(-50.0)
        assert rc._daily_pnl == -50.0
        rc.record_fill(-160.0)
        assert rc._daily_pnl == -210.0

    @pytest.mark.asyncio
    async def test_daily_loss_limit_blocks_after_losses(self):
        rc = RiskController()
        rc.record_fill(-250.0)  # Exceeds $200 daily limit

        with patch("src.risk.controller.async_session") as mock_sess:
            mock_ctx = AsyncMock()
            mock_repo = MagicMock()
            mock_repo.get_all = AsyncMock(return_value=[])
            mock_sess.return_value.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)
            with patch("src.risk.controller.PositionRepository", return_value=mock_repo):
                result = await rc.check(_make_decision())

        assert not result.allowed
        assert "daily_loss_limit" in result.reason


class TestPortfolioDrawdown:
    def test_update_portfolio_value_tracks_peak(self):
        rc = RiskController()
        rc.update_portfolio_value(10000.0)
        assert rc._peak_value == 10000.0
        rc.update_portfolio_value(8000.0)
        assert rc._peak_value == 10000.0  # Peak stays
        rc.update_portfolio_value(12000.0)
        assert rc._peak_value == 12000.0


class TestConcurrencyLock:
    def test_lock_attribute_exists(self):
        rc = RiskController()
        assert hasattr(rc, "_lock")
        assert isinstance(rc._lock, asyncio.Lock)
