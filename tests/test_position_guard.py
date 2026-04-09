"""Tests for position repository sell guard."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.db.repositories import PositionRepository


class TestSellGuard:
    @pytest.mark.asyncio
    async def test_sell_clamped_to_position_size(self):
        """Selling more than held should clamp to current size, not go negative."""
        mock_session = AsyncMock()
        repo = PositionRepository(mock_session)

        existing = MagicMock()
        existing.size = 5.0
        existing.avg_entry_price = 0.50
        existing.realized_pnl = 0.0
        existing.cost_basis = 2.50

        repo.get_by_token = AsyncMock(return_value=existing)

        await repo.upsert_from_fill(
            market_id="m1",
            clob_token_id="tok1",
            outcome="Yes",
            side="sell",
            price=0.60,
            size=10.0,  # Selling 10 but only hold 5
        )

        assert existing.size >= 0  # Must not go negative
        assert existing.size == 0.0  # Clamped to sell only 5
        assert existing.realized_pnl == pytest.approx(0.50)  # (0.60 - 0.50) * 5

    @pytest.mark.asyncio
    async def test_normal_sell_works(self):
        """Normal partial sell should work correctly."""
        mock_session = AsyncMock()
        repo = PositionRepository(mock_session)

        existing = MagicMock()
        existing.size = 10.0
        existing.avg_entry_price = 0.50
        existing.realized_pnl = 0.0
        existing.cost_basis = 5.0

        repo.get_by_token = AsyncMock(return_value=existing)

        await repo.upsert_from_fill(
            market_id="m1",
            clob_token_id="tok1",
            outcome="Yes",
            side="sell",
            price=0.70,
            size=3.0,
        )

        assert existing.size == 7.0
        assert existing.realized_pnl == pytest.approx(0.60)  # (0.70 - 0.50) * 3
