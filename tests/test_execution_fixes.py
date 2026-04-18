"""Tests for execution, signal freshness, and portfolio valuation fixes."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.db.models import StrategyMode
from src.ensemble.strategies import TradeDecision
from src.execution.executor import ExecutionEngine
from src.exchange.base import OrderBook, OrderResult, TradeRecord
from src.main import _compute_portfolio_value, _signal_is_tradeable
from src.risk.controller import RiskCheck, RiskController


def _make_decision(token_id: str = "tok_yes") -> TradeDecision:
    return TradeDecision(
        market_id="m1",
        strategy=StrategyMode.INFORMATION,
        side="buy",
        token_id=token_id,
        edge=0.10,
        confidence=0.80,
        suggested_size=100.0,
        notes="test",
    )


class TestSignalFreshness:
    def test_stale_signal_rejected(self):
        signal = MagicMock()
        signal.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert _signal_is_tradeable(signal) is False

    def test_fresh_signal_accepted(self):
        signal = MagicMock()
        signal.expires_at = datetime.now(timezone.utc) + timedelta(minutes=5)
        assert _signal_is_tradeable(signal) is True


class TestPortfolioValue:
    def test_marks_yes_and_no_positions_to_market(self):
        yes_market = MagicMock()
        yes_market.id = "m_yes"
        yes_market.best_bid = 0.62
        yes_market.best_ask = 0.64
        yes_market.last_price = 0.63

        no_market = MagicMock()
        no_market.id = "m_no"
        no_market.best_bid = 0.28
        no_market.best_ask = 0.30
        no_market.last_price = 0.29

        yes_position = MagicMock()
        yes_position.market_id = "m_yes"
        yes_position.outcome = "Yes"
        yes_position.size = 10.0
        yes_position.cost_basis = 5.50
        yes_position.avg_entry_price = 0.55

        no_position = MagicMock()
        no_position.market_id = "m_no"
        no_position.outcome = "No"
        no_position.size = 8.0
        no_position.cost_basis = 2.40
        no_position.avg_entry_price = 0.30

        value = _compute_portfolio_value(
            [yes_position, no_position],
            [yes_market, no_market],
            bankroll_usd=100.0,
        )

        expected_cash = 100.0 - 5.50 - 2.40
        expected_marked = 10.0 * 0.62 + 8.0 * 0.70
        assert value == pytest.approx(expected_cash + expected_marked)


class TestRiskControllerCurrentValue:
    @pytest.mark.asyncio
    async def test_drawdown_uses_current_portfolio_value(self):
        rc = RiskController()
        rc.update_portfolio_value(1000.0)
        rc.update_portfolio_value(800.0)

        with pytest.MonkeyPatch.context() as mp:
            mock_repo = MagicMock()
            mock_repo.get_all = AsyncMock(return_value=[])

            mock_session_factory = MagicMock()
            mock_session_factory.__aenter__ = AsyncMock(return_value=AsyncMock())
            mock_session_factory.__aexit__ = AsyncMock(return_value=False)

            mp.setattr("src.risk.controller.async_session", MagicMock(return_value=mock_session_factory))
            mp.setattr("src.risk.controller.PositionRepository", MagicMock(return_value=mock_repo))
            mp.setattr("src.risk.controller.settings.max_portfolio_drawdown_pct", 0.15)

            result = await rc.check(_make_decision())

        assert result.allowed is False
        assert "drawdown_limit" in result.reason


class TestExecutionStateAndFillConfirmation:
    @pytest.mark.asyncio
    async def test_shadow_arms_before_execute(self):
        exchange = MagicMock()
        risk = MagicMock()
        risk.check = AsyncMock(return_value=RiskCheck(allowed=True))
        intents = MagicMock()
        intents.create_from_decision = AsyncMock(return_value=1)
        intents.arm = AsyncMock(return_value=True)
        intents.execute = AsyncMock(return_value=True)

        engine = ExecutionEngine(exchange=exchange, risk=risk, intents=intents)
        engine._mode = "shadow"

        result = await engine.process(_make_decision())

        assert result is True
        intents.arm.assert_awaited_once_with(1, 0.0)
        intents.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_live_cancels_unconfirmed_order(self):
        exchange = MagicMock()
        exchange.get_order_book = AsyncMock(
            return_value=OrderBook(
                bids=[],
                asks=[MagicMock(price=0.55, size=50.0)],
            )
        )
        exchange.place_limit_order = AsyncMock(
            return_value=OrderResult(order_id="oid-1", success=True)
        )
        exchange.get_trades = AsyncMock(return_value=[])
        exchange.cancel_order = AsyncMock(return_value=True)

        risk = MagicMock()
        risk.check = AsyncMock(return_value=RiskCheck(allowed=True))
        intents = MagicMock()
        intents.create_from_decision = AsyncMock(return_value=1)
        intents.arm = AsyncMock(return_value=True)
        intents.execute = AsyncMock(return_value=True)
        intents.invalidate = AsyncMock(return_value=True)

        engine = ExecutionEngine(exchange=exchange, risk=risk, intents=intents)
        engine._mode = "live"

        result = await engine.process(_make_decision())

        assert result is False
        intents.execute.assert_not_called()
        intents.invalidate.assert_awaited_once()
        exchange.cancel_order.assert_awaited_once_with("oid-1")

    @pytest.mark.asyncio
    async def test_live_uses_confirmed_fill_size_and_price(self):
        exchange = MagicMock()
        exchange.get_order_book = AsyncMock(
            return_value=OrderBook(
                bids=[],
                asks=[MagicMock(price=0.55, size=50.0)],
            )
        )
        exchange.place_limit_order = AsyncMock(
            return_value=OrderResult(order_id="oid-1", success=True)
        )
        exchange.get_trades = AsyncMock(
            return_value=[
                TradeRecord(
                    trade_id="t1",
                    order_id="oid-1",
                    token_id="tok_yes",
                    side="buy",
                    price=0.54,
                    size=50.0,
                    fee=1.25,
                    timestamp=0.0,
                )
            ]
        )

        risk = MagicMock()
        risk.check = AsyncMock(return_value=RiskCheck(allowed=True))
        risk.record_fill = MagicMock()
        intents = MagicMock()
        intents.create_from_decision = AsyncMock(return_value=1)
        intents.arm = AsyncMock(return_value=True)
        intents.execute = AsyncMock(return_value=True)

        mock_session = AsyncMock()
        mock_session_factory = MagicMock()
        mock_session_factory.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.__aexit__ = AsyncMock(return_value=False)

        market_repo = MagicMock()
        market_repo.get_by_id = AsyncMock(
            return_value=MagicMock(
                clob_token_id_yes="tok_yes",
                clob_token_id_no="tok_no",
                outcome_yes="Yes",
                outcome_no="No",
            )
        )
        trade_repo = MagicMock()
        trade_repo.record = AsyncMock()
        pos_repo = MagicMock()
        pos_repo.upsert_from_fill = AsyncMock(return_value=0.0)

        engine = ExecutionEngine(exchange=exchange, risk=risk, intents=intents)
        engine._mode = "live"

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr("src.execution.executor.async_session", MagicMock(return_value=mock_session_factory))
            mp.setattr("src.execution.executor.MarketRepository", MagicMock(return_value=market_repo))
            mp.setattr("src.execution.executor.TradeRepository", MagicMock(return_value=trade_repo))
            mp.setattr("src.execution.executor.PositionRepository", MagicMock(return_value=pos_repo))

            result = await engine.process(_make_decision())

        assert result is True
        intents.execute.assert_awaited_once_with(
            1,
            exchange_order_id="oid-1",
            filled_price=0.54,
            filled_size=50.0,
        )
        trade_repo.record.assert_awaited_once()
        pos_repo.upsert_from_fill.assert_awaited_once_with(
            market_id="m1",
            clob_token_id="tok_yes",
            outcome="Yes",
            side="buy",
            price=0.54,
            size=50.0,
        )
        risk.record_fill.assert_called_once_with(-1.25)
