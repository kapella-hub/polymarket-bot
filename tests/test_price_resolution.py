"""Tests for market price resolution and 0.50 fallback elimination."""

import pytest

from src.markets.models import MarketInfo, Outcome


def _make_market(
    best_bid=None,
    best_ask=None,
    yes_price=None,
) -> MarketInfo:
    """Build a MarketInfo with the given price fields."""
    outcomes = [
        Outcome(name="Yes", clob_token_id="tok_yes", price=yes_price),
        Outcome(name="No", clob_token_id="tok_no", price=None),
    ]
    return MarketInfo(
        id="0xtest",
        question="Test market?",
        best_bid=best_bid,
        best_ask=best_ask,
        outcomes=outcomes,
    )


class TestResolvePrice:
    def test_prefers_best_bid(self):
        m = _make_market(best_bid=0.35, best_ask=0.37, yes_price=0.36)
        assert m.resolve_price() == 0.35

    def test_falls_back_to_best_ask(self):
        m = _make_market(best_bid=None, best_ask=0.001, yes_price=0.0005)
        assert m.resolve_price() == 0.001

    def test_falls_back_to_yes_price(self):
        m = _make_market(best_bid=None, best_ask=None, yes_price=0.42)
        assert m.resolve_price() == 0.42

    def test_returns_none_when_all_missing(self):
        m = _make_market(best_bid=None, best_ask=None, yes_price=None)
        assert m.resolve_price() is None

    def test_never_returns_050(self):
        """The old code returned 0.50 as a fallback. This must never happen."""
        m = _make_market(best_bid=None, best_ask=None, yes_price=None)
        price = m.resolve_price()
        assert price is None or price != 0.5

    def test_zero_bid_falls_through(self):
        """best_bid of 0.0 is falsy — should fall through to best_ask."""
        m = _make_market(best_bid=0.0, best_ask=0.05, yes_price=0.03)
        # 0.0 is falsy in Python, so resolve_price should skip it
        assert m.resolve_price() == 0.05


class TestDbToInfo:
    """Test that _db_to_info properly passes price to YES outcome."""

    def test_with_best_bid(self):
        from unittest.mock import MagicMock

        from src.llm.batch import BatchScheduler

        db_market = MagicMock()
        db_market.id = "0xabc"
        db_market.question = "Test?"
        db_market.category = "Sports"
        db_market.end_date = None
        db_market.volume = 1000.0
        db_market.liquidity = 500.0
        db_market.best_bid = 0.60
        db_market.best_ask = 0.62
        db_market.last_price = 0.61
        db_market.outcome_yes = "Yes"
        db_market.outcome_no = "No"
        db_market.clob_token_id_yes = "tok_yes"
        db_market.clob_token_id_no = "tok_no"
        db_market.description = "A test"
        db_market.resolution_source = None
        db_market.tags = None

        info = BatchScheduler._db_to_info(db_market)
        assert info.yes_price == 0.60
        assert info.resolve_price() == 0.60

    def test_without_best_bid_uses_last_price(self):
        from unittest.mock import MagicMock

        from src.llm.batch import BatchScheduler

        db_market = MagicMock()
        db_market.id = "0xabc"
        db_market.question = "Test?"
        db_market.category = "Sports"
        db_market.end_date = None
        db_market.volume = 1000.0
        db_market.liquidity = 500.0
        db_market.best_bid = None
        db_market.best_ask = 0.001
        db_market.last_price = 0.0005
        db_market.outcome_yes = "Yes"
        db_market.outcome_no = "No"
        db_market.clob_token_id_yes = "tok_yes"
        db_market.clob_token_id_no = "tok_no"
        db_market.description = "A test"
        db_market.resolution_source = None
        db_market.tags = None

        info = BatchScheduler._db_to_info(db_market)
        # YES outcome should use last_price since best_bid is None
        assert info.yes_price == 0.0005
        # resolve_price should return best_ask (higher priority than yes_price
        # since best_bid is None, next is best_ask)
        assert info.resolve_price() == 0.001

    def test_all_prices_none(self):
        from unittest.mock import MagicMock

        from src.llm.batch import BatchScheduler

        db_market = MagicMock()
        db_market.id = "0xabc"
        db_market.question = "Test?"
        db_market.category = None
        db_market.end_date = None
        db_market.volume = 0.0
        db_market.liquidity = 0.0
        db_market.best_bid = None
        db_market.best_ask = None
        db_market.last_price = None
        db_market.outcome_yes = "Yes"
        db_market.outcome_no = "No"
        db_market.clob_token_id_yes = "tok_yes"
        db_market.clob_token_id_no = "tok_no"
        db_market.description = None
        db_market.resolution_source = None
        db_market.tags = None

        info = BatchScheduler._db_to_info(db_market)
        assert info.resolve_price() is None


class TestEnsembleNoPriceBailout:
    """Verify ensemble returns None when market has no price data."""

    @pytest.mark.asyncio
    async def test_no_price_returns_none(self):
        from unittest.mock import AsyncMock, MagicMock

        from src.alpha.base import AlphaOutput, AlphaSource
        from src.ensemble.engine import EnsembleEngine

        class FakeAlpha(AlphaSource):
            name = "fake"

            async def compute(self, market, context):
                return AlphaOutput(edge=0.15, confidence=0.8, notes="test")

        market = MagicMock()
        market.best_bid = None
        market.last_price = None
        market.best_ask = None
        market.id = "0xtest"
        market.volume = 100.0
        market.category = None
        market.end_date = None
        market.clob_token_id_yes = "tok_yes"
        market.clob_token_id_no = "tok_no"

        engine = EnsembleEngine(alphas=[FakeAlpha()])
        result = await engine.evaluate(market, {})
        # Should return None because no market price available
        assert result is None


class TestExecutorRejectsNoPriceLive:
    """Verify live/paper executors reject trades with no price."""

    @pytest.mark.asyncio
    async def test_paper_rejects_no_fill_price(self):
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.db.models import StrategyMode
        from src.ensemble.strategies import TradeDecision
        from src.exchange.base import OrderBook
        from src.execution.executor import ExecutionEngine
        from src.risk.controller import RiskCheck

        mock_exchange = MagicMock()
        mock_exchange.get_order_book = AsyncMock(
            return_value=OrderBook(bids=[], asks=[])  # empty book
        )

        mock_risk = MagicMock()
        mock_risk.check = AsyncMock(
            return_value=RiskCheck(allowed=True, reason="ok")
        )

        mock_intents = MagicMock()
        mock_intents.create_from_decision = AsyncMock(return_value=1)
        mock_intents.invalidate = AsyncMock()

        engine = ExecutionEngine(
            exchange=mock_exchange,
            risk=mock_risk,
            intents=mock_intents,
        )
        engine._mode = "paper"

        decision = TradeDecision(
            market_id="0xtest",
            strategy=StrategyMode.INFORMATION,
            side="buy",
            token_id="tok_yes",
            edge=0.1,
            confidence=0.5,
            suggested_size=100.0,
        )

        result = await engine.process(decision)
        assert result is False
        mock_intents.invalidate.assert_called_once()
