"""Tests for cross-market arbitrage alpha."""

import pytest
from unittest.mock import MagicMock

from src.alpha.cross_market import CrossMarketAlpha


def _make_market(mid: str, question: str, best_bid=None, best_ask=None, last_price=None):
    m = MagicMock()
    m.id = mid
    m.question = question
    m.best_bid = best_bid
    m.best_ask = best_ask
    m.last_price = last_price
    return m


class TestGroupExtraction:
    def test_hungary_pm(self):
        key = CrossMarketAlpha._extract_group_key(
            "will the next prime minister of hungary be péter magyar?"
        )
        assert key == "hungary_pm"

    def test_nba_mvp(self):
        key = CrossMarketAlpha._extract_group_key(
            "will lebron james win the 2025–2026 nba mvp?"
        )
        assert key == "nba_mvp_2025"

    def test_unrelated(self):
        key = CrossMarketAlpha._extract_group_key(
            "will bitcoin reach $100k by december?"
        )
        assert key is None


class TestGroupUpdate:
    def test_builds_groups(self):
        alpha = CrossMarketAlpha()
        markets = [
            _make_market("0x1", "Will the next Prime Minister of Hungary be Péter Magyar?", best_bid=0.60),
            _make_market("0x2", "Will the next Prime Minister of Hungary be Viktor Orbán?", best_bid=0.32),
            _make_market("0x3", "Will the next Prime Minister of Hungary be László Toroczkai?", best_ask=0.001),
        ]
        alpha.update_groups(markets)
        assert "hungary_pm" in alpha._group_cache
        group = alpha._group_cache["hungary_pm"]
        assert group["count"] == 3
        # Sum should be ~0.921
        assert 0.9 < group["total_price"] < 0.95
        # Gap should be positive (underpriced)
        assert group["gap"] > 0.05

    def test_ignores_incomplete_groups(self):
        alpha = CrossMarketAlpha()
        # Only one market, no leader
        markets = [
            _make_market("0x1", "Will the next Prime Minister of Hungary be László Toroczkai?", best_ask=0.001),
        ]
        alpha.update_groups(markets)
        assert "hungary_pm" not in alpha._group_cache


class TestArbitrageSignal:
    @pytest.mark.asyncio
    async def test_underpriced_group_gives_positive_edge(self):
        alpha = CrossMarketAlpha()
        markets = [
            _make_market("0x1", "Will the next Prime Minister of Hungary be Péter Magyar?", best_bid=0.60),
            _make_market("0x2", "Will the next Prime Minister of Hungary be Viktor Orbán?", best_bid=0.32),
            _make_market("0x3", "Will the next Prime Minister of Hungary be László Toroczkai?", best_ask=0.001, last_price=0.001),
        ]
        alpha.update_groups(markets)

        # Compute for the leader
        result = await alpha.compute(markets[0], {})
        assert result is not None
        assert result.edge > 0  # Underpriced group -> positive edge
        assert "hungary_pm" in result.notes

    @pytest.mark.asyncio
    async def test_no_signal_for_unrelated_market(self):
        alpha = CrossMarketAlpha()
        markets = [
            _make_market("0x1", "Will the next Prime Minister of Hungary be Péter Magyar?", best_bid=0.60),
            _make_market("0x2", "Will the next Prime Minister of Hungary be Viktor Orbán?", best_bid=0.32),
            _make_market("0x3", "Will the next Prime Minister of Hungary be László Toroczkai?", best_ask=0.001),
        ]
        alpha.update_groups(markets)

        unrelated = _make_market("0x99", "Will bitcoin reach $100k?", best_bid=0.50)
        result = await alpha.compute(unrelated, {})
        assert result is None
