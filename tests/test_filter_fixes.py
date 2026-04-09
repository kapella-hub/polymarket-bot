"""Tests for market filter bug fixes."""

from unittest.mock import MagicMock
from src.markets.filters import MarketFilter
from datetime import datetime, timezone, timedelta


def _make_market(**kwargs):
    m = MagicMock()
    m.id = kwargs.get("id", "m1")
    m.active = kwargs.get("active", True)
    m.volume = kwargs.get("volume", 100_000)
    m.liquidity = kwargs.get("liquidity", 50_000)
    m.spread = kwargs.get("spread", 0.05)
    m.category = kwargs.get("category", "Politics")
    m.end_date = kwargs.get("end_date", datetime.now(timezone.utc) + timedelta(days=30))
    m.outcomes = kwargs.get("outcomes", ["Yes", "No"])
    return m


class TestCategoryBlacklist:
    def test_blacklisted_category_rejected(self):
        f = MarketFilter(category_blacklist=["Sports", "Pop Culture"])
        m = _make_market(category="Sports")
        reason = f._check(m)
        assert reason is not None
        assert "blacklisted" in reason

    def test_none_category_rejected_when_whitelist_set(self):
        """Markets with no category should be rejected when whitelist is active."""
        f = MarketFilter(category_whitelist=["Politics", "Crypto"])
        m = _make_market(category=None)
        reason = f._check(m)
        assert reason is not None
        assert "not_whitelisted" in reason

    def test_none_category_passes_when_only_blacklist_set(self):
        """Markets with no category should pass when only blacklist is set."""
        f = MarketFilter(category_blacklist=["Sports"])
        m = _make_market(category=None)
        reason = f._check(m)
        assert reason is None

    def test_valid_category_passes_whitelist(self):
        f = MarketFilter(category_whitelist=["Politics", "Crypto"])
        m = _make_market(category="Politics")
        reason = f._check(m)
        assert reason is None
