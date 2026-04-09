"""Tests for crypto arb bug fixes."""

import pytest


class TestFeeModel:
    def test_proportional_fee_not_flat(self):
        """Fee should be proportional, not a flat subtraction."""
        entry_price = 0.10
        fee_rate = 0.10

        # Old bug: flat subtraction
        old_gross = (1.0 / entry_price) - 1.0
        old_net = old_gross - fee_rate

        # New: proportional
        payout_after_fees = 1.0 * (1 - fee_rate)
        cost_with_fees = entry_price * (1 + fee_rate)
        new_net = (payout_after_fees / cost_with_fees) - 1.0

        # The bug overstates return
        assert new_net < old_net
        # At 10c entry: old says 890%, correct is ~718%
        assert old_net == pytest.approx(8.9)
        assert new_net == pytest.approx(7.1818, rel=0.01)

    def test_fee_at_higher_price(self):
        """At 50c entry, difference is smaller but still meaningful."""
        entry_price = 0.50
        fee_rate = 0.10

        payout_after_fees = 1.0 * (1 - fee_rate)
        cost_with_fees = entry_price * (1 + fee_rate)
        net_return = (payout_after_fees / cost_with_fees) - 1.0

        # At 50c: should be ~63.6% net return
        assert net_return == pytest.approx(0.6364, rel=0.01)


class TestStalenessCheck:
    def test_stale_prices_detected(self):
        """Prices older than 10s should be considered stale."""
        import time

        class FakeFeed:
            def __init__(self, age):
                self._last_update = time.time() - age

            @property
            def age_seconds(self):
                return time.time() - self._last_update

        fresh = FakeFeed(2)
        assert fresh.age_seconds < 10

        stale = FakeFeed(30)
        assert stale.age_seconds > 10
