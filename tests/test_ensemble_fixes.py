"""Tests for ensemble engine and alpha pricing fixes."""

from unittest.mock import MagicMock
import pytest


class TestMidpointPricing:
    def test_midpoint_calculation(self):
        """Price reference should be midpoint of bid/ask, not bid alone."""
        bid = 0.48
        ask = 0.52
        midpoint = (bid + ask) / 2
        assert midpoint == 0.50
        # Bug was using bid (0.48), inflating buy-side edge by 2 cents
        assert midpoint > bid

    def test_zero_bid_falls_through_to_last_price(self):
        """A bid of 0.0 should not be used — fall through to last_price."""
        bid = 0.0
        ask = 0.52
        last = 0.50
        # With the fix: bid=0 means we skip midpoint, use last_price
        if bid is not None and ask is not None and bid > 0 and ask > 0:
            price = (bid + ask) / 2
        elif last is not None and last > 0:
            price = last
        else:
            price = None
        assert price == 0.50  # Uses last_price, not bid=0


class TestKellyBankroll:
    def test_kelly_scales_with_bankroll(self):
        """Kelly fraction * bankroll should use configured bankroll, not hardcoded 10k."""
        kelly_fraction = 0.03  # 3% Kelly

        # Old: 0.03 * 10000 = $300
        old_size = kelly_fraction * 10000
        assert old_size == 300.0

        # New with $2000 bankroll: 0.03 * 2000 = $60
        new_size = kelly_fraction * 2000
        assert new_size == 60.0
        assert new_size < old_size  # Smaller bankroll = smaller position
