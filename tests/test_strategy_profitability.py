"""Tests for strategy ranking and adaptive threshold helpers."""

from run_certainty_sniper import (
    CERTAINTY_END,
    CERTAINTY_START,
    certainty_coin_is_tradeable,
    certainty_confirm_threshold,
    certainty_move_threshold,
    certainty_signal_score,
    orderbook_imbalance,
)
from run_sniper import continuation_signal_score
from run_sniper import continuation_asset_is_tradeable, continuation_move_gate, continuation_target_size
from run_maker_shadow import maker_quote_price


class TestCertaintyAdaptiveThresholds:
    def test_move_threshold_relaxes_later_in_window(self):
        early = certainty_move_threshold(CERTAINTY_START)
        late = certainty_move_threshold(CERTAINTY_END)
        assert late < early

    def test_confirm_threshold_relaxes_later_in_window(self):
        early = certainty_confirm_threshold(CERTAINTY_START)
        late = certainty_confirm_threshold(CERTAINTY_END)
        assert late < early


class TestCertaintyScoring:
    def test_certainty_focuses_on_liquid_coins(self):
        assert certainty_coin_is_tradeable("btc") is True
        assert certainty_coin_is_tradeable("eth") is True
        assert certainty_coin_is_tradeable("doge") is False
        assert certainty_coin_is_tradeable("xrp") is False

    def test_orderbook_imbalance_respects_direction(self):
        assert orderbook_imbalance(120.0, 80.0, "up") > 0
        assert orderbook_imbalance(120.0, 80.0, "down") < 0

    def test_score_prefers_stronger_cheaper_signal(self):
        weak = certainty_signal_score(
            move_abs=0.006,
            move_threshold=0.0055,
            confirm_count=2,
            imbalance=0.02,
            entry_price=0.90,
            ask_usd_vol=12.0,
            elapsed=CERTAINTY_START,
        )
        strong = certainty_signal_score(
            move_abs=0.010,
            move_threshold=0.0055,
            confirm_count=4,
            imbalance=0.25,
            entry_price=0.84,
            ask_usd_vol=35.0,
            elapsed=CERTAINTY_END,
        )
        assert strong > weak


class TestContinuationScoring:
    def test_score_penalizes_reversal_and_expensive_entry(self):
        good = continuation_signal_score(
            lead_move_abs=0.0030,
            confirm_move_abs=0.0025,
            reversal=0.0002,
            entry_price=0.71,
            ask_usd_vol=30.0,
        )
        bad = continuation_signal_score(
            lead_move_abs=0.0030,
            confirm_move_abs=0.0025,
            reversal=0.0011,
            entry_price=0.87,
            ask_usd_vol=10.0,
        )
        assert good > bad


class TestContinuationAdmission:
    def test_joint_move_gate_allows_eth_led_drift_when_btc_confirms(self):
        should_trade, reason = continuation_move_gate(
            btc_move_abs=0.0010,
            eth_move_abs=0.0027,
            elapsed=420,
        )
        assert should_trade is True
        assert reason is None

    def test_joint_move_gate_rejects_small_combined_move(self):
        should_trade, reason = continuation_move_gate(
            btc_move_abs=0.0004,
            eth_move_abs=0.0008,
            elapsed=420,
        )
        assert should_trade is False
        assert reason == "joint_move_too_small"

    def test_shadow_asset_requires_direction_and_min_confirmation(self):
        assert continuation_asset_is_tradeable(0.0012, "up", 420) is True
        assert continuation_asset_is_tradeable(-0.0012, "up", 420) is False
        assert continuation_asset_is_tradeable(0.0002, "up", 420) is False


class TestContinuationSizing:
    def test_target_size_has_continuation_floor(self):
        assert continuation_target_size(20.0) == 5.0

    def test_target_size_scales_up_with_bankroll(self):
        assert continuation_target_size(200.0) > 5.0


class TestMakerShadow:
    def test_maker_quote_requires_real_spread(self):
        assert maker_quote_price(0.60, 0.61, 0.88) is None

    def test_maker_quote_prices_inside_spread(self):
        assert maker_quote_price(0.60, 0.63, 0.88) == 0.61

    def test_maker_quote_respects_max_entry(self):
        assert maker_quote_price(0.88, 0.91, 0.88) is None
