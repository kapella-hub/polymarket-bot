"""Tests for strategy ranking and adaptive threshold helpers."""

from run_certainty_sniper import (
    CERTAINTY_END,
    CERTAINTY_START,
    certainty_confirm_threshold,
    certainty_move_threshold,
    certainty_signal_score,
    orderbook_imbalance,
)
from run_sniper import continuation_signal_score


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
