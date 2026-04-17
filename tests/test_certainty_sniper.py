import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_certainty_sniper import (
    kelly_size, gate1_move, gate2_confirm, compute_all_moves,
    MIN_MOVE_PCT, CONFIRM_MOVE_PCT, MIN_CONFIRMING,
    BET_MIN, BET_MAX,
)


def test_kelly_size_typical():
    size = kelly_size(bankroll=151.25)
    assert BET_MIN <= size <= BET_MAX


def test_kelly_size_small_bankroll():
    size = kelly_size(bankroll=35.0)
    assert size == BET_MIN  # floor at $8


def test_kelly_size_large_bankroll():
    size = kelly_size(bankroll=500.0)
    assert size == BET_MAX  # cap at $40


def test_gate1_move_passes():
    passed, move = gate1_move(cur=101.0, start=100.0)
    assert passed is True
    assert abs(move - 0.01) < 1e-9


def test_gate1_move_fails_small():
    passed, move = gate1_move(cur=100.3, start=100.0)
    assert passed is False  # 0.3% < 0.7%


def test_gate1_move_missing_price():
    passed, move = gate1_move(cur=0, start=100.0)
    assert passed is False


def test_gate2_confirm_up_passes():
    moves = {"btc": 0.008, "eth": 0.006, "sol": -0.002, "xrp": 0.003, "doge": 0.001, "bnb": 0.007}
    assert gate2_confirm(moves, "up") is True  # btc, eth, bnb all >=0.5% up


def test_gate2_confirm_up_fails():
    moves = {"btc": 0.008, "eth": 0.002, "sol": -0.002, "xrp": 0.001, "doge": 0.001, "bnb": 0.003}
    assert gate2_confirm(moves, "up") is False  # only btc >=0.5% up


def test_gate2_confirm_down_passes():
    moves = {"btc": -0.009, "eth": -0.006, "sol": 0.001, "xrp": -0.005, "doge": 0.001, "bnb": 0.002}
    assert gate2_confirm(moves, "down") is True


class FakeWSFeed:
    def __init__(self, prices):
        self._prices = prices
    def get(self, asset):
        return self._prices.get(asset)


def test_compute_all_moves():
    feed = FakeWSFeed({"BTC": 76000.0, "ETH": 2400.0, "SOL": 130.0,
                       "XRP": 0.60, "DOGE": 0.15, "BNB": 600.0})
    period_prices = {"BTC": 75000.0, "ETH": 2400.0, "SOL": 130.0,
                     "XRP": 0.60, "DOGE": 0.15, "BNB": 600.0}
    moves = compute_all_moves(feed, period_prices)
    assert abs(moves["btc"] - 0.01333) < 0.001  # BTC up ~1.33%
    assert moves["eth"] == 0.0  # ETH flat


def test_compute_all_moves_missing_price():
    feed = FakeWSFeed({"BTC": 76000.0})  # only BTC
    period_prices = {"BTC": 75000.0, "ETH": 2400.0}
    moves = compute_all_moves(feed, period_prices)
    assert "btc" in moves
    assert "eth" not in moves  # skipped — no current price


def test_kelly_size_unclamped():
    # bankroll=120 puts raw half-Kelly size ~within [8, 40], pins actual formula output
    size = kelly_size(bankroll=120.0)
    assert abs(size - 20.0) < 0.10  # formula gives bankroll * ~16.67%


def test_gate1_move_none_price():
    passed, move = gate1_move(cur=None, start=100.0)
    assert passed is False


def test_gate2_confirm_down_fails():
    moves = {"btc": -0.009, "eth": -0.002, "sol": 0.001, "xrp": 0.001, "doge": 0.001, "bnb": 0.003}
    assert gate2_confirm(moves, "down") is False  # only btc <=-0.5% down
