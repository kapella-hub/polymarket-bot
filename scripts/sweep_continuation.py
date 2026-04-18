"""Threshold sweep for Continuation Mode.

Goal: find thresholds that yield EV-positive trades with meaningful frequency.
"""
import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "data" / "backtest_klines_72h.json"


def build(b): return {int(x[0] // 1000): float(x[4]) for x in b}


def at(s, t):
    for o in range(121):
        p = s.get(t - o)
        if p is not None: return p
    return None


def simulate(data, btc_min, eth_min):
    btc = build(data["BTCUSDT"]); eth = build(data["ETHUSDT"])
    all_ts = sorted(btc.keys())
    first = (all_ts[0] // 900 + 1) * 900
    last = (all_ts[-1] // 900) * 900

    fires, wins = 0, 0
    for period in range(first, last, 900):
        btc_start = at(btc, period); btc_end = at(btc, period + 900)
        if not btc_start or not btc_end: continue
        fired = False
        for elapsed in range(240, 601, 60):
            if fired: break
            now = period + elapsed
            btc_now = at(btc, now); eth_now = at(eth, now)
            btc_5min = at(btc, now - 300); eth_5min = at(eth, now - 300)
            if not all([btc_now, eth_now, btc_5min, eth_5min]): continue
            btc_m = (btc_now - btc_5min) / btc_5min
            eth_m = (eth_now - eth_5min) / eth_5min
            if abs(btc_m) < btc_min: continue
            if abs(eth_m) < eth_min: continue
            if (btc_m > 0) != (eth_m > 0): continue
            # reversal check
            recent = [at(btc, now - o) for o in range(61) if at(btc, now - o)]
            if recent:
                p_now = recent[0]
                if btc_m > 0:
                    rev = (max(recent) - p_now) / p_now if p_now > 0 else 0
                else:
                    rev = (p_now - min(recent)) / p_now if p_now > 0 else 0
                if rev >= 0.0012: continue
            fires += 1
            fired = True
            won = (btc_m > 0 and btc_end > btc_start) or (btc_m < 0 and btc_end < btc_start)
            if won: wins += 1
    return fires, wins


def main():
    data = json.loads(DATA_PATH.read_text())
    configs = [
        (0.003, 0.0025, "0.30%/0.25% (current)"),
        (0.0025, 0.0025, "0.25%/0.25%"),
        (0.0025, 0.002, "0.25%/0.20%"),
        (0.002, 0.002, "0.20%/0.20%"),
        (0.002, 0.0015, "0.20%/0.15%"),
        (0.0015, 0.0015, "0.15%/0.15%"),
    ]
    print("=" * 72)
    print(f"{'Config':30s} {'Fires':>6s} {'/day':>6s} {'W':>3s} {'L':>3s} {'WR':>6s} {'EV@0.80':>10s}")
    print("-" * 72)
    for btc, eth, label in configs:
        fires, wins = simulate(data, btc, eth)
        losses = fires - wins
        wr = wins / fires if fires else 0
        # EV at $0.80 entry (realistic continuation fire price)
        ev = wr * (0.98 - 0.80) - (1 - wr) * 0.80 if fires else 0
        print(f"{label:30s} {fires:>6d} {fires/3:>6.1f} {wins:>3d} {losses:>3d} {wr*100:>5.1f}% ${ev:>+8.3f}")


if __name__ == "__main__":
    main()
