"""Backtest Correlated Burst Mode against cached 72h Binance klines.

Rules:
  - Fires in first 90s of each 15-min period
  - >=2 coins moved >=0.25% in same direction from period open
  - At least one must be BTC or ETH (tier 1)
  - 1 fire per period max
  - Resolution: did BTC/ETH/majority end the period on the signal side?
"""
import json
import time
from pathlib import Path


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]  # only 3 available in cache
WINDOW_SEC = 90
MIN_MOVE = 0.0020  # lowered from 0.25% to 0.20% to probe real regime
MIN_COINS = 2
REQUIRE_TIER1 = True
TIER1 = {"BTCUSDT", "ETHUSDT"}

DATA_PATH = Path(__file__).parent.parent / "data" / "backtest_klines_72h.json"


def build_price_series(bars: list) -> dict[int, float]:
    return {int(b[0] // 1000): float(b[4]) for b in bars}


def price_at(series: dict, ts: int) -> float | None:
    for offset in range(121):
        p = series.get(ts - offset)
        if p is not None:
            return p
    return None


def simulate(data: dict) -> dict:
    series = {s: build_price_series(data[s]) for s in SYMBOLS if s in data}
    if not series:
        return {"error": "no data"}

    all_ts = sorted(series["BTCUSDT"].keys())
    start_ts, end_ts = all_ts[0], all_ts[-1]
    first_period = (start_ts // 900 + 1) * 900
    last_period = (end_ts // 900) * 900

    stats = {
        "periods_evaluated": 0,
        "would_fire": 0,
        "up_fires": 0,
        "down_fires": 0,
        "wins": 0,
        "losses": 0,
        "fires": [],
    }

    for period_ts in range(first_period, last_period, 900):
        period_end_ts = period_ts + 900
        btc_start = price_at(series["BTCUSDT"], period_ts)
        btc_end = price_at(series["BTCUSDT"], period_end_ts)
        if btc_start is None or btc_end is None:
            continue
        stats["periods_evaluated"] += 1

        # Sample at various elapsed times in first 90s; fire on first qualifying
        fired_this_period = False
        for elapsed in range(30, WINDOW_SEC + 1, 30):
            if fired_this_period:
                break
            now_ts = period_ts + elapsed
            moves = {}
            for sym in SYMBOLS:
                if sym not in series:
                    continue
                p_start = price_at(series[sym], period_ts)
                p_now = price_at(series[sym], now_ts)
                if p_start is None or p_now is None or p_start <= 0:
                    continue
                moves[sym] = (p_now - p_start) / p_start

            # Check UP and DOWN directions
            for direction in ("up", "down"):
                sign = 1 if direction == "up" else -1
                aligned = [(s, m) for s, m in moves.items()
                           if (m * sign) >= MIN_MOVE]
                if len(aligned) < MIN_COINS:
                    continue
                if REQUIRE_TIER1 and not any(s in TIER1 for s, _ in aligned):
                    continue

                # Fire
                stats["would_fire"] += 1
                fired_this_period = True
                if direction == "up":
                    stats["up_fires"] += 1
                    won = btc_end > btc_start
                else:
                    stats["down_fires"] += 1
                    won = btc_end < btc_start
                if won:
                    stats["wins"] += 1
                else:
                    stats["losses"] += 1
                stats["fires"].append({
                    "period_ts": period_ts,
                    "elapsed": elapsed,
                    "direction": direction,
                    "aligned_count": len(aligned),
                    "aligned_coins": [s for s, _ in aligned],
                    "moves": {s: round(m * 100, 3) for s, m in moves.items()},
                    "won": won,
                })
                break

    return stats


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found")
        return
    data = json.loads(DATA_PATH.read_text())
    stats = simulate(data)
    if "error" in stats:
        print("ERROR:", stats["error"])
        return

    fires = stats["would_fire"]
    wins = stats["wins"]
    wr = (wins / fires * 100) if fires else 0.0
    fires_per_day = fires / 3.0  # 72h

    entry = 0.62  # burst-mode typical entry
    payout_win = 1.0 * 0.98
    ev = wr / 100 * (payout_win - entry) - (1 - wr / 100) * entry

    print("=" * 60)
    print("CORRELATED BURST MODE BACKTEST — 72h")
    print("=" * 60)
    print(f"Periods:               {stats['periods_evaluated']}")
    print(f"Would-fire events:     {fires}")
    print(f"  UP direction:        {stats['up_fires']}")
    print(f"  DOWN direction:      {stats['down_fires']}")
    print(f"Fires per day (avg):   {fires_per_day:.1f}")
    print()
    print(f"Wins / Losses:         {wins} / {stats['losses']}")
    print(f"Win rate:              {wr:.1f}%")
    print(f"Est. EV/trade @ ${entry}: ${ev:+.3f}")
    print()
    go_fr = fires_per_day <= 15
    go_wr = wr >= 65
    print(f"Gate fire rate <=15/d: {'PASS' if go_fr else 'FAIL'} ({fires_per_day:.1f})")
    print(f"Gate win rate >=65%:   {'PASS' if go_wr else 'FAIL'} ({wr:.1f}%)")
    print(f"DECISION:              {'SHIP' if (go_fr and go_wr) else 'DO NOT SHIP'}")
    print()
    if stats["fires"][:5]:
        print("Sample fires:")
        for f in stats["fires"][:5]:
            print(f"  period={f['period_ts']} elapsed={f['elapsed']}s "
                  f"dir={f['direction']} coins={f['aligned_coins']} "
                  f"moves={f['moves']} {'WIN' if f['won'] else 'LOSS'}")


if __name__ == "__main__":
    main()
