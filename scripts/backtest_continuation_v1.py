"""Backtest Continuation Mode v1 against 72h of Binance 1m klines.

Thresholds mirror docs/superpowers/plans/2026-04-17-continuation-mode-v1.md:
  - window 240-600s elapsed in each 15-min period
  - 5-min rolling BTC >= 0.30%, ETH >= 0.25%
  - same direction
  - 60s peak-to-current counter-move < 0.12%
  - 1 trade per period cap

For each would-fire event, resolve: did the coin end the period on the same side
(buy_up wins if close > period_start, buy_down wins if close < period_start)?
"""
import json
import time
from pathlib import Path
from typing import Optional
import urllib.request


SYMBOLS = ["BTCUSDT", "ETHUSDT"]
HOURS = 72

WINDOW_START = 240
WINDOW_END = 600
LOOKBACK_SEC = 300
REV_WINDOW_SEC = 60
BTC_MIN = 0.003
ETH_MIN = 0.0025
REV_BLOCK = 0.0012
MAX_PER_PERIOD = 1

DATA_PATH = Path(__file__).parent.parent / "data" / "backtest_klines_72h.json"


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list:
    all_bars = []
    cur = start_ms
    while cur < end_ms:
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={symbol}&interval=1m&startTime={cur}&endTime={end_ms}&limit=1000")
        with urllib.request.urlopen(url, timeout=10) as r:
            bars = json.loads(r.read())
        if not bars:
            break
        all_bars.extend(bars)
        cur = bars[-1][0] + 60_000
        if len(bars) < 1000:
            break
    return all_bars


def load_or_fetch() -> dict:
    if DATA_PATH.exists():
        age_h = (time.time() - DATA_PATH.stat().st_mtime) / 3600
        if age_h < 2:
            print(f"Using cached klines ({age_h:.1f}h old)")
            return json.loads(DATA_PATH.read_text())
    print("Fetching 72h klines from Binance...")
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - HOURS * 3600 * 1000
    data = {s: fetch_klines(s, start_ms, end_ms) for s in SYMBOLS}
    DATA_PATH.parent.mkdir(exist_ok=True)
    DATA_PATH.write_text(json.dumps(data))
    print(f"Saved {sum(len(v) for v in data.values())} bars to {DATA_PATH}")
    return data


def build_price_series(bars: list) -> dict[int, float]:
    """Map open_ts (epoch seconds) -> close price."""
    return {int(b[0] // 1000): float(b[4]) for b in bars}


def price_at(series: dict, ts: int) -> Optional[float]:
    """Last known close at or before ts (within 120s)."""
    for offset in range(121):
        p = series.get(ts - offset)
        if p is not None:
            return p
    return None


def compute_move(series: dict, now_ts: int, lookback: int) -> Optional[float]:
    p_now = price_at(series, now_ts)
    p_old = price_at(series, now_ts - lookback)
    if p_now is None or p_old is None or p_old <= 0:
        return None
    return (p_now - p_old) / p_old


def reversal_counter_move(series: dict, now_ts: int, window: int, direction: str) -> float:
    recent = []
    for offset in range(window + 1):
        p = series.get(now_ts - offset)
        if p is not None:
            recent.append(p)
    if not recent:
        return 0.0
    p_now = recent[0]  # newest first
    if direction == "up":
        p_peak = max(recent)
        return max(0.0, (p_peak - p_now) / p_now) if p_now > 0 else 0.0
    else:
        p_trough = min(recent)
        return max(0.0, (p_now - p_trough) / p_now) if p_now > 0 else 0.0


def simulate(data: dict) -> dict:
    btc_series = build_price_series(data["BTCUSDT"])
    eth_series = build_price_series(data["ETHUSDT"])

    all_ts = sorted(btc_series.keys())
    if not all_ts:
        return {"error": "no data"}
    start_ts, end_ts = all_ts[0], all_ts[-1]

    # Find all 15-min period boundaries (epoch % 900 == 0) within range
    first_period = (start_ts // 900 + 1) * 900
    last_period = (end_ts // 900) * 900

    stats = {
        "periods_evaluated": 0,
        "would_fire": 0,
        "up_fires": 0,
        "down_fires": 0,
        "wins": 0,
        "losses": 0,
        "skip_btc_too_small": 0,
        "skip_eth_too_small": 0,
        "skip_direction_mismatch": 0,
        "skip_reversal": 0,
        "fires": [],
    }

    for period_ts in range(first_period, last_period, 900):
        period_end_ts = period_ts + 900
        period_end_price = price_at(btc_series, period_end_ts)
        period_start_price = price_at(btc_series, period_ts)
        if period_end_price is None or period_start_price is None:
            continue
        stats["periods_evaluated"] += 1

        fired_this_period = 0
        for elapsed in range(WINDOW_START, WINDOW_END + 1, 60):
            if fired_this_period >= MAX_PER_PERIOD:
                break
            now_ts = period_ts + elapsed
            btc_move = compute_move(btc_series, now_ts, LOOKBACK_SEC)
            eth_move = compute_move(eth_series, now_ts, LOOKBACK_SEC)
            if btc_move is None or eth_move is None:
                continue

            if abs(btc_move) < BTC_MIN:
                stats["skip_btc_too_small"] += 1
                continue
            if abs(eth_move) < ETH_MIN:
                stats["skip_eth_too_small"] += 1
                continue
            if (btc_move > 0) != (eth_move > 0):
                stats["skip_direction_mismatch"] += 1
                continue
            direction = "up" if btc_move > 0 else "down"
            rev = reversal_counter_move(btc_series, now_ts, REV_WINDOW_SEC, direction)
            if rev >= REV_BLOCK:
                stats["skip_reversal"] += 1
                continue

            # Would fire — resolve
            stats["would_fire"] += 1
            fired_this_period += 1
            if direction == "up":
                stats["up_fires"] += 1
                won = period_end_price > period_start_price
            else:
                stats["down_fires"] += 1
                won = period_end_price < period_start_price
            if won:
                stats["wins"] += 1
            else:
                stats["losses"] += 1
            stats["fires"].append({
                "period_ts": period_ts,
                "elapsed": elapsed,
                "direction": direction,
                "btc_move": round(btc_move * 100, 3),
                "eth_move": round(eth_move * 100, 3),
                "won": won,
                "period_move_pct": round((period_end_price - period_start_price)
                                         / period_start_price * 100, 3),
            })

    return stats


def main():
    data = load_or_fetch()
    stats = simulate(data)

    if "error" in stats:
        print("ERROR:", stats["error"])
        return

    fires = stats["would_fire"]
    wins = stats["wins"]
    total_decisions = fires + sum(stats[k] for k in stats if k.startswith("skip_"))
    wr = (wins / fires * 100) if fires else 0.0
    fires_per_day = fires / (HOURS / 24)

    # EV estimate assuming $0.62 entry (burst sniper default), 2% fee
    entry = 0.65
    payout_win = 1.0 * 0.98
    ev_per_trade = wr / 100 * (payout_win - entry) - (1 - wr / 100) * entry

    print("=" * 60)
    print("CONTINUATION MODE v1 BACKTEST")
    print(f"Window: {HOURS}h | Data points: {len(data['BTCUSDT'])} BTC bars, "
          f"{len(data['ETHUSDT'])} ETH bars")
    print("=" * 60)
    print(f"Periods evaluated:     {stats['periods_evaluated']}")
    print(f"Total evaluation ticks: {total_decisions}")
    print()
    print(f"Would-fire events:     {fires}")
    print(f"  UP direction:        {stats['up_fires']}")
    print(f"  DOWN direction:      {stats['down_fires']}")
    print(f"Fires per day (avg):   {fires_per_day:.1f}")
    print()
    print(f"Hypothetical wins:     {wins}")
    print(f"Hypothetical losses:   {stats['losses']}")
    print(f"Win rate:              {wr:.1f}%")
    print()
    print(f"Est. EV/trade @ ${entry} entry:  ${ev_per_trade:+.3f}")
    print()
    print("Skip reasons:")
    print(f"  btc_move_too_small:   {stats['skip_btc_too_small']}")
    print(f"  eth_move_too_small:   {stats['skip_eth_too_small']}")
    print(f"  direction_mismatch:   {stats['skip_direction_mismatch']}")
    print(f"  reversal_block:       {stats['skip_reversal']}")
    print()
    print("Go/No-Go:")
    go = fires_per_day <= 10 and wr >= 60
    print(f"  Fire rate <=10/day:  {fires_per_day:.1f} "
          f"{'PASS' if fires_per_day <= 10 else 'FAIL'}")
    print(f"  Win rate >=60%:      {wr:.1f}% {'PASS' if wr >= 60 else 'FAIL'}")
    print(f"  DECISION:            {'PROCEED to live' if go else 'DO NOT SHIP — retune'}")
    print()
    if fires and len(stats["fires"]) >= 5:
        print("Sample fires (first 5):")
        for f in stats["fires"][:5]:
            print(f"  period={f['period_ts']} elapsed={f['elapsed']}s "
                  f"dir={f['direction']} btc={f['btc_move']}% eth={f['eth_move']}% "
                  f"result={'WIN' if f['won'] else 'LOSS'} "
                  f"(period_end_move={f['period_move_pct']}%)")


if __name__ == "__main__":
    main()
