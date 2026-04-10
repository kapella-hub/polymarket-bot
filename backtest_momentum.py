#!/usr/bin/env python3
"""
Backtest: 15-Min Momentum Prediction

Instead of arbing (reacting to moves), PREDICT direction
before the period starts based on momentum from previous periods.

Buy at $0.51 at period start. If right → $0.90 payout (77% return).
If wrong → lose $0.51. Need >56% accuracy to profit after fees.
"""
import asyncio, sys, httpx
from datetime import datetime, timezone, timedelta

async def fetch_klines(client, symbol, start_ms, end_ms):
    all_klines = []
    current = start_ms
    while current < end_ms:
        resp = await client.get("https://api.binance.com/api/v3/klines", params={
            "symbol": symbol, "interval": "1m",
            "startTime": current, "endTime": end_ms, "limit": 1000,
        })
        data = resp.json()
        if not data or isinstance(data, dict):
            break
        all_klines.extend(data)
        current = data[-1][0] + 60000
        await asyncio.sleep(0.2)
    return all_klines

async def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    bet_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08
    entry_price = 0.51  # What we'd pay at period start
    fee_rate = 0.10

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    print("Fetching %d days of BTC 1-min candles..." % days)
    async with httpx.AsyncClient(timeout=30) as client:
        klines = await fetch_klines(client, "BTCUSDT", start_ms, end_ms)
    print("Got %d candles" % len(klines))

    # Build minute price map
    prices = {}
    for k in klines:
        ts = k[0] // 1000
        prices[ts] = float(k[4])  # Close price

    # Identify 15-min periods
    all_ts = sorted(prices.keys())
    if not all_ts:
        print("No data")
        return

    min_ts = min(all_ts)
    max_ts = max(all_ts)
    period_start = (min_ts // 900) * 900

    periods = []
    while period_start + 900 <= max_ts:
        start_p = prices.get(period_start)
        end_p = prices.get(period_start + 14 * 60)
        if start_p and end_p:
            direction = "up" if end_p > start_p else "down"
            change = (end_p - start_p) / start_p
            periods.append({
                "ts": period_start,
                "start": start_p,
                "end": end_p,
                "direction": direction,
                "change": change,
            })
        period_start += 900

    print("Total 15-min periods: %d" % len(periods))

    # Test multiple momentum strategies
    strategies = {
        "last_1": 1,     # Follow last period
        "last_2": 2,     # Follow majority of last 2
        "last_3": 3,     # Follow majority of last 3
    }

    for name, lookback in strategies.items():
        trades = []
        bank = bankroll
        wins = 0
        losses = 0
        peak = bankroll
        max_dd = 0

        for i in range(lookback, len(periods)):
            # Momentum signal: majority direction of last N periods
            recent = [periods[j]["direction"] for j in range(i - lookback, i)]
            up_count = sum(1 for d in recent if d == "up")
            prediction = "up" if up_count > lookback / 2 else "down"

            # Only trade if clear signal (not 50/50)
            if lookback > 1 and up_count == lookback / 2:
                continue

            # Simulate trade
            actual = periods[i]["direction"]
            size = bank * bet_pct
            size = max(5.0, min(size, bank * 0.15, 500.0))
            if size > bank - 5:
                continue

            tokens = size / entry_price
            won = prediction == actual

            if won:
                payout = tokens * (1.0 - fee_rate)
                pnl = payout - size
                wins += 1
            else:
                pnl = -size
                losses += 1

            bank += pnl
            peak = max(peak, bank)
            dd = (peak - bank) / peak
            max_dd = max(max_dd, dd)

            if bank < 10:
                break

        total = wins + losses
        wr = wins / total * 100 if total else 0
        pnl = bank - bankroll

        print("\n=== %s (lookback=%d) ===" % (name, lookback))
        print("  Trades: %d (%d W / %d L)" % (total, wins, losses))
        print("  Win rate: %.1f%%" % wr)
        print("  Final: $%.2f" % bank)
        print("  P&L: $%+.2f (%+.0f%%)" % (pnl, pnl / bankroll * 100))
        print("  Max drawdown: %.1f%%" % (max_dd * 100))
        print("  Breakeven win rate: ~56%% (at $0.51 entry, 10%% fee)")

    # Also test: momentum + minimum move filter
    print("\n\n=== MOMENTUM + MIN MOVE FILTER ===")
    for min_move in [0.001, 0.002, 0.003, 0.005]:
        bank = bankroll
        wins = 0
        losses = 0
        skipped = 0
        peak = bankroll
        max_dd = 0

        for i in range(1, len(periods)):
            prev = periods[i - 1]
            if abs(prev["change"]) < min_move:
                skipped += 1
                continue

            prediction = prev["direction"]  # Follow momentum
            actual = periods[i]["direction"]

            size = bank * bet_pct
            size = max(5.0, min(size, bank * 0.15, 500.0))
            if size > bank - 5:
                continue

            tokens = size / entry_price
            won = prediction == actual

            if won:
                payout = tokens * (1.0 - fee_rate)
                pnl = payout - size
                wins += 1
            else:
                pnl = -size
                losses += 1

            bank += pnl
            peak = max(peak, bank)
            dd = (peak - bank) / peak
            max_dd = max(max_dd, dd)

            if bank < 10:
                break

        total = wins + losses
        wr = wins / total * 100 if total else 0
        pnl = bank - bankroll
        print("\n  min_move=%.1f%% | Trades: %d (skip %d) | WR: %.1f%% | Final: $%.2f | DD: %.1f%%" % (
            min_move * 100, total, skipped, wr, bank, max_dd * 100))

asyncio.run(main())
