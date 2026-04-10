#!/usr/bin/env python3
"""
Backtest: 15-Min Crypto Latency Arb v3

Uses Binance 1-minute klines to simulate the strategy:
1. Every 15-min window, check price at each minute
2. If price moved >min_move% from period start after min_elapsed minutes
3. Assume entry at a configurable ask price (simulating market conditions)
4. Check if direction held at period end -> win/loss
5. Apply 10% fee on wins, compound bankroll

No Polymarket data needed — the edge is whether the confirmed move holds.
"""

import asyncio
import sys
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

import httpx


@dataclass
class BacktestConfig:
    bankroll: float = 300.0
    bet_fraction: float = 0.08       # 8% of bankroll per trade
    min_move_pct: float = 0.003      # 0.3% minimum move to signal
    min_elapsed_min: int = 5         # Wait 5 min into period
    max_entry_price: float = 0.65    # Simulated ask price when signal fires
    fee_rate: float = 0.10           # 10% on winning payout
    max_concurrent: int = 3
    max_trade_usd: float = 500.0      # Liquidity cap — realistic max fill
    coins: list = None

    def __post_init__(self):
        if self.coins is None:
            self.coins = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


@dataclass
class Trade:
    coin: str
    period_start: int
    signal_minute: int
    direction: str          # "up" or "down"
    move_at_signal: float   # % move when signal fired
    move_at_end: float      # % move at period end
    entry_price: float
    size_usd: float
    won: bool
    pnl: float
    bankroll_after: float


async def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    """Fetch Binance klines in batches."""
    all_klines = []
    current = start_ms
    async with httpx.AsyncClient(timeout=30) as client:
        while current < end_ms:
            resp = await client.get(
                "https://api.binance.com/api/v3/klines",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": current,
                    "endTime": end_ms,
                    "limit": 1000,
                },
            )
            data = resp.json()
            if not data or isinstance(data, dict):
                break
            all_klines.extend(data)
            current = data[-1][0] + 60000  # Next minute
            await asyncio.sleep(0.2)  # Rate limit
    return all_klines


def run_backtest(klines_by_coin: dict, config: BacktestConfig) -> list[Trade]:
    """Run the backtest on downloaded kline data."""
    trades = []
    bankroll = config.bankroll

    # Build minute-price maps: {coin: {unix_minute_ts: close_price}}
    price_maps = {}
    for coin, klines in klines_by_coin.items():
        pm = {}
        for k in klines:
            ts = k[0] // 1000  # Open time in seconds
            close = float(k[4])
            pm[ts] = close
        price_maps[coin] = pm

    # Find all 15-min period boundaries in the data
    all_ts = set()
    for pm in price_maps.values():
        all_ts.update(pm.keys())
    if not all_ts:
        return trades

    min_ts = min(all_ts)
    max_ts = max(all_ts)

    # Align to 15-min boundaries
    period_start = (min_ts // 900) * 900
    periods = []
    while period_start + 900 <= max_ts:
        periods.append(period_start)
        period_start += 900

    for period in periods:
        for coin, pm in price_maps.items():
            # Get price at period start (minute 0)
            start_price = pm.get(period)
            if not start_price:
                continue

            # Get price at period end (minute 14 close ≈ minute 15 open)
            end_price = pm.get(period + 14 * 60)
            if not end_price:
                continue

            # Check each minute from min_elapsed onward for a signal
            signal_fired = False
            for minute in range(config.min_elapsed_min, 14):
                minute_ts = period + minute * 60
                price = pm.get(minute_ts)
                if not price:
                    continue

                move_pct = (price - start_price) / start_price

                if abs(move_pct) >= config.min_move_pct:
                    # Signal! Would we trade?
                    direction = "up" if move_pct > 0 else "down"
                    end_move = (end_price - start_price) / start_price

                    # Simulate entry price based on how stale the market is
                    # Early signals (min 5-7): market still ~50/50, ask ≈ 0.50-0.55
                    # Late signals (min 10-14): market may have repriced, ask ≈ 0.55-0.75
                    staleness_factor = minute / 14  # 0.36 at min 5, 1.0 at min 14
                    entry_price = 0.48 + staleness_factor * 0.20  # 0.55 to 0.68

                    if entry_price > config.max_entry_price:
                        continue

                    # Position size — capped by liquidity
                    size = bankroll * config.bet_fraction
                    size = max(5.0, min(size, bankroll * 0.20, config.max_trade_usd))
                    if size > bankroll - 5.0:
                        continue

                    tokens = size / entry_price
                    won = (direction == "up" and end_move > 0) or \
                          (direction == "down" and end_move < 0)

                    if won:
                        payout = tokens * (1.0 - config.fee_rate)
                        pnl = payout - size
                    else:
                        pnl = -size

                    bankroll += pnl

                    trades.append(Trade(
                        coin=coin.replace("USDT", ""),
                        period_start=period,
                        signal_minute=minute,
                        direction=direction,
                        move_at_signal=move_pct,
                        move_at_end=end_move,
                        entry_price=entry_price,
                        size_usd=round(size, 2),
                        won=won,
                        pnl=round(pnl, 2),
                        bankroll_after=round(bankroll, 2),
                    ))

                    signal_fired = True
                    break  # One trade per coin per period

            if signal_fired and bankroll < 5:
                print(f"  BUSTED at period {datetime.fromtimestamp(period, tz=timezone.utc)}")
                return trades

    return trades


def print_results(trades: list[Trade], config: BacktestConfig):
    """Print backtest summary."""
    if not trades:
        print("No trades generated.")
        return

    wins = sum(1 for t in trades if t.won)
    losses = len(trades) - wins
    total_pnl = sum(t.pnl for t in trades)
    final_bankroll = trades[-1].bankroll_after

    # Daily breakdown
    days = {}
    for t in trades:
        day = datetime.fromtimestamp(t.period_start, tz=timezone.utc).strftime("%Y-%m-%d")
        if day not in days:
            days[day] = {"trades": 0, "wins": 0, "pnl": 0.0}
        days[day]["trades"] += 1
        if t.won:
            days[day]["wins"] += 1
        days[day]["pnl"] += t.pnl

    # Max drawdown
    peak = config.bankroll
    max_dd = 0
    running = config.bankroll
    for t in trades:
        running += t.pnl
        peak = max(peak, running)
        dd = (peak - running) / peak
        max_dd = max(max_dd, dd)

    # Streaks
    max_win_streak = max_loss_streak = 0
    cur_win = cur_loss = 0
    for t in trades:
        if t.won:
            cur_win += 1
            cur_loss = 0
        else:
            cur_loss += 1
            cur_win = 0
        max_win_streak = max(max_win_streak, cur_win)
        max_loss_streak = max(max_loss_streak, cur_loss)

    print(f"\n{'='*65}")
    print(f"  BACKTEST RESULTS — 15-Min Crypto Latency Arb v3")
    print(f"{'='*65}")
    print(f"  Period:        {len(days)} days")
    print(f"  Starting:      ${config.bankroll:.2f}")
    print(f"  Final:         ${final_bankroll:.2f}")
    print(f"  Total P&L:     ${total_pnl:+.2f} ({total_pnl/config.bankroll*100:+.1f}%)")
    print(f"  Trades:        {len(trades)} ({wins}W / {losses}L)")
    print(f"  Win rate:      {wins/len(trades)*100:.1f}%")
    print(f"  Avg win:       ${sum(t.pnl for t in trades if t.won)/max(wins,1):+.2f}")
    print(f"  Avg loss:      ${sum(t.pnl for t in trades if not t.won)/max(losses,1):+.2f}")
    print(f"  Max drawdown:  {max_dd*100:.1f}%")
    print(f"  Win streak:    {max_win_streak}")
    print(f"  Loss streak:   {max_loss_streak}")
    print(f"  Bet fraction:  {config.bet_fraction*100:.0f}%")
    print(f"  Fee rate:      {config.fee_rate*100:.0f}%")
    print(f"{'='*65}")

    print(f"\n  Daily breakdown:")
    print(f"  {'Date':<12} {'Trades':>6} {'Wins':>5} {'Win%':>6} {'P&L':>10}")
    print(f"  {'-'*45}")
    for day, stats in sorted(days.items()):
        wr = stats["wins"]/stats["trades"]*100 if stats["trades"] else 0
        print(f"  {day:<12} {stats['trades']:>6} {stats['wins']:>5} {wr:>5.0f}% ${stats['pnl']:>+9.2f}")

    # Show compounding curve at key milestones
    print(f"\n  Compounding curve:")
    milestones = [10, 25, 50, 100, 200, 500]
    for m in milestones:
        if m <= len(trades):
            t = trades[m-1]
            print(f"  After trade {m:>4}: ${t.bankroll_after:>10.2f}")
    if len(trades) > 0:
        print(f"  Final ({len(trades):>4}):  ${trades[-1].bankroll_after:>10.2f}")

    print(f"{'='*65}")


async def main():
    days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    bet_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08

    config = BacktestConfig(
        bankroll=bankroll,
        bet_fraction=bet_pct,
    )

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    print(f"Fetching {days_back} days of 1-min klines for {len(config.coins)} coins...")
    print(f"  From: {start.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  To:   {end.strftime('%Y-%m-%d %H:%M UTC')}")

    klines = {}
    for coin in config.coins:
        print(f"  Fetching {coin}...", end=" ", flush=True)
        data = await fetch_klines(coin, "1m", start_ms, end_ms)
        klines[coin] = data
        print(f"{len(data)} candles")

    print(f"\nRunning backtest (${bankroll:.0f} bankroll, {bet_pct*100:.0f}% per trade)...")
    trades = run_backtest(klines, config)
    print_results(trades, config)


if __name__ == "__main__":
    asyncio.run(main())
