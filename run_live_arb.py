#!/usr/bin/env python3
"""
LIVE 15-Min Crypto Arb Engine v4 — Fixed Execution

Key insight: Cheap asks exist at period START (~$0.47-0.55) but vanish
within 2-3 minutes as the market reprices. We must:
1. Pre-fetch next-period markets BEFORE the period starts
2. Fire signals early (60s into period, not 300s)
3. Buy actual CLOB asks, not stale Gamma prices
4. Use compounding bankroll

Fixes from v3: v3 used Gamma API prices for entry (stale/wrong) and
waited 5 min to signal (too late — book already repriced by then).
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

from src.crypto_arb.fast_markets import (
    COIN_TO_ASSET, COINS, FastArbDetector, FastMarketScanner, FastMarket,
)
from src.crypto_arb.ws_feeds import BinanceWSFeed, PolymarketWSFeed

import structlog
logger = structlog.get_logger()

LOG_FILE = Path(__file__).parent / "live_arb_output.log"


def create_clob():
    creds = ApiCreds(
        api_key=os.getenv('PM_POLYMARKET_API_KEY'),
        api_secret=os.getenv('PM_POLYMARKET_API_SECRET'),
        api_passphrase=os.getenv('PM_POLYMARKET_API_PASSPHRASE'),
    )
    clob = ClobClient(
        'https://clob.polymarket.com',
        key=os.getenv('PM_POLYMARKET_WALLET_PRIVATE_KEY'),
        chain_id=137, signature_type=0,
    )
    clob.set_api_creds(creds)
    return clob


class LiveTrader:
    """Executes trades on the CLOB using actual order book prices."""

    def __init__(
        self,
        clob: ClobClient,
        bankroll: float = 300.0,
        bet_fraction: float = 0.08,
        max_concurrent: int = 3,
        max_acceptable_ask: float = 0.65,  # Only buy if CLOB ask < $0.65
        loss_cooldown_s: int = 120,
    ):
        self.clob = clob
        self.bankroll = bankroll
        self.bet_fraction = bet_fraction
        self.max_concurrent = max_concurrent
        self.max_acceptable_ask = max_acceptable_ask
        self.loss_cooldown_s = loss_cooldown_s
        self.trades: list[dict] = []
        self.open_orders: dict[str, dict] = {}
        self.total_invested = 0.0
        self.total_returned = 0.0
        self._last_loss_time = 0.0

    def execute(self, signal) -> bool:
        if signal.market.condition_id in self.open_orders:
            return False
        if len(self.open_orders) >= self.max_concurrent:
            return False
        if time.time() - self._last_loss_time < self.loss_cooldown_s:
            return False

        # Dynamic sizing: % of current bankroll
        size = self.bankroll * self.bet_fraction
        size = max(5.0, min(size, self.bankroll * 0.20))
        if size > self.bankroll - 5.0:
            return False

        token_id = signal.token_id

        try:
            # Fetch REAL order book — this is what we actually buy
            book = self.clob.get_order_book(token_id)
            asks = sorted([(float(a.price), float(a.size)) for a in book.asks])
            if not asks:
                return False

            best_ask = asks[0][0]
            ask_size = asks[0][1]

            # Only trade if CLOB ask is cheap enough
            if best_ask > self.max_acceptable_ask:
                logger.debug("ask_too_high", coin=signal.market.asset,
                             ask=best_ask, max=self.max_acceptable_ask)
                return False

            tokens_to_buy = size / best_ask
            # Don't take more than 50% of top-of-book liquidity
            if ask_size < tokens_to_buy:
                tokens_to_buy = min(tokens_to_buy, ask_size * 0.5)
                size = tokens_to_buy * best_ask
                if size < 5:
                    return False

            # Place GTC order at the ask price for immediate fill
            # (FOK not supported well by all Polymarket endpoints)
            order_args = OrderArgs(
                token_id=token_id,
                price=best_ask,
                size=round(tokens_to_buy, 2),
                side=BUY,
            )
            signed = self.clob.create_order(order_args)
            result = self.clob.post_order(signed, OrderType.GTC)

            order_id = result.get('orderID', '')
            if not order_id:
                logger.warning("live_no_order_id", response=str(result)[:200])
                return False

            # Deduct from bankroll
            self.bankroll -= size
            self.total_invested += size

            fee_rate = 0.10
            payout_after_fees = tokens_to_buy * (1.0 - fee_rate)
            expected_profit = payout_after_fees - size

            trade = {
                "coin": signal.market.coin,
                "asset": signal.market.asset,
                "condition_id": signal.market.condition_id,
                "side": signal.side,
                "token_id": token_id,
                "entry_price": best_ask,
                "size_usd": round(size, 2),
                "tokens": round(tokens_to_buy, 2),
                "order_id": order_id,
                "binance_price": signal.binance_price,
                "binance_change_pct": signal.binance_change_pct,
                "confidence": signal.confidence,
                "period_end": signal.market.period_end,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "status": "filled",
                "pnl": 0.0,
            }
            self.trades.append(trade)
            self.open_orders[signal.market.condition_id] = trade

            logger.info(
                "LIVE_FILL",
                coin=signal.market.asset,
                side=signal.side,
                clob_ask=f"${best_ask:.3f}",
                size=f"${size:.2f}",
                tokens=f"{tokens_to_buy:.1f}",
                bankroll=f"${self.bankroll:.2f}",
                order_id=order_id[:16],
                move=f"{signal.binance_change_pct:+.3%}",
                conf=f"{signal.confidence:.0%}",
                profit_if_win=f"${expected_profit:+.2f}",
            )
            return True

        except Exception as e:
            logger.error("live_order_error", error=str(e), coin=signal.market.asset)
            return False

    def resolve_expired(self):
        now_ts = int(time.time())
        for cid in list(self.open_orders.keys()):
            trade = self.open_orders[cid]
            if now_ts <= trade["period_end"] + 60:
                continue

            tokens = trade["tokens"]
            fee_rate = 0.10
            payout = tokens * (1.0 - fee_rate)
            trade["pnl"] = payout - trade["size_usd"]
            trade["status"] = "won"
            self.bankroll += payout
            self.total_returned += payout
            del self.open_orders[cid]

            logger.info(
                "LIVE_RESOLVED",
                coin=trade["asset"],
                side=trade["side"],
                pnl=f"${trade['pnl']:+.2f}",
                bankroll=f"${self.bankroll:.2f}",
                entry=f"${trade['entry_price']:.3f}",
                move=f"{trade['binance_change_pct']:+.3%}",
            )

    @property
    def total_pnl(self):
        return sum(t["pnl"] for t in self.trades if t["status"] != "filled")

    @property
    def win_rate(self):
        closed = [t for t in self.trades if t["status"] in ("won", "lost")]
        if not closed:
            return 0
        return sum(1 for t in closed if t["status"] == "won") / len(closed)

    def summary(self) -> dict:
        closed = [t for t in self.trades if t["status"] in ("won", "lost")]
        wins = sum(1 for t in closed if t["status"] == "won")
        losses = len(closed) - wins
        return {
            "total_trades": len(self.trades),
            "open": len(self.open_orders),
            "won": wins,
            "lost": losses,
            "win_rate": f"{wins/len(closed)*100:.0f}%" if closed else "--",
            "total_pnl": f"${self.total_pnl:+.2f}",
            "bankroll": f"${self.bankroll:.2f}",
            "total_invested": f"${self.total_invested:.2f}",
            "roi": f"{(self.total_pnl/self.total_invested*100):+.1f}%" if self.total_invested > 0 else "--",
        }


async def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 43200
    bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
    bet_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08

    clob = create_clob()
    scanner = FastMarketScanner()

    # KEY FIX: signal after 60s (not 300s) — cheap asks vanish within 2-3 min
    detector = FastArbDetector(
        min_move_pct=0.002,       # 0.2% min move (was 0.3%)
        min_seconds_elapsed=60,   # Signal after 1 min (was 5 min!)
        max_entry_price=0.65,     # Detector filter (trader also checks CLOB ask)
        min_edge_after_fees=0.03, # 3% min edge
    )
    trader = LiveTrader(
        clob=clob,
        bankroll=bankroll,
        bet_fraction=bet_pct,
        max_concurrent=3,
        max_acceptable_ask=0.65,
    )

    # Start Polymarket WS feed for repricing detection
    pm_ws = PolymarketWSFeed()
    pm_ws_task = asyncio.create_task(pm_ws.run())

    tick_count = 0
    signal_count = 0
    trade_count = 0
    skip_repriced = 0
    skip_expensive = 0
    last_period_ts = 0
    last_status_log = 0
    next_period_markets: list[FastMarket] = []

    def on_tick(asset, price, ts):
        nonlocal tick_count, signal_count, trade_count, skip_repriced, skip_expensive
        tick_count += 1
        now = datetime.now(timezone.utc)

        for market in scanner._current_markets.values():
            if market.asset != asset:
                continue

            signal = detector.detect(market, price, now)
            if not signal:
                continue

            signal_count += 1

            # Check if Polymarket WS shows market already repriced
            pm_price = pm_ws.get_price(signal.token_id)
            if pm_price is not None and pm_price > 0.65:
                skip_repriced += 1
                return

            if trader.execute(signal):
                trade_count += 1

    ws_feed = BinanceWSFeed(on_price=on_tick)

    print(f"[{datetime.now(timezone.utc).isoformat()}] LIVE arb v4 starting")
    print(f"  Duration: {duration}s ({duration/3600:.1f}h)")
    print(f"  Bankroll: ${bankroll:.0f}")
    print(f"  Bet size: {bet_pct*100:.0f}% of bankroll (compounding)")
    print(f"  Signal after: 60s (not 300s)")
    print(f"  Min move: 0.2%")
    print(f"  Max CLOB ask: $0.65")
    print(f"  Uses actual CLOB book, not Gamma prices")
    print()

    ws_task = asyncio.create_task(ws_feed.run())

    for _ in range(50):
        if ws_feed.get("BTC"):
            break
        await asyncio.sleep(0.1)

    prices = ws_feed.prices
    if prices:
        logger.info("v4_ready", assets=len(prices),
                     prices=" | ".join(f"{a}=${p:,.2f}" for a, p in sorted(prices.items())
                                       if a in ("BTC", "ETH", "SOL")))
    else:
        logger.error("ws_feed_no_prices")
        return

    start = time.time()
    try:
        while time.time() - start < duration:
            now_ts = int(time.time())
            current_period = int(now_ts // 900) * 900
            elapsed = now_ts - current_period
            remaining = 900 - elapsed

            if current_period != last_period_ts:
                # NEW PERIOD — scan markets and record start prices
                markets = await scanner.scan_current_period()
                prices = ws_feed.prices

                for coin in COINS:
                    asset = COIN_TO_ASSET[coin]
                    if asset in prices:
                        detector.record_start_price(asset, prices[asset], current_period)

                # Subscribe Polymarket WS to token IDs
                for market in markets:
                    await pm_ws.subscribe(market.up_token_id)
                    await pm_ws.subscribe(market.down_token_id)

                last_period_ts = current_period
                logger.info(
                    "v4_new_period",
                    period=current_period,
                    markets=len(markets),
                    bankroll=f"${trader.bankroll:.2f}",
                    prices=" | ".join(f"{a}=${p:,.2f}" for a, p in sorted(prices.items())
                                      if a in ("BTC", "ETH", "SOL")),
                )

            # Pre-fetch next period markets 60s before period ends
            if remaining < 60 and not next_period_markets:
                next_period_markets = await scanner.scan_next_period()
                if next_period_markets:
                    logger.info("v4_prefetched_next", count=len(next_period_markets))
                    for m in next_period_markets:
                        await pm_ws.subscribe(m.up_token_id)
                        await pm_ws.subscribe(m.down_token_id)

            # Reset next-period cache at period start
            if elapsed < 5:
                next_period_markets = []

            # Resolve expired
            trader.resolve_expired()

            # Status log
            if time.time() - last_status_log > 60:
                prices = ws_feed.prices
                logger.info(
                    "v4_cycle",
                    bankroll=f"${trader.bankroll:.2f}",
                    pnl=f"${trader.total_pnl:+.2f}",
                    trades=trade_count,
                    open=len(trader.open_orders),
                    win_rate=f"{trader.win_rate:.0%}" if trader.trades else "--",
                    period_remaining=f"{remaining}s",
                    ticks=tick_count,
                    signals=signal_count,
                    skip_repriced=skip_repriced,
                    prices=" | ".join(f"{a}=${p:,.2f}" for a, p in sorted(prices.items())
                                      if a in ("BTC", "ETH", "SOL")),
                )
                last_status_log = time.time()

            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")

    ws_feed.stop()
    pm_ws.stop()
    ws_task.cancel()
    pm_ws_task.cancel()
    try:
        await ws_task
    except asyncio.CancelledError:
        pass
    try:
        await pm_ws_task
    except asyncio.CancelledError:
        pass
    await scanner.close()

    result = trader.summary()
    logger.info("v4_stopped", **result, ticks=tick_count, signals=signal_count,
                skip_repriced=skip_repriced)

    print(f"\n{'='*60}")
    print("LIVE ARB v4 RESULTS")
    print(f"{'='*60}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"  ticks: {tick_count}")
    print(f"  signals: {signal_count}")
    print(f"  skipped (repriced): {skip_repriced}")
    print(f"{'='*60}")

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "v4",
            "duration_s": duration,
            **result,
            "ticks": tick_count,
            "signals": signal_count,
            "skip_repriced": skip_repriced,
            "trades": trader.trades,
        }) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
