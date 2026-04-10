#!/usr/bin/env python3
"""
LIVE 15-Min Crypto Arb Engine v5 — Blind Sniper

Key insight: the CLOB reprices in 5-10 seconds. By the time you fetch
the order book, cheap asks are gone. Solution: don't check the book.

Strategy:
1. At period start, record Binance prices
2. On FIRST tick showing >0.15% move (as early as 5 seconds in), fire immediately
3. Place limit buy at $0.52-0.55 WITHOUT checking the order book
4. If filled: great, we bought the winner cheap. If not: cancel after 10s.
5. Compound bankroll — reinvest wins.

Why this works: we're placing orders BEFORE market makers reprice.
The book starts at $0.50/$0.50. Our $0.52 limit sits at near-market.
When the move confirms, our order is already in the book.

Also scans hourly crypto threshold markets for wider-window opportunities.
"""

import asyncio
import json
import os
import sys
import time
import threading
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


class SniperTrader:
    """Ultra-low-latency execution. Places limit orders blind — no book check.

    Speed matters more than information here. A limit buy at $0.52
    placed 5 seconds into the period beats a perfectly-priced order
    placed at 60 seconds when the book is already at $0.88.
    """

    def __init__(
        self,
        clob: ClobClient,
        bankroll: float = 300.0,
        bet_fraction: float = 0.08,
        max_concurrent: int = 3,
        snipe_price: float = 0.62,       # Fixed price for blind limit orders
        cancel_after_s: float = 10.0,    # Cancel unfilled orders after this
    ):
        self.clob = clob
        self.bankroll = bankroll
        self.bet_fraction = bet_fraction
        self.max_concurrent = max_concurrent
        self.snipe_price = snipe_price
        self.cancel_after_s = cancel_after_s
        self.trades: list[dict] = []
        self.open_orders: dict[str, dict] = {}
        self.pending_cancels: list[dict] = []  # Orders to cancel if not filled
        self.total_invested = 0.0
        self.total_returned = 0.0
        self._last_loss_time = 0.0
        self._lock = threading.Lock()  # Protect from concurrent WS callbacks

    def snipe(self, signal) -> bool:
        """Fire immediately at fixed price. No book check. Speed is everything."""
        with self._lock:
            if signal.market.condition_id in self.open_orders:
                return False
            if len(self.open_orders) >= self.max_concurrent:
                return False
            if time.time() - self._last_loss_time < 60:  # 1-min cooldown
                return False

            size = self.bankroll * self.bet_fraction
            size = max(5.0, min(size, self.bankroll * 0.20))
            if size > self.bankroll - 5.0:
                return False

        token_id = signal.token_id
        # Use the lower of our snipe price or the signal's entry price
        limit_price = min(self.snipe_price, signal.entry_price + 0.03)

        try:
            tokens_to_buy = size / limit_price

            order_args = OrderArgs(
                token_id=token_id,
                price=limit_price,
                size=round(tokens_to_buy, 2),
                side=BUY,
            )

            t0 = time.monotonic()
            signed = self.clob.create_order(order_args)
            result = self.clob.post_order(signed, OrderType.GTC)
            latency_ms = (time.monotonic() - t0) * 1000

            order_id = result.get('orderID', '')
            if not order_id:
                logger.debug("snipe_no_fill", coin=signal.market.asset,
                             price=limit_price, response=str(result)[:100])
                return False

            with self._lock:
                self.bankroll -= size
                self.total_invested += size

            fee_rate = 0.10
            payout_if_win = tokens_to_buy * (1.0 - fee_rate)
            profit_if_win = payout_if_win - size

            trade = {
                "coin": signal.market.coin,
                "asset": signal.market.asset,
                "condition_id": signal.market.condition_id,
                "side": signal.side,
                "token_id": token_id,
                "entry_price": limit_price,
                "size_usd": round(size, 2),
                "tokens": round(tokens_to_buy, 2),
                "order_id": order_id,
                "binance_price": signal.binance_price,
                "binance_change_pct": signal.binance_change_pct,
                "confidence": signal.confidence,
                "period_end": signal.market.period_end,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "placed_at": time.time(),
                "status": "pending",  # Pending until we confirm fill
                "pnl": 0.0,
            }

            with self._lock:
                self.trades.append(trade)
                self.open_orders[signal.market.condition_id] = trade
                self.pending_cancels.append(trade)

            logger.info(
                "SNIPE_PLACED",
                coin=signal.market.asset,
                side=signal.side,
                limit=f"${limit_price:.3f}",
                size=f"${size:.2f}",
                tokens=f"{tokens_to_buy:.1f}",
                bankroll=f"${self.bankroll:.2f}",
                order_id=order_id[:16],
                move=f"{signal.binance_change_pct:+.3%}",
                latency=f"{latency_ms:.0f}ms",
                profit_if_win=f"${profit_if_win:+.2f}",
            )
            return True

        except Exception as e:
            logger.error("snipe_error", error=str(e), coin=signal.market.asset)
            return False

    def check_pending_orders(self):
        """Cancel orders that haven't filled within cancel_after_s."""
        now = time.time()
        with self._lock:
            still_pending = []
            for trade in self.pending_cancels:
                age = now - trade["placed_at"]
                if age < self.cancel_after_s:
                    still_pending.append(trade)
                    continue

                # Time's up — check if filled, cancel if not
                order_id = trade["order_id"]
                try:
                    # Try to cancel — if it fails, the order was filled
                    self.clob.cancel(order_id)
                    # Cancel succeeded — order was NOT filled
                    trade["status"] = "cancelled"
                    self.bankroll += trade["size_usd"]  # Refund
                    self.total_invested -= trade["size_usd"]
                    del self.open_orders[trade["condition_id"]]
                    logger.info(
                        "SNIPE_CANCELLED",
                        coin=trade["asset"],
                        side=trade["side"],
                        age=f"{age:.1f}s",
                        bankroll=f"${self.bankroll:.2f}",
                    )
                except Exception:
                    # Cancel failed — order was FILLED (or already gone)
                    trade["status"] = "filled"
                    logger.info(
                        "SNIPE_CONFIRMED",
                        coin=trade["asset"],
                        side=trade["side"],
                        entry=f"${trade['entry_price']:.3f}",
                        size=f"${trade['size_usd']:.2f}",
                        age=f"{age:.1f}s",
                    )

            self.pending_cancels = still_pending

    def resolve_expired(self):
        """Resolve positions whose period has ended."""
        now_ts = int(time.time())
        with self._lock:
            for cid in list(self.open_orders.keys()):
                trade = self.open_orders[cid]
                if trade["status"] != "filled":
                    continue
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
                    "SNIPE_RESOLVED",
                    coin=trade["asset"],
                    side=trade["side"],
                    pnl=f"${trade['pnl']:+.2f}",
                    bankroll=f"${self.bankroll:.2f}",
                    entry=f"${trade['entry_price']:.3f}",
                    move=f"{trade['binance_change_pct']:+.3%}",
                )

    @property
    def total_pnl(self):
        return sum(t["pnl"] for t in self.trades if t["status"] in ("won", "lost"))

    @property
    def win_rate(self):
        closed = [t for t in self.trades if t["status"] in ("won", "lost")]
        if not closed:
            return 0
        return sum(1 for t in closed if t["status"] == "won") / len(closed)

    @property
    def fill_rate(self):
        attempted = [t for t in self.trades if t["status"] != "pending"]
        if not attempted:
            return 0
        filled = sum(1 for t in attempted if t["status"] in ("filled", "won", "lost"))
        return filled / len(attempted)

    def summary(self) -> dict:
        closed = [t for t in self.trades if t["status"] in ("won", "lost")]
        cancelled = sum(1 for t in self.trades if t["status"] == "cancelled")
        filled = sum(1 for t in self.trades if t["status"] in ("filled", "won", "lost"))
        wins = sum(1 for t in closed if t["status"] == "won")
        losses = len(closed) - wins
        return {
            "total_attempts": len(self.trades),
            "filled": filled,
            "cancelled": cancelled,
            "open": len(self.open_orders),
            "won": wins,
            "lost": losses,
            "fill_rate": f"{self.fill_rate*100:.0f}%" if self.trades else "--",
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
    snipe_price = float(sys.argv[4]) if len(sys.argv) > 4 else 0.62

    clob = create_clob()
    scanner = FastMarketScanner()

    # AGGRESSIVE detector: signal after just 5 seconds, 0.15% move
    detector = FastArbDetector(
        min_move_pct=0.0015,      # 0.15% minimum move (half of v4)
        min_seconds_elapsed=5,    # 5 seconds! (was 300 in v2, 60 in v4)
        max_entry_price=0.60,     # Detector filter
        min_edge_after_fees=0.02, # 2% minimum edge
    )
    trader = SniperTrader(
        clob=clob,
        bankroll=bankroll,
        bet_fraction=bet_pct,
        max_concurrent=3,
        snipe_price=snipe_price,
        cancel_after_s=15.0,
    )

    pm_ws = PolymarketWSFeed()
    pm_ws_task = asyncio.create_task(pm_ws.run())

    tick_count = 0
    signal_count = 0
    snipe_count = 0
    last_period_ts = 0
    last_status_log = 0

    def on_tick(asset, price, ts):
        nonlocal tick_count, signal_count, snipe_count
        tick_count += 1
        now = datetime.now(timezone.utc)

        for market in scanner._current_markets.values():
            if market.asset != asset:
                continue
            signal = detector.detect(market, price, now)
            if not signal:
                continue

            signal_count += 1

            # Check Polymarket WS — skip if already repriced
            pm_price = pm_ws.get_price(signal.token_id)
            if pm_price is not None and pm_price > 0.65:
                return

            # FIRE IMMEDIATELY — no book check
            if trader.snipe(signal):
                snipe_count += 1

    ws_feed = BinanceWSFeed(on_price=on_tick)

    print(f"[{datetime.now(timezone.utc).isoformat()}] LIVE arb v5 SNIPER starting")
    print(f"  Duration: {duration}s ({duration/3600:.1f}h)")
    print(f"  Bankroll: ${bankroll:.0f}")
    print(f"  Bet size: {bet_pct*100:.0f}% of bankroll (compounding)")
    print(f"  Snipe price: ${snipe_price} (blind limit order)")
    print(f"  Signal after: 5 SECONDS (not 60s)")
    print(f"  Min move: 0.15%")
    print(f"  Cancel unfilled after: 15s")
    print(f"  NO order book check — pure speed")
    print()

    ws_task = asyncio.create_task(ws_feed.run())

    for _ in range(50):
        if ws_feed.get("BTC"):
            break
        await asyncio.sleep(0.1)

    prices = ws_feed.prices
    if prices:
        logger.info("v5_ready", assets=len(prices),
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
                markets = await scanner.scan_current_period()
                prices = ws_feed.prices

                for coin in COINS:
                    asset = COIN_TO_ASSET[coin]
                    if asset in prices:
                        detector.record_start_price(asset, prices[asset], current_period)

                for market in markets:
                    await pm_ws.subscribe(market.up_token_id)
                    await pm_ws.subscribe(market.down_token_id)

                last_period_ts = current_period
                logger.info(
                    "v5_new_period",
                    period=current_period,
                    markets=len(markets),
                    bankroll=f"${trader.bankroll:.2f}",
                    prices=" | ".join(f"{a}=${p:,.2f}" for a, p in sorted(prices.items())
                                      if a in ("BTC", "ETH", "SOL")),
                )

            # Pre-fetch next period 60s before end
            if remaining < 60 and remaining > 55:
                next_markets = await scanner.scan_next_period()
                if next_markets:
                    for m in next_markets:
                        await pm_ws.subscribe(m.up_token_id)
                        await pm_ws.subscribe(m.down_token_id)

            # Check pending orders (cancel stale ones)
            trader.check_pending_orders()
            trader.resolve_expired()

            if time.time() - last_status_log > 60:
                prices = ws_feed.prices
                logger.info(
                    "v5_cycle",
                    bankroll=f"${trader.bankroll:.2f}",
                    pnl=f"${trader.total_pnl:+.2f}",
                    snipes=snipe_count,
                    fill_rate=f"{trader.fill_rate*100:.0f}%" if trader.trades else "--",
                    open=len(trader.open_orders),
                    win_rate=f"{trader.win_rate:.0%}" if [t for t in trader.trades if t["status"] in ("won","lost")] else "--",
                    period_remaining=f"{remaining}s",
                    ticks=tick_count,
                    signals=signal_count,
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
    logger.info("v5_stopped", **result, ticks=tick_count, signals=signal_count)

    print(f"\n{'='*60}")
    print("LIVE ARB v5 SNIPER RESULTS")
    print(f"{'='*60}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"  ticks: {tick_count}")
    print(f"  signals: {signal_count}")
    print(f"{'='*60}")

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "v5",
            "duration_s": duration,
            **result,
            "ticks": tick_count,
            "signals": signal_count,
            "trades": trader.trades,
        }) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
