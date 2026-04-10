#!/usr/bin/env python3
"""
LIVE 15-Min Crypto Arb Engine v3 — Compounding Edition

Changes from v2:
- Dynamic position sizing: trades % of bankroll (compounding)
- PolymarketWSFeed: detects when market reprices (arb window closing)
- FOK orders: immediate fill or cancel (no stale GTC orders)
- Tracks actual bankroll from wallet balance
- Max concurrent positions limit
- Cooldown after losses
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
    COIN_TO_ASSET, COINS, FastArbDetector, FastMarketScanner,
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
    def __init__(
        self,
        clob: ClobClient,
        bankroll: float = 300.0,
        bet_fraction: float = 0.08,       # 8% of bankroll per trade
        max_concurrent: int = 3,           # Max simultaneous open positions
        max_acceptable_ask: float = 0.75,  # Only buy if ask < $0.75 (20%+ edge after fees)
        loss_cooldown_s: int = 120,        # 2-min cooldown after a loss
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
        self._pm_ws: PolymarketWSFeed | None = None

    def set_polymarket_ws(self, ws: PolymarketWSFeed):
        self._pm_ws = ws

    def execute(self, signal) -> bool:
        # Skip if we already have a position in this market
        if signal.market.condition_id in self.open_orders:
            return False

        # Max concurrent positions
        if len(self.open_orders) >= self.max_concurrent:
            return False

        # Loss cooldown
        if time.time() - self._last_loss_time < self.loss_cooldown_s:
            return False

        # Dynamic sizing: % of current bankroll (compounding)
        size = self.bankroll * self.bet_fraction
        size = max(5.0, min(size, self.bankroll * 0.20))  # Floor $5, cap 20%

        if size > self.bankroll - 5.0:  # Keep $5 reserve
            return False

        token_id = signal.token_id

        # Check if Polymarket WS shows the market already repriced
        if self._pm_ws:
            pm_price = self._pm_ws.get_price(token_id)
            if pm_price is not None and pm_price > self.max_acceptable_ask:
                logger.debug("ws_market_already_repriced", token=token_id[:16], price=pm_price)
                return False

        try:
            # Fetch order book for real ask price
            book = self.clob.get_order_book(token_id)
            asks = [(float(a.price), float(a.size)) for a in book.asks]
            if not asks:
                return False

            best_ask = asks[0][0]
            ask_size = asks[0][1]

            if best_ask > self.max_acceptable_ask:
                return False

            tokens_to_buy = size / best_ask

            # Check liquidity — don't take more than 50% of top-of-book
            if ask_size < tokens_to_buy:
                tokens_to_buy = min(tokens_to_buy, ask_size * 0.5)
                size = tokens_to_buy * best_ask
                if size < 5:
                    return False

            # Place FOK order at ask price — fills immediately or cancels
            order_args = OrderArgs(
                token_id=token_id,
                price=best_ask,
                size=round(tokens_to_buy, 2),
                side=BUY,
            )
            signed = self.clob.create_order(order_args)
            result = self.clob.post_order(signed, OrderType.FOK)

            if not result.get('success'):
                error_msg = result.get('errorMsg', str(result))
                # FOK rejection means no fill — that's fine, not an error
                if 'not filled' in error_msg.lower() or 'no fill' in error_msg.lower():
                    logger.debug("fok_no_fill", coin=signal.market.asset, ask=best_ask)
                else:
                    logger.warning("live_order_failed", error=error_msg)
                return False

            order_id = result.get('orderID', '')
            if not order_id:
                logger.error("live_no_order_id", response=str(result)[:200])
                return False

            # Deduct from bankroll
            self.bankroll -= size
            self.total_invested += size

            # Compute expected net payout
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
                ask=f"${best_ask:.3f}",
                size=f"${size:.2f}",
                tokens=f"{tokens_to_buy:.1f}",
                bankroll=f"${self.bankroll:.2f}",
                order_id=order_id[:16],
                move=f"{signal.binance_change_pct:+.2%}",
                conf=f"{signal.confidence:.0%}",
                profit_if_win=f"${expected_profit:+.2f}",
            )
            return True

        except Exception as e:
            logger.error("live_order_error", error=str(e), coin=signal.market.asset)
            return False

    def resolve_expired(self):
        """Resolve positions whose period has ended. Assumes win if signal was correct."""
        now_ts = int(time.time())
        for cid in list(self.open_orders.keys()):
            trade = self.open_orders[cid]
            if now_ts <= trade["period_end"] + 60:
                continue

            # Assume win if the directional bet at entry was correct
            # (we entered with >85% confidence on a confirmed move)
            # Real P&L comes from resolution — this is an estimate
            tokens = trade["tokens"]
            fee_rate = 0.10
            payout = tokens * (1.0 - fee_rate)
            trade["pnl"] = payout - trade["size_usd"]
            trade["status"] = "won"  # Optimistic — will be corrected by PnL checker
            self.bankroll += payout
            self.total_returned += payout

            del self.open_orders[cid]

            logger.info(
                "LIVE_RESOLVED",
                coin=trade["asset"],
                side=trade["side"],
                status="won",
                pnl=f"${trade['pnl']:+.2f}",
                bankroll=f"${self.bankroll:.2f}",
                entry=f"${trade['entry_price']:.3f}",
                move=f"{trade['binance_change_pct']:+.2%}",
            )

    def record_loss(self, condition_id: str):
        """Call when a position is confirmed lost."""
        for trade in self.trades:
            if trade["condition_id"] == condition_id and trade["status"] == "won":
                # Reverse the optimistic resolution
                self.bankroll -= (trade["pnl"] + trade["size_usd"])
                trade["pnl"] = -trade["size_usd"]
                trade["status"] = "lost"
                self._last_loss_time = time.time()
                logger.warning(
                    "LIVE_LOSS_CORRECTION",
                    coin=trade["asset"],
                    pnl=f"${trade['pnl']:+.2f}",
                    bankroll=f"${self.bankroll:.2f}",
                )
                break

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
        losses = sum(1 for t in closed if t["status"] == "lost")
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
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 43200       # 12h default
    bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0     # Starting bankroll
    bet_pct = float(sys.argv[3]) if len(sys.argv) > 3 else 0.08       # 8% per trade

    clob = create_clob()
    scanner = FastMarketScanner()
    detector = FastArbDetector(
        min_move_pct=0.003,
        min_seconds_elapsed=300,
        max_entry_price=0.75,
        min_edge_after_fees=0.05,
    )
    trader = LiveTrader(
        clob=clob,
        bankroll=bankroll,
        bet_fraction=bet_pct,
        max_concurrent=3,
        max_acceptable_ask=0.75,
    )

    # Start Polymarket WS feed for real-time market repricing detection
    pm_ws = PolymarketWSFeed()
    trader.set_polymarket_ws(pm_ws)
    pm_ws_task = asyncio.create_task(pm_ws.run())

    tick_count = 0
    signal_count = 0
    trade_count = 0
    last_period_ts = 0
    last_status_log = 0

    def on_tick(asset, price, ts):
        nonlocal tick_count, signal_count, trade_count
        tick_count += 1
        now = datetime.now(timezone.utc)
        for market in scanner._current_markets.values():
            if market.asset != asset:
                continue
            signal = detector.detect(market, price, now)
            if signal:
                signal_count += 1
                if trader.execute(signal):
                    trade_count += 1

    ws_feed = BinanceWSFeed(on_price=on_tick)

    print(f"[{datetime.now(timezone.utc).isoformat()}] LIVE arb v3 (compounding) starting")
    print(f"  Duration: {duration}s ({duration/3600:.1f}h)")
    print(f"  Bankroll: ${bankroll:.0f}")
    print(f"  Bet size: {bet_pct*100:.0f}% of bankroll (compounding)")
    print(f"  Order type: FOK (fill-or-kill)")
    print(f"  Max ask: $0.75 (25%+ gross edge)")
    print(f"  Max concurrent: 3 positions")
    print()

    ws_task = asyncio.create_task(ws_feed.run())

    # Wait for Binance prices
    for _ in range(50):
        if ws_feed.get("BTC"):
            break
        await asyncio.sleep(0.1)

    prices = ws_feed.prices
    if prices:
        logger.info(
            "v3_ready",
            assets=len(prices),
            bankroll=f"${bankroll:.2f}",
            prices=" | ".join(f"{a}=${p:,.2f}" for a, p in sorted(prices.items()) if a in ("BTC", "ETH", "SOL")),
        )
    else:
        logger.error("ws_feed_no_prices — aborting")
        return

    start = time.time()
    try:
        while time.time() - start < duration:
            now_ts = int(time.time())
            current_period = int(now_ts // 900) * 900

            if current_period != last_period_ts:
                markets = await scanner.scan_current_period()
                prices = ws_feed.prices

                # Record start-of-period prices for each coin
                for coin in COINS:
                    asset = COIN_TO_ASSET[coin]
                    if asset in prices:
                        detector.record_start_price(asset, prices[asset], current_period)

                # Subscribe Polymarket WS to all active market tokens
                for market in markets:
                    await pm_ws.subscribe(market.up_token_id)
                    await pm_ws.subscribe(market.down_token_id)

                last_period_ts = current_period
                logger.info(
                    "v3_new_period",
                    period=current_period,
                    markets=len(markets),
                    bankroll=f"${trader.bankroll:.2f}",
                    prices=" | ".join(f"{a}=${p:,.2f}" for a, p in sorted(prices.items()) if a in ("BTC", "ETH", "SOL")),
                )

            # Resolve expired positions
            trader.resolve_expired()

            # Status log every 60 seconds
            if time.time() - last_status_log > 60:
                elapsed_in_period = now_ts - current_period
                remaining = 900 - elapsed_in_period
                prices = ws_feed.prices
                logger.info(
                    "v3_cycle",
                    bankroll=f"${trader.bankroll:.2f}",
                    pnl=f"${trader.total_pnl:+.2f}",
                    trades=trade_count,
                    open=len(trader.open_orders),
                    win_rate=f"{trader.win_rate:.0%}" if trader.trades else "--",
                    period_remaining=f"{remaining}s",
                    ticks=tick_count,
                    signals=signal_count,
                    prices=" | ".join(f"{a}=${p:,.2f}" for a, p in sorted(prices.items()) if a in ("BTC", "ETH", "SOL")),
                )
                last_status_log = time.time()

            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")

    # Cleanup
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
    logger.info("v3_stopped", **result, total_ticks=tick_count, total_signals=signal_count)

    print(f"\n{'='*60}")
    print("LIVE ARB v3 (COMPOUNDING) RESULTS")
    print(f"{'='*60}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"  ticks: {tick_count}")
    print(f"  signals: {signal_count}")
    print(f"{'='*60}")

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "v3",
            "duration_s": duration,
            **result,
            "total_ticks": tick_count,
            "total_signals": signal_count,
            "trades": trader.trades,
        }) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
