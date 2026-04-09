#!/usr/bin/env python3
"""
LIVE 15-Min Crypto Latency Arbitrage Engine (WebSocket)

Places REAL orders on Polymarket. Start with small sizes.
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
from src.crypto_arb.ws_feeds import BinanceWSFeed

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
    def __init__(self, clob: ClobClient, max_per_trade: float = 10.0):
        self.clob = clob
        self.max_per_trade = max_per_trade
        self.trades: list[dict] = []
        self.open_orders: dict[str, dict] = {}  # condition_id -> trade info
        self.total_invested = 0.0
        self.total_returned = 0.0

    def execute(self, signal) -> bool:
        if signal.market.condition_id in self.open_orders:
            return False

        size = self.max_per_trade
        token_id = signal.token_id
        price = signal.entry_price

        # Place a limit order at the current market price
        try:
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=round(size / price, 2),  # Convert to token quantity
                side=BUY,
            )
            signed = self.clob.create_order(order_args)
            result = self.clob.post_order(signed, OrderType.GTC)

            if not result.get('success'):
                logger.warning("live_order_failed", error=result.get('errorMsg', ''))
                return False

            order_id = result.get('orderID', '')

            trade = {
                "coin": signal.market.coin,
                "asset": signal.market.asset,
                "condition_id": signal.market.condition_id,
                "side": signal.side,
                "token_id": token_id,
                "entry_price": price,
                "size_usd": size,
                "order_id": order_id,
                "binance_price": signal.binance_price,
                "binance_change_pct": signal.binance_change_pct,
                "confidence": signal.confidence,
                "period_end": signal.market.period_end,
                "entry_time": datetime.now(timezone.utc).isoformat(),
                "status": "placed",
            }
            self.trades.append(trade)
            self.open_orders[signal.market.condition_id] = trade
            self.total_invested += size

            logger.info(
                "LIVE_TRADE",
                coin=signal.market.asset,
                side=signal.side,
                price=f"${price:.3f}",
                size=f"${size:.2f}",
                order_id=order_id[:16],
                move=f"{signal.binance_change_pct:+.2%}",
                conf=f"{signal.confidence:.0%}",
            )
            return True

        except Exception as e:
            logger.error("live_order_error", error=str(e))
            return False

    def cleanup_expired(self):
        """Remove tracking for expired periods. Resolution is automatic on Polymarket."""
        now_ts = int(time.time())
        for cid in list(self.open_orders.keys()):
            trade = self.open_orders[cid]
            if now_ts > trade["period_end"] + 120:  # 2 min after period ends
                trade["status"] = "resolved"
                del self.open_orders[cid]
                logger.info(
                    "live_period_ended",
                    coin=trade["asset"],
                    side=trade["side"],
                    price=f"${trade['entry_price']:.3f}",
                )

    def summary(self) -> dict:
        return {
            "total_trades": len(self.trades),
            "open": len(self.open_orders),
            "total_invested": f"${self.total_invested:.2f}",
            "trades": [
                {
                    "coin": t["asset"], "side": t["side"],
                    "price": t["entry_price"], "size": t["size_usd"],
                    "status": t["status"],
                }
                for t in self.trades[-10:]
            ],
        }


async def main():
    duration = int(sys.argv[1]) if len(sys.argv) > 1 else 7200
    trade_size = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    clob = create_clob()
    scanner = FastMarketScanner()
    detector = FastArbDetector(
        min_move_pct=0.003,
        min_seconds_elapsed=300,
        max_entry_price=0.60,
        min_edge_after_fees=0.05,
    )
    trader = LiveTrader(clob=clob, max_per_trade=trade_size)

    tick_count = 0
    signal_count = 0
    last_period_ts = 0
    last_status_log = 0

    def on_tick(asset, price, ts):
        nonlocal tick_count, signal_count
        tick_count += 1
        now = datetime.now(timezone.utc)
        for market in scanner._current_markets.values():
            if market.asset != asset:
                continue
            signal = detector.detect(market, price, now)
            if signal:
                signal_count += 1
                trader.execute(signal)

    ws_feed = BinanceWSFeed(on_price=on_tick)

    print(f"[{datetime.now(timezone.utc).isoformat()}] LIVE arb engine starting")
    print(f"  Duration: {duration}s | Trade size: ${trade_size:.0f}")
    print(f"  REAL MONEY - orders will be placed on Polymarket")
    print()

    ws_task = asyncio.create_task(ws_feed.run())

    for _ in range(50):
        if ws_feed.get("BTC"):
            break
        await asyncio.sleep(0.1)

    logger.info("live_ws_ready", assets=len(ws_feed.prices))

    start = time.time()
    try:
        while time.time() - start < duration:
            now_ts = int(time.time())
            current_period = int(now_ts // 900) * 900

            if current_period != last_period_ts:
                markets = await scanner.scan_current_period()
                prices = ws_feed.prices
                for coin in COINS:
                    asset = COIN_TO_ASSET[coin]
                    if asset in prices:
                        detector.record_start_price(asset, prices[asset], current_period)
                last_period_ts = current_period
                logger.info("live_new_period", period=current_period, markets=len(markets))

            trader.cleanup_expired()

            if time.time() - last_status_log > 60:
                prices = ws_feed.prices
                logger.info(
                    "live_cycle",
                    invested=f"${trader.total_invested:.2f}",
                    trades=len(trader.trades),
                    open=len(trader.open_orders),
                    ticks=tick_count,
                    signals=signal_count,
                    prices=" | ".join(f"{a}=${p:,.2f}" for a, p in sorted(prices.items()) if a in ("BTC", "ETH", "SOL")),
                )
                last_status_log = time.time()

            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\nStopping...")

    ws_feed.stop()
    ws_task.cancel()
    await scanner.close()

    result = trader.summary()
    logger.info("live_arb_stopped", **result)

    print(f"\n{'='*60}")
    print("LIVE ARB RESULTS")
    print(f"{'='*60}")
    for k, v in result.items():
        if k != "trades":
            print(f"  {k}: {v}")
    print(f"{'='*60}")

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **result}) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
