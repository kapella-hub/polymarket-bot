#!/usr/bin/env python3
"""Check CLOB order book state for 15-min crypto markets."""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds
from src.crypto_arb.fast_markets import FastMarketScanner


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


async def main():
    clob = create_clob()
    scanner = FastMarketScanner()

    now_ts = int(time.time())
    current_period = int(now_ts // 900) * 900
    elapsed = now_ts - current_period
    remaining = 900 - elapsed

    print(f"Period: {current_period} | Elapsed: {elapsed}s | Remaining: {remaining}s")
    print()

    markets = await scanner.scan_current_period()
    print(f"Markets found: {len(markets)}")
    print()

    for m in markets:
        print(f"=== {m.asset} ===")
        print(f"  Gamma: up={m.up_price:.3f} down={m.down_price:.3f} bid={m.best_bid} ask={m.best_ask}")

        for label, token_id in [("UP", m.up_token_id), ("DOWN", m.down_token_id)]:
            try:
                book = clob.get_order_book(token_id)
                asks = sorted([(float(a.price), float(a.size)) for a in book.asks])
                bids = sorted([(float(b.price), float(b.size)) for b in book.bids], reverse=True)

                cheap_asks = [(p, s) for p, s in asks if p < 0.75]
                print(f"  {label}: {len(bids)}b/{len(asks)}a | "
                      f"top_bid=${bids[0][0]:.2f}x{bids[0][1]:.0f} | "
                      f"top_ask=${asks[0][0]:.2f}x{asks[0][1]:.0f} | "
                      f"cheap(<$0.75)={len(cheap_asks)}"
                      if bids and asks else f"  {label}: EMPTY")
            except Exception as e:
                print(f"  {label}: ERROR {e}")
        print()

    # Check next period too
    print("--- NEXT PERIOD ---")
    next_markets = await scanner.scan_next_period()
    print(f"Next period markets: {len(next_markets)}")
    for m in next_markets[:3]:
        print(f"  {m.asset}: up={m.up_price:.3f} down={m.down_price:.3f}")
        try:
            book = clob.get_order_book(m.up_token_id)
            asks = sorted([(float(a.price), float(a.size)) for a in book.asks])
            bids = sorted([(float(b.price), float(b.size)) for b in book.bids], reverse=True)
            cheap = [(p, s) for p, s in asks if p < 0.75]
            print(f"    UP: {len(bids)}b/{len(asks)}a | cheap(<$0.75)={len(cheap)}")
            if cheap:
                print(f"    Cheap asks: {cheap[:3]}")
        except Exception as e:
            print(f"    ERROR: {e}")

    await scanner.close()


if __name__ == "__main__":
    asyncio.run(main())
