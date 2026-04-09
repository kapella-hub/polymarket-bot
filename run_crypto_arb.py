#!/usr/bin/env python3
"""Run the crypto latency arb engine in paper mode."""
import asyncio
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from src.crypto_arb.engine import CryptoArbEngine

LOG_FILE = Path(__file__).parent / "crypto_arb_paper.log"

async def main():
    engine = CryptoArbEngine(
        mode="paper",
        bankroll=1000.0,
        poll_interval=3.0,     # Binance every 3s
        scan_interval=60.0,    # Polymarket markets every 60s
    )

    duration = 3600  # 1 hour default
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])

    print(f"[{datetime.now(timezone.utc).isoformat()}] Crypto arb paper trading starting")
    print(f"  Duration: {duration}s ({duration/3600:.1f}h)")
    print(f"  Bankroll: $1,000")
    print(f"  Log: {LOG_FILE}")
    print()

    try:
        result = await engine.run(duration_seconds=duration)
    except KeyboardInterrupt:
        engine.stop()
        result = engine.trader.summary()

    # Write results to log
    with open(LOG_FILE, "a") as f:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_s": duration,
            **result,
            "trades": [
                {
                    "market": t.signal.market.question[:80],
                    "side": t.signal.side,
                    "entry_price": t.signal.entry_price,
                    "size_usd": t.size_usd,
                    "status": t.status,
                    "pnl": t.pnl,
                    "asset": t.signal.market.asset,
                    "threshold": t.signal.market.threshold,
                    "binance_at_entry": t.signal.binance_price,
                }
                for t in engine.trader.trades
            ],
        }
        f.write(json.dumps(entry) + "\n")

    print(f"\n{'='*60}")
    print("PAPER TRADING RESULTS")
    print(f"{'='*60}")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
