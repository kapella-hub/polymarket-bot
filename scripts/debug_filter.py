#!/usr/bin/env python3
"""Debug market filter funnel to see why 0 candidates pass."""
import asyncio, httpx, json
from datetime import datetime, timezone

async def main():
    now = datetime.now(timezone.utc)
    async with httpx.AsyncClient(timeout=20) as client:
        all_markets = []
        for offset in [0, 100, 200]:
            resp = await client.get("https://gamma-api.polymarket.com/markets", params={
                "closed": "false", "active": "true", "limit": 100, "offset": offset,
            })
            data = resp.json()
            all_markets.extend(data)
            if len(data) < 100:
                break
            await asyncio.sleep(0.3)

        total = len(all_markets)
        funnel = {"total": total, "has_prices": 0, "price_mid": 0, "vol_ok": 0,
                  "has_bid_ask": 0, "spread_ok": 0, "date_ok": 0, "has_tokens": 0}
        passing = []

        for m in all_markets:
            prices = m.get("outcomePrices", [])
            if isinstance(prices, str):
                try:
                    prices = json.loads(prices)
                except:
                    continue
            if not prices or len(prices) < 2:
                continue
            funnel["has_prices"] += 1

            yes_price = float(prices[0])
            if not (0.08 <= yes_price <= 0.92):
                continue
            funnel["price_mid"] += 1

            vol = float(m.get("volumeNum", 0))
            if vol < 15000:
                continue
            funnel["vol_ok"] += 1

            bb = m.get("bestBid")
            ba = m.get("bestAsk")
            if bb is None or ba is None:
                continue
            funnel["has_bid_ask"] += 1

            spread = float(ba) - float(bb)
            if spread > 0.15:
                continue
            funnel["spread_ok"] += 1

            end_str = m.get("endDate")
            if not end_str:
                continue
            try:
                end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                days = (end - now).total_seconds() / 86400
            except:
                continue
            if not (0.5 <= days <= 30):
                continue
            funnel["date_ok"] += 1

            tokens = m.get("clobTokenIds", [])
            if isinstance(tokens, str):
                try:
                    tokens = json.loads(tokens)
                except:
                    continue
            if len(tokens) < 2:
                continue
            funnel["has_tokens"] += 1

            passing.append({
                "q": m.get("question", "")[:60],
                "yes": yes_price,
                "vol": vol,
                "days": days,
                "spread": spread,
            })

        print(f"Filter funnel ({total} total markets):")
        for k, v in funnel.items():
            print(f"  {k}: {v}")

        print(f"\nPassing markets ({len(passing)}):")
        for p in passing[:15]:
            print(f"  {p['q']}")
            print(f"    yes=${p['yes']:.2f} vol=${p['vol']:,.0f} days={p['days']:.0f} spread={p['spread']:.3f}")

        # Also show WHY markets fail at the date filter
        if funnel["spread_ok"] > funnel["date_ok"]:
            print(f"\nMarkets failing DATE filter (spread_ok but not date_ok):")
            count = 0
            for m in all_markets:
                prices = m.get("outcomePrices", [])
                if isinstance(prices, str):
                    try: prices = json.loads(prices)
                    except: continue
                if not prices or len(prices) < 2: continue
                yes_price = float(prices[0])
                if not (0.08 <= yes_price <= 0.92): continue
                vol = float(m.get("volumeNum", 0))
                if vol < 15000: continue
                bb, ba = m.get("bestBid"), m.get("bestAsk")
                if bb is None or ba is None: continue
                if float(ba) - float(bb) > 0.15: continue

                end_str = m.get("endDate")
                if end_str:
                    try:
                        end = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                        days = (end - now).total_seconds() / 86400
                        if days < 0.5 or days > 30:
                            q = m.get("question", "")[:55]
                            print(f"  {q}  days={days:.0f}")
                            count += 1
                            if count >= 10:
                                break
                    except:
                        pass

asyncio.run(main())
