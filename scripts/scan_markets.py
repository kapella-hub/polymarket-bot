#!/usr/bin/env python3
"""Scan Polymarket for weather, crypto threshold, and high-opportunity markets."""

import asyncio
import httpx
import json
import sys

async def main():
    async with httpx.AsyncClient(timeout=15) as client:
        # 1. Search for weather/temperature events
        print("=" * 60)
        print("WEATHER / TEMPERATURE MARKETS")
        print("=" * 60)
        for term in ["high-temperature", "temperature", "nyc", "london-weather",
                      "weather", "degrees", "forecast-high"]:
            try:
                resp = await client.get("https://gamma-api.polymarket.com/events", params={
                    "slug_contains": term, "closed": "false", "limit": 5,
                })
                events = resp.json()
                for ev in events:
                    title = ev.get("title", "")
                    slug = ev.get("slug", "")
                    markets = ev.get("markets", [])
                    if not markets:
                        continue
                    # Skip non-weather
                    if not any(w in title.lower() for w in ["temp", "weather", "high", "degrees", "forecast"]):
                        if not any(w in slug for w in ["temp", "weather", "high", "degrees"]):
                            continue
                    print(f"\n  EVENT: {title[:65]}")
                    print(f"  slug: {slug[:55]}")
                    print(f"  markets: {len(markets)}")
                    for m in markets[:5]:
                        q = m.get("question", "")[:55]
                        outcomes = m.get("outcomes", "")
                        prices = m.get("outcomePrices", "")
                        vol = float(m.get("volumeNum", 0))
                        end = m.get("endDate", "")[:16]
                        print(f"    Q: {q}")
                        print(f"      outcomes={outcomes}  prices={prices}  vol={vol:.0f}  end={end}")
            except Exception as e:
                pass
            await asyncio.sleep(0.3)

        # 2. Search by browsing active events with high volume
        print("\n" + "=" * 60)
        print("HIGH-VOLUME ACTIVE EVENTS (by volume)")
        print("=" * 60)
        try:
            resp = await client.get("https://gamma-api.polymarket.com/events", params={
                "closed": "false", "limit": 50, "order": "volume",
            })
            events = resp.json()
            # Group by category-like patterns
            for ev in events[:30]:
                title = ev.get("title", "")
                slug = ev.get("slug", "")
                markets = ev.get("markets", [])
                total_vol = sum(float(m.get("volumeNum", 0)) for m in markets)
                if total_vol < 100000:
                    continue
                end_dates = [m.get("endDate", "")[:10] for m in markets if m.get("endDate")]
                nearest_end = min(end_dates) if end_dates else "?"

                # Categorize
                cat = "other"
                tl = title.lower()
                sl = slug.lower()
                if any(w in tl or w in sl for w in ["temp", "weather", "high", "degree"]):
                    cat = "WEATHER"
                elif any(w in tl or w in sl for w in ["bitcoin", "btc", "eth", "crypto", "solana", "price"]):
                    cat = "CRYPTO"
                elif any(w in sl for w in ["updown", "15m"]):
                    cat = "15MIN"
                elif any(w in tl for w in ["NBA", "NFL", "NHL", "MLB", "Premier League"]):
                    cat = "SPORTS"

                if cat in ("WEATHER", "CRYPTO"):
                    print(f"\n  [{cat}] {title[:60]}")
                    print(f"    slug: {slug[:55]}")
                    print(f"    vol: ${total_vol:,.0f}  markets: {len(markets)}  nearest_end: {nearest_end}")
                    for m in markets[:3]:
                        q = m.get("question", "")[:55]
                        prices = m.get("outcomePrices", "")
                        bid = m.get("bestBid", "--")
                        ask = m.get("bestAsk", "--")
                        print(f"      {q}  bid={bid} ask={ask}")
        except Exception as e:
            print(f"  ERROR: {e}")

        # 3. Direct search for today's weather
        print("\n" + "=" * 60)
        print("SEARCHING: today's weather markets")
        print("=" * 60)
        for query in ["april 10 temperature", "april 10 high", "nyc high april",
                       "london temperature april", "weather april 10"]:
            try:
                resp = await client.get("https://gamma-api.polymarket.com/markets", params={
                    "closed": "false", "limit": 5, "active": "true",
                })
                # Can't really text-search via API, so let's try slug patterns
            except Exception:
                pass

        # 4. Try fetching known weather slug patterns
        print("\n" + "=" * 60)
        print("TRYING KNOWN WEATHER SLUG PATTERNS")
        print("=" * 60)
        import time
        from datetime import datetime, timezone, timedelta

        today = datetime.now(timezone.utc)
        for days_offset in range(0, 3):
            d = today + timedelta(days=days_offset)
            date_str = d.strftime("%Y-%m-%d")
            for city in ["nyc", "new-york-city", "london", "tokyo", "seoul",
                          "los-angeles", "chicago", "miami"]:
                for pattern in [
                    f"{city}-high-temperature-{date_str}",
                    f"{city}-temperature-{date_str}",
                    f"high-temperature-{city}-{date_str}",
                    f"{city}-weather-{date_str}",
                ]:
                    try:
                        resp = await client.get(
                            f"https://gamma-api.polymarket.com/markets",
                            params={"slug": pattern}
                        )
                        data = resp.json()
                        if data:
                            print(f"  FOUND: {pattern}")
                            for m in data[:1]:
                                print(f"    Q: {m.get('question', '')[:65]}")
                                print(f"    outcomes: {m.get('outcomes', '')}")
                                print(f"    prices: {m.get('outcomePrices', '')}")
                                print(f"    vol: {float(m.get('volumeNum', 0)):.0f}")
                    except Exception:
                        pass
                await asyncio.sleep(0.1)

asyncio.run(main())
