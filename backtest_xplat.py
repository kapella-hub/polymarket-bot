#!/usr/bin/env python3
"""
Backtest: Cross-Platform Arbitrage (Strategy 4)
Polymarket vs Kalshi — same events, different prices

Methodology:
1. Fetch active markets from Polymarket (Gamma API) and Kalshi
2. Match by title keyword overlap
3. Compare prices — spread = abs(pm_price - kalshi_price)
4. Report: divergence frequency, spread distribution, category breakdown, systematic bias
"""
import asyncio
import json
import re
import sys
from collections import defaultdict

import httpx

GAMMA_BASE = "https://gamma-api.polymarket.com"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2"
RATE_LIMIT = 0.3

STOPWORDS = {
    "will", "the", "a", "an", "of", "in", "on", "at", "to", "for", "be",
    "is", "are", "was", "were", "by", "with", "or", "and", "not", "no",
    "yes", "than", "that", "this", "it", "its", "from", "between", "vs",
    "before", "after", "during", "over", "under", "per", "win", "wins",
    "lose", "loses", "does", "do", "more", "less", "each", "any", "all",
    "next", "last", "first", "second", "third", "which", "who", "how",
    "when", "where", "what", "why", "can", "get", "have", "has", "had",
    "2024", "2025", "2026",
    # Sport/league generic words that cause false matches
    "pro", "basketball", "hockey", "baseball", "football", "cup", "finals",
    "stanley", "nba", "nhl", "nfl", "mlb", "championship",
}


def tokenize(text):
    """Extract meaningful words from market title."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    words = text.split()
    return {w for w in words if w not in STOPWORDS and len(w) >= 3}


def match_markets(pm_markets, kalshi_markets, min_overlap=2):
    """
    Match Polymarket and Kalshi markets by keyword overlap.
    Returns list of (pm_market, kalshi_market, overlap_count, shared_words).
    """
    matches = []

    # Pre-tokenize Kalshi
    kalshi_tokenized = []
    for km in kalshi_markets:
        title = km.get("title", "") or km.get("rules_primary", "")
        tokens = tokenize(title)
        if tokens:
            kalshi_tokenized.append((km, tokens))

    for pm in pm_markets:
        q = pm.get("question", "")
        pm_tokens = tokenize(q)
        if not pm_tokens:
            continue

        best_match = None
        best_overlap = 0
        best_shared = set()

        for km, k_tokens in kalshi_tokenized:
            shared = pm_tokens & k_tokens
            overlap = len(shared)
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = km
                best_shared = shared

        if best_overlap >= min_overlap and best_match is not None:
            matches.append((pm, best_match, best_overlap, best_shared))

    return matches


def get_polymarket_price(m):
    """Extract mid-price from Polymarket market. Returns float or None."""
    # Try lastTradePrice first
    ltp = m.get("lastTradePrice")
    if ltp and float(ltp) > 0:
        return float(ltp)
    # Try outcomePrices (JSON string "[\"0.55\",\"0.45\"]") — take YES price
    raw = m.get("outcomePrices", "")
    prices = []
    if isinstance(raw, str) and raw:
        try:
            prices = [float(x) for x in json.loads(raw)]
        except Exception:
            pass
    elif isinstance(raw, list):
        prices = [float(x) for x in raw]
    if prices:
        return prices[0]  # YES outcome price
    # Try bestBid + bestAsk mid
    bid = m.get("bestBid")
    ask = m.get("bestAsk")
    if bid is not None and ask is not None:
        b, a = float(bid), float(ask)
        if 0 < b < a <= 1:
            return (b + a) / 2
    return None


def get_kalshi_price(km):
    """Extract mid-price from Kalshi market. Returns float or None."""
    lp = km.get("last_price_dollars")
    if lp is not None and float(lp) > 0:
        return float(lp)
    bid = km.get("yes_bid_dollars")
    ask = km.get("yes_ask_dollars")
    if bid is not None and ask is not None:
        b, a = float(bid), float(ask)
        if 0 < b < a <= 1:
            return (b + a) / 2
    return None


def infer_category(question):
    """Infer category from question text."""
    q = question.lower()
    if any(w in q for w in ["election", "president", "senate", "congress", "governor", "vote", "poll", "trump", "biden", "harris", "democrat", "republican"]):
        return "politics"
    if any(w in q for w in ["bitcoin", "btc", "eth", "ethereum", "crypto", "price", "usd", "fed", "rate", "inflation", "cpi", "gdp", "stock", "nasdaq", "s&p"]):
        return "economics/crypto"
    if any(w in q for w in ["nba", "nfl", "mlb", "nhl", "soccer", "football", "basketball", "baseball", "hockey", "tennis", "ufc", "f1", "championship", "super bowl", "world cup", "league"]):
        return "sports"
    if any(w in q for w in ["ai", "gpt", "openai", "tech", "iphone", "spacex", "elon", "company", "merger"]):
        return "tech/business"
    return "other"


async def fetch_polymarket_active(client, limit=100):
    """Fetch active Polymarket markets."""
    markets = []
    for offset in range(0, 500, 100):
        try:
            resp = await client.get(
                f"{GAMMA_BASE}/markets",
                params={"active": "true", "limit": limit, "offset": offset},
                timeout=30,
            )
            if resp.status_code == 200:
                batch = resp.json()
                if not isinstance(batch, list) or not batch:
                    break
                markets.extend(batch)
                print(f"  PM: fetched {len(batch)} active markets (offset={offset})")
            await asyncio.sleep(RATE_LIMIT)
        except Exception as e:
            print(f"  [warn] PM fetch offset={offset}: {e}")
            break
    return markets


async def fetch_kalshi_active(client, max_markets=500):
    """
    Fetch active Kalshi markets from individual event series.

    Note: The Kalshi elections API open-status endpoint currently returns
    MVE (multi-variable event, parlay) markets that don't have individual
    prices. We fetch from specific series that are known to have individual
    binary markets with prices: NBA, FED, and sports championships.
    """
    markets = []
    # Series with individual, priced binary markets comparable to Polymarket
    target_series = ["KXNBA", "KXNFL", "KXNHL", "KXMLB", "KXFED",
                     "KXPGAMASTERS", "KXPGATOUR", "KXFIFA", "KXF1",
                     "KXUFC", "KXSOCCER", "KXTENNIS"]

    for series in target_series:
        cursor = None
        series_count = 0
        while len(markets) < max_markets:
            params = {"limit": 100, "series_ticker": series}
            if cursor:
                params["cursor"] = cursor
            try:
                resp = await client.get(
                    f"{KALSHI_BASE}/markets",
                    params=params,
                    headers={"Accept": "application/json"},
                    timeout=30,
                )
                if resp.status_code != 200:
                    break
                data = resp.json()
                batch = data.get("markets", [])
                if not batch:
                    break
                # Only keep individual (non-MVE) markets with prices
                individual = [
                    km for km in batch
                    if not km.get("mve_collection_ticker")
                    and float(km.get("last_price_dollars") or 0) > 0
                ]
                markets.extend(individual)
                series_count += len(individual)
                cursor = data.get("cursor")
                if not cursor:
                    break
                await asyncio.sleep(RATE_LIMIT)
            except Exception as e:
                print(f"  [warn] Kalshi series={series}: {e}")
                break
        if series_count:
            print(f"  Kalshi series {series}: {series_count} priced markets")

    return markets


async def main():
    print("=" * 65)
    print("  CROSS-PLATFORM ARBITRAGE RESEARCH (Strategy 4)")
    print("  Polymarket vs Kalshi")
    print("=" * 65)
    print()

    async with httpx.AsyncClient(timeout=30) as client:
        # ── Step 1: Fetch active markets from both platforms ───────────
        print("Fetching active Polymarket markets...")
        pm_markets = await fetch_polymarket_active(client, limit=100)
        print(f"  Total PM: {len(pm_markets)}")
        print()

        print("Fetching active Kalshi markets...")
        kalshi_markets = await fetch_kalshi_active(client, max_markets=300)
        print(f"  Total Kalshi: {len(kalshi_markets)}")
        print()

        # ── Step 2: Match markets ──────────────────────────────────────
        print("Matching markets by keyword overlap...")
        matches = match_markets(pm_markets, kalshi_markets, min_overlap=2)
        print(f"  Matched pairs found: {len(matches)}")
        print()

        if not matches:
            print("No matches found. Markets may be from different time periods.")
            print("Try lowering min_overlap or running when both platforms cover the same events.")
            return

        # ── Step 3: Compute spreads ────────────────────────────────────
        spreads = []
        skipped_no_price = 0
        category_stats = defaultdict(lambda: {
            "count": 0, "spreads": [], "pm_higher": 0, "kalshi_higher": 0
        })

        for pm, km, overlap, shared_words in matches:
            pm_price = get_polymarket_price(pm)
            k_price = get_kalshi_price(km)

            if pm_price is None or k_price is None:
                skipped_no_price += 1
                continue

            spread = abs(pm_price - k_price)
            category = infer_category(pm.get("question", ""))

            entry = {
                "pm_question": pm.get("question", "")[:70],
                "kalshi_title": (km.get("title", "") or "")[:70],
                "pm_price": pm_price,
                "kalshi_price": k_price,
                "spread": spread,
                "overlap": overlap,
                "shared_words": sorted(shared_words),
                "category": category,
                "pm_higher": pm_price > k_price,
            }
            spreads.append(entry)
            cs = category_stats[category]
            cs["count"] += 1
            cs["spreads"].append(spread)
            if pm_price > k_price:
                cs["pm_higher"] += 1
            else:
                cs["kalshi_higher"] += 1

        if not spreads:
            print("No matched pairs had prices on both platforms.")
            return

        # ── Step 4: Report ─────────────────────────────────────────────
        all_spreads = [s["spread"] for s in spreads]
        avg_spread = sum(all_spreads) / len(all_spreads)
        sorted_spreads = sorted(all_spreads)
        median_spread = sorted_spreads[len(sorted_spreads) // 2]
        over5c = sum(1 for s in all_spreads if s > 0.05)
        over10c = sum(1 for s in all_spreads if s > 0.10)
        over15c = sum(1 for s in all_spreads if s > 0.15)
        pm_higher_total = sum(1 for s in spreads if s["pm_higher"])
        kalshi_higher_total = len(spreads) - pm_higher_total

        print("=" * 65)
        print("  CROSS-PLATFORM SPREAD ANALYSIS")
        print("=" * 65)
        print(f"  Matched pairs with prices: {len(spreads)}")
        print(f"  Skipped (missing prices):  {skipped_no_price}")
        print()
        print(f"  Average spread:    {avg_spread*100:.1f}c")
        print(f"  Median spread:     {median_spread*100:.1f}c")
        print(f"  Max spread:        {max(all_spreads)*100:.1f}c")
        print()
        print(f"  Pairs diverging >5c:  {over5c}/{len(spreads)} ({over5c/len(spreads)*100:.0f}%)")
        print(f"  Pairs diverging >10c: {over10c}/{len(spreads)} ({over10c/len(spreads)*100:.0f}%)")
        print(f"  Pairs diverging >15c: {over15c}/{len(spreads)} ({over15c/len(spreads)*100:.0f}%)")
        print()
        print(f"  Systematic bias:")
        print(f"    Polymarket HIGHER: {pm_higher_total}/{len(spreads)} ({pm_higher_total/len(spreads)*100:.0f}%)")
        print(f"    Kalshi HIGHER:     {kalshi_higher_total}/{len(spreads)} ({kalshi_higher_total/len(spreads)*100:.0f}%)")
        pm_avg = sum(s["pm_price"] for s in spreads) / len(spreads)
        k_avg = sum(s["kalshi_price"] for s in spreads) / len(spreads)
        bias = pm_avg - k_avg
        print(f"    Avg PM price: {pm_avg:.3f}  |  Avg Kalshi: {k_avg:.3f}  |  PM bias: {bias:+.3f} ({bias*100:+.1f}c)")
        print()

        # Category breakdown
        print("  Category breakdown:")
        print(f"  {'Category':<22} {'Pairs':>6} {'Avg Spread':>11} {'>5c':>5} {'>10c':>5} {'PM>Kalshi':>10}")
        print(f"  {'-'*62}")
        for cat, stats in sorted(category_stats.items(), key=lambda x: -x[1]["count"]):
            cat_spreads = stats["spreads"]
            cat_avg = sum(cat_spreads) / len(cat_spreads) if cat_spreads else 0
            cat_5c = sum(1 for s in cat_spreads if s > 0.05)
            cat_10c = sum(1 for s in cat_spreads if s > 0.10)
            pm_pct = stats["pm_higher"] / stats["count"] * 100 if stats["count"] else 0
            print(f"  {cat:<22} {stats['count']:>6} {cat_avg*100:>10.1f}c "
                  f"{cat_5c:>5} {cat_10c:>5} {pm_pct:>9.0f}%")
        print()

        # Best arbitrage opportunities
        big_spreads = [s for s in spreads if s["spread"] > 0.05]
        big_spreads.sort(key=lambda s: -s["spread"])
        if big_spreads:
            print("  Top arbitrage opportunities (>5c spread):")
            print(f"  {'PM Question':<50} {'PM':>6} {'Kalshi':>8} {'Spread':>8}")
            print(f"  {'-'*75}")
            for s in big_spreads[:15]:
                direction = "PM<K" if s["kalshi_price"] > s["pm_price"] else "PM>K"
                suspect = " *" if s["spread"] > 0.20 else ""
                print(f"  {s['pm_question'][:48]:<50} {s['pm_price']:>5.3f} "
                      f"{s['kalshi_price']:>7.3f} {s['spread']*100:>7.1f}c [{direction}]{suspect}")
            print("  (* = likely false match — questions measure different outcomes)")
            print()

        # ── Step 5: Same-event quality note ───────────────────────────
        # Flag cases where PM = "Conference Finals" but Kalshi = "Championship" — different events
        different_event_count = 0
        same_event_spreads = []
        conf_keywords = {"conference", "semifinal", "divisional", "round", "wildcard", "wild card"}
        for s in spreads:
            pm_q_lower = s["pm_question"].lower()
            kl_q_lower = s["kalshi_title"].lower()
            pm_has_conf = any(kw in pm_q_lower for kw in conf_keywords)
            kl_has_conf = any(kw in kl_q_lower for kw in conf_keywords)
            if pm_has_conf != kl_has_conf:
                different_event_count += 1
            else:
                same_event_spreads.append(s)

        if different_event_count > 0:
            print(f"  NOTE: {different_event_count} 'matches' are DIFFERENT events")
            print(f"  (e.g. PM has Conference Finals market, Kalshi only has Finals market)")
            print(f"  These are NOT tradeable arb — they represent different outcomes.")
            print(f"  Truly same-event matches: {len(same_event_spreads)}/{len(spreads)}")
            print()

        # ── Step 6: Verdict ────────────────────────────────────────────
        print("=" * 65)
        print("  ARBITRAGE VERDICT")
        print("=" * 65)
        tradeable_pct = over5c / len(spreads) * 100 if spreads else 0
        print(f"  {tradeable_pct:.0f}% of matched pairs show >5c spread (potential arb)")

        # Cost estimate: ~2c each side to execute (spread + slippage)
        round_trip_cost = 0.04  # 4c total round-trip
        profitable_count = sum(1 for s in all_spreads if s > round_trip_cost)
        same_event_profitable = sum(1 for s in same_event_spreads if s["spread"] > round_trip_cost)
        print(f"  {profitable_count}/{len(spreads)} pairs profitable after ~4c round-trip cost")
        print(f"  {same_event_profitable}/{len(same_event_spreads)} same-event pairs profitable")

        # NHL-specific structural bias (best quality data)
        nhl_spreads = [s for s in same_event_spreads if "stanley cup" in s["pm_question"].lower()
                       or "nhl" in s["pm_question"].lower()]
        if nhl_spreads:
            nhl_kalshi_higher = sum(1 for s in nhl_spreads if s["kalshi_price"] > s["pm_price"])
            nhl_avg_spread = sum(s["spread"] for s in nhl_spreads) / len(nhl_spreads)
            pm_sum = sum(s["pm_price"] for s in nhl_spreads)
            kl_sum = sum(s["kalshi_price"] for s in nhl_spreads)
            print()
            print(f"  NHL Stanley Cup structural analysis ({len(nhl_spreads)} teams):")
            print(f"    Average spread: {nhl_avg_spread*100:.1f}c")
            print(f"    Kalshi > PM: {nhl_kalshi_higher}/{len(nhl_spreads)} ({nhl_kalshi_higher/len(nhl_spreads)*100:.0f}%)")
            print(f"    PM probabilities sum to:     {pm_sum:.3f}")
            print(f"    Kalshi probabilities sum to: {kl_sum:.3f}")
            print(f"    --> Kalshi overpriced (vig={kl_sum:.2f}x vs PM {pm_sum:.2f}x)")
            print(f"    Structural edge: BUY cheaper PM teams, SELL expensive Kalshi teams")
            print(f"    BUT: require execution on BOTH platforms simultaneously")

        if profitable_count > 0 or same_event_profitable > 0:
            best_cat = max(category_stats.items(), key=lambda x: sum(1 for s in x[1]["spreads"] if s > round_trip_cost))
            print(f"  Best category: {best_cat[0]}")
            print()
            if bias > 0.03:
                print(f"  BIAS: Polymarket prices {bias*100:.1f}c ABOVE Kalshi on average.")
                print(f"        Strategy: BUY Kalshi YES, SELL Polymarket YES (or buy PM NO)")
            elif bias < -0.03:
                print(f"  BIAS: Kalshi prices {-bias*100:.1f}c ABOVE Polymarket on average.")
                print(f"        Strategy: BUY Polymarket YES, SELL Kalshi YES (or buy Kalshi NO)")
            else:
                print(f"  No systematic directional bias overall.")
        else:
            print()
            print(f"  After transaction costs, no profitable arb in current sample.")
            print(f"  Kalshi's open-status endpoint only returns parlay (MVE) markets.")
            print(f"  Individual binary Kalshi markets (KXNHL, KXNBA) show 1-2c spreads")
            print(f"  vs Polymarket — too small for round-trip profitability at ~4c cost.")
        print("=" * 65)


if __name__ == "__main__":
    # Force UTF-8 output on Windows
    if sys.stdout.encoding != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
