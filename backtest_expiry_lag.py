#!/usr/bin/env python3
"""
Backtest: Near-Expiry Price Lag Strategy

Hypothesis: In the final 24 hours before resolution, Polymarket prices
often haven't caught up to publicly known outcomes. If a market is about
to resolve YES but is still trading at $0.75, that's a 25-cent edge.

Methodology:
- Fetch ~500 recently resolved binary markets from Gamma API
- For each, reconstruct the T-24h price using trade history from data-api
- Simulate buying the winning side at T-24h price
- Report: average entry price, # opportunities, avg P&L, category breakdown

API Notes:
- CLOB prices-history endpoint does NOT serve historical data (empty for all markets)
- data-api.polymarket.com/trades gives individual trades (newest-first, max offset 3000)
- We paginate trades until we find the last trade before T-24h for the winning token
"""

import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

import httpx

# ---- Config -----------------------------------------------------------------
GAMMA_BASE    = "https://gamma-api.polymarket.com"
DATA_API_BASE = "https://data-api.polymarket.com"
PAGES         = 5           # Gamma market pages (100 each = 500 markets)
GAMMA_OFFSET  = 35000       # Offset into closed markets (Feb-Mar 2025, has CLOB history)
MAX_ENTRY     = 0.90        # Only trade if price < $0.90
MIN_ENTRY     = 0.40        # Skip if price < $0.40 (likely noise or wrong token)
SLEEP_BETWEEN = 0.35        # Seconds between API calls
RESOLVE_THRESHOLD = 0.88    # outcomePrices threshold to call winner
MIN_VOLUME    = 3000        # Min market volume to include
MIN_DURATION_DAYS = 1.5     # Min market duration to have T-24h data
MAX_TRADE_PAGES = 30        # Max 100*30 = 3000 trades to paginate (API limit)

# ---- Helpers ----------------------------------------------------------------

def parse_json_field(raw) -> list:
    if not raw:
        return []
    try:
        if isinstance(raw, list):
            return raw
        return json.loads(raw)
    except Exception:
        return []


def infer_category(question: str, tags: list) -> str:
    tag_str = " ".join(str(t).lower() for t in tags)
    q = question.lower()
    combined = q + " " + tag_str

    if any(w in combined for w in ["nfl", "nba", "nhl", "mlb", "soccer", "football",
                                    "basketball", "match", "playoff", "super bowl",
                                    "world cup", "premier league", "champions league",
                                    "f1", "tennis", "ufc", "mma", "ncaa", "tournament",
                                    "baseball", "hockey", "esport", "counter-strike",
                                    "vs.", " vs ", "fight", "bout", "boxing"]):
        return "sports"
    if any(w in combined for w in ["bitcoin", "btc", "eth", "ethereum", "solana",
                                    "crypto", "price above", "price below", "token",
                                    "defi", "nft", "web3"]):
        return "crypto"
    if any(w in combined for w in ["election", "president", "congress", "senate",
                                    "vote", "democrat", "republican", "political",
                                    "legislation", "supreme court", "governor",
                                    "mayor", "prime minister"]):
        return "politics"
    if any(w in combined for w in ["oscar", "grammy", "emmy", "box office", "movie",
                                    "film", "celebrity", "award"]):
        return "entertainment"
    if any(w in combined for w in ["unemployment", "gdp", "inflation", "fed rate",
                                    "interest rate", "stock", "s&p", "nasdaq", "earnings"]):
        return "finance"
    return "other"


async def fetch_gamma_markets(
    client: httpx.AsyncClient,
    pages: int,
    page_size: int = 100,
    base_offset: int = 0,
) -> list:
    """Fetch resolved binary markets from Gamma API with pagination."""
    all_markets = []
    print(f"Fetching {pages * page_size} markets from Gamma (offset={base_offset})...")

    for page in range(pages):
        offset = base_offset + page * page_size
        try:
            resp = await client.get(
                f"{GAMMA_BASE}/markets",
                params={"closed": "true", "limit": page_size, "offset": offset},
                timeout=20,
            )
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            all_markets.extend(batch)
            print(f"  Page {page+1}/{pages}: {len(batch)} markets (total: {len(all_markets)})")
        except Exception as e:
            print(f"  ERROR page {page+1}: {e}")
        await asyncio.sleep(0.3)

    return all_markets


async def get_t24h_price(
    client: httpx.AsyncClient,
    condition_id: str,
    end_ts: int,
    winner_idx: int,
) -> Optional[float]:
    """
    Find the winning token's last trade price at or before T-24h.
    Paginates data-api/trades (newest-first) until we reach pre-T24h trades.
    Returns None if no trade found in the window.
    """
    t24 = end_ts - 24 * 3600
    t48 = end_ts - 48 * 3600  # earliest reasonable look-back

    last_winning_price_before_t24: Optional[float] = None
    last_winning_ts_before_t24: int = 0

    for page in range(MAX_TRADE_PAGES):
        offset = page * 100
        try:
            resp = await client.get(
                f"{DATA_API_BASE}/trades",
                params={"market": condition_id, "limit": 100, "offset": offset},
                timeout=15,
            )
            resp.raise_for_status()
            trades = resp.json()
        except Exception:
            break

        if not isinstance(trades, list) or not trades:
            break

        # Check for API error (offset exceeded)
        if isinstance(trades, dict) and trades.get("error"):
            break

        for trade in trades:
            ts = trade.get("timestamp", 0)
            outcome_idx = trade.get("outcomeIndex")
            price = trade.get("price")

            if outcome_idx is None or price is None:
                continue

            # Only look at the winning side's trades
            if int(outcome_idx) != winner_idx:
                continue

            if ts <= t24 and ts > last_winning_ts_before_t24:
                last_winning_price_before_t24 = float(price)
                last_winning_ts_before_t24 = ts

        # If oldest trade in this page is before T-48h, we've gone far enough
        oldest_ts = min(t.get("timestamp", 0) for t in trades)
        if oldest_ts < t48:
            break

        await asyncio.sleep(0.1)

    return last_winning_price_before_t24


# ---- Main Backtest ----------------------------------------------------------

async def main():
    results = []
    skipped = 0
    processed = 0
    no_history = 0

    async with httpx.AsyncClient(timeout=20) as client:
        # Fetch resolved markets
        raw_markets = await fetch_gamma_markets(
            client, PAGES, page_size=100, base_offset=GAMMA_OFFSET
        )
        print(f"\nTotal raw markets: {len(raw_markets)}")

        # Filter to binary, resolved, sufficient volume
        binary_markets = []
        for m in raw_markets:
            outcomes = parse_json_field(m.get("outcomes", "[]"))
            if len(outcomes) != 2:
                continue
            op = parse_json_field(m.get("outcomePrices", "[]"))
            if len(op) < 2:
                continue
            if not m.get("conditionId"):
                continue
            end_date_str = m.get("endDate", "")
            if not end_date_str:
                continue
            created_str = m.get("createdAt", "")
            if not created_str:
                continue
            vol = float(m.get("volumeNum", 0) or 0)
            if vol < MIN_VOLUME:
                continue

            # Check market duration
            try:
                if end_date_str.endswith("Z"):
                    end_date_str = end_date_str[:-1] + "+00:00"
                end_ts = int(datetime.fromisoformat(end_date_str).timestamp())

                if created_str.endswith("Z"):
                    created_str = created_str[:-1] + "+00:00"
                created_ts = int(datetime.fromisoformat(created_str).timestamp())
                duration_days = (end_ts - created_ts) / 86400
            except Exception:
                continue

            if duration_days < MIN_DURATION_DAYS:
                continue

            binary_markets.append(m)

        print(f"Binary markets (vol>={MIN_VOLUME}, dur>={MIN_DURATION_DAYS}d): {len(binary_markets)}")
        print("\nLooking up T-24h prices via trade history...")
        print("-" * 60)

        for idx, m in enumerate(binary_markets):
            if idx > 0 and idx % 20 == 0:
                opp_so_far = sum(1 for r in results if r["tradeable"])
                print(f"  [{idx}/{len(binary_markets)}] processed={processed} skip={skipped} no_hist={no_history} opps={opp_so_far}")

            question     = m.get("question", "")
            end_date_str = m.get("endDate", "")
            op           = parse_json_field(m.get("outcomePrices", "[]"))
            cid          = m.get("conditionId", "")
            tags         = m.get("tags", []) or []
            category     = infer_category(question, tags)

            # Parse end timestamp
            try:
                if end_date_str.endswith("Z"):
                    end_date_str = end_date_str[:-1] + "+00:00"
                end_ts = int(datetime.fromisoformat(end_date_str).timestamp())
            except Exception:
                skipped += 1
                continue

            # Determine winner
            try:
                yes_p = float(op[0])
                no_p  = float(op[1])
            except (ValueError, IndexError):
                skipped += 1
                continue

            if yes_p >= RESOLVE_THRESHOLD:
                winner_idx = 0
            elif no_p >= RESOLVE_THRESHOLD:
                winner_idx = 1
            else:
                skipped += 1
                continue

            # Get T-24h price via trades
            price_t24 = await get_t24h_price(client, cid, end_ts, winner_idx)
            await asyncio.sleep(SLEEP_BETWEEN)
            processed += 1

            if price_t24 is None:
                no_history += 1
                continue

            tradeable = MIN_ENTRY <= price_t24 < MAX_ENTRY
            pnl = (1.0 - price_t24) if tradeable else None

            results.append({
                "question":    question[:80],
                "category":    category,
                "winner_side": "YES" if winner_idx == 0 else "NO",
                "price_t24":   price_t24,
                "end_ts":      end_ts,
                "tradeable":   tradeable,
                "pnl":         pnl,
                "volume":      float(m.get("volumeNum", 0) or 0),
            })

        print(f"\nDone. processed={processed}  skipped={skipped}  no_history={no_history}  results={len(results)}")

    # ---- Analysis -----------------------------------------------------------
    tradeable     = [r for r in results if r["tradeable"]]
    all_with_data = [r for r in results if r.get("price_t24") is not None]

    print("\n" + "=" * 70)
    print("NEAR-EXPIRY PRICE LAG BACKTEST RESULTS")
    print("=" * 70)

    print(f"\nMarkets with T-24h data:    {len(all_with_data)}")
    print(f"Tradeable (${MIN_ENTRY:.2f}-${MAX_ENTRY:.2f}): {len(tradeable)}")

    if not tradeable:
        print("\nNo tradeable opportunities found in this market sample.")
        print("(Try adjusting GAMMA_OFFSET to sample different market cohorts.)")
        return

    prices = [r["price_t24"] for r in tradeable]
    avg_p  = sum(prices) / len(prices)
    b1 = sum(1 for p in prices if 0.40 <= p < 0.50)
    b2 = sum(1 for p in prices if 0.50 <= p < 0.60)
    b3 = sum(1 for p in prices if 0.60 <= p < 0.70)
    b4 = sum(1 for p in prices if 0.70 <= p < 0.80)
    b5 = sum(1 for p in prices if 0.80 <= p < 0.90)
    n  = len(tradeable)

    print(f"\n-- Price Distribution at T-24h (winning side) --")
    print(f"  Average:        ${avg_p:.4f}")
    print(f"  Min / Max:      ${min(prices):.4f} / ${max(prices):.4f}")
    print(f"  $0.40-0.50:     {b1:>4} ({100*b1/n:.1f}%)")
    print(f"  $0.50-0.60:     {b2:>4} ({100*b2/n:.1f}%)")
    print(f"  $0.60-0.70:     {b3:>4} ({100*b3/n:.1f}%)")
    print(f"  $0.70-0.80:     {b4:>4} ({100*b4/n:.1f}%)")
    print(f"  $0.80-0.90:     {b5:>4} ({100*b5/n:.1f}%)")

    pnls    = [r["pnl"] for r in tradeable if r["pnl"] is not None]
    avg_pnl = sum(pnls) / len(pnls) if pnls else 0
    tot_pnl = sum(pnls)

    print(f"\n-- P&L (buy winner at T-24h, exit at $1.00) --")
    print(f"  # Opportunities:          {n}")
    print(f"  Avg P&L per $1 bet:       ${avg_pnl:.4f} ({100*avg_pnl:.1f}%)")
    print(f"  Total P&L ($1 flat):      ${tot_pnl:.2f}")

    def apnl(subset):
        ps = [r["pnl"] for r in subset if r["pnl"] is not None]
        return sum(ps) / len(ps) if ps else 0.0

    t_lo = [r for r in tradeable if r["price_t24"] < 0.70]
    t_mid = [r for r in tradeable if 0.70 <= r["price_t24"] < 0.80]
    t_hi = [r for r in tradeable if r["price_t24"] >= 0.80]

    print(f"\n-- Threshold Sensitivity --")
    print(f"  All  (<$0.90):    {n:>4} trades  avg_pnl=${apnl(tradeable):.4f}")
    print(f"  <$0.70 (big lag): {len(t_lo):>4} trades  avg_pnl=${apnl(t_lo):.4f}")
    print(f"  $0.70-$0.80:      {len(t_mid):>4} trades  avg_pnl=${apnl(t_mid):.4f}")
    print(f"  $0.80-$0.90:      {len(t_hi):>4} trades  avg_pnl=${apnl(t_hi):.4f}")

    # Category breakdown
    cat_data = defaultdict(list)
    for r in tradeable:
        cat_data[r["category"]].append(r)

    print(f"\n-- Category Breakdown --")
    sorted_cats = sorted(cat_data.items(), key=lambda x: -len(x[1]))
    for cat, items in sorted_cats:
        cp = [r["pnl"] for r in items if r["pnl"] is not None]
        cavg = sum(cp) / len(cp) if cp else 0
        cavg_price = sum(r["price_t24"] for r in items) / len(items)
        print(f"  {cat:<16} {len(items):>4} trades  avg_entry=${cavg_price:.3f}  avg_pnl=${cavg:.4f}")

    # Most-lagged category (lowest avg T-24h price = most edge remaining)
    cat_avg_prices = {
        cat: sum(r["price_t24"] for r in items) / len(items)
        for cat, items in cat_data.items() if len(items) >= 3
    }

    if cat_avg_prices:
        print(f"\n-- Lag Ranking (lower T-24h price = more edge) --")
        for cat, avg_p_cat in sorted(cat_avg_prices.items(), key=lambda x: x[1]):
            print(f"  {cat:<16}  avg T-24h ${avg_p_cat:.4f}  (lag = ${1.0-avg_p_cat:.4f})")

    # Top 10 biggest lags
    top10 = sorted(tradeable, key=lambda r: r.get("pnl", 0) or 0, reverse=True)[:10]
    print(f"\n-- Top 10 Best Lag Opportunities --")
    for r in top10:
        dt = datetime.fromtimestamp(r["end_ts"], tz=timezone.utc).strftime("%Y-%m-%d")
        winner = r["winner_side"]
        print(f"  [{r['category']:<12}] [{winner}] {r['question'][:52]:<52}  T24=${r['price_t24']:.3f}  pnl=${r['pnl']:.3f}  ({dt})")

    # "Signal check": what fraction of markets at $0.60-0.90 actually resolved YES?
    bracket = [r for r in all_with_data if 0.60 <= r.get("price_t24", 0) < 0.90]
    if bracket:
        yes_w = sum(1 for r in bracket if r["winner_side"] == "YES")
        no_w  = sum(1 for r in bracket if r["winner_side"] == "NO")
        print(f"\n-- Price Information Content (do T-24h prices predict outcome?) --")
        print(f"  Markets priced $0.60-$0.90 at T-24h: {len(bracket)}")
        print(f"    Winner was YES: {yes_w} ({100*yes_w/len(bracket):.1f}%)")
        print(f"    Winner was NO:  {no_w} ({100*no_w/len(bracket):.1f}%)")
        print(f"  (Note: we query the WINNING token's price, so 100% resolve in winning direction.")
        print(f"   The ratio tells you how often YES vs NO was the laggy side.)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Sample: {len(binary_markets)} binary markets (offset={GAMMA_OFFSET})")
    print(f"  Markets with T-24h price data:     {len(all_with_data)}")
    print(f"  Tradeable opportunities (<${MAX_ENTRY:.2f}):  {len(tradeable)}")
    print(f"  Avg last-24h price (winning side): ${avg_p:.4f}")
    print(f"  Avg P&L per $1 flat bet:           ${avg_pnl:.4f} ({100*avg_pnl:.1f}%)")
    print(f"  Total P&L ($1 on each):            ${tot_pnl:.2f}")
    print(f"  NOTE: Theoretical P&L with perfect outcome foresight.")
    print(f"        Without foresight, blindly buying YES at $0.60-$0.90")
    print(f"        loses money -- market price has real information.")
    if cat_avg_prices:
        most_lagged = min(cat_avg_prices, key=lambda c: cat_avg_prices[c])
        print(f"  Most-lagged category:              {most_lagged} (avg T-24h ${cat_avg_prices[most_lagged]:.4f})")


if __name__ == "__main__":
    asyncio.run(main())
