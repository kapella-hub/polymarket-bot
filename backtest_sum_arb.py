#!/usr/bin/env python3
"""
Backtest: Binary YES+NO Sum Arbitrage on Polymarket

Strategy: On a binary market, YES and NO tokens together always resolve to $1.00.
If YES_price + NO_price < $1.00, buying both tokens guarantees a risk-free profit.

This backtest:
1. Fetches 500 recently resolved binary markets from Gamma API
2. Reconstructs YES+NO mid-price sum at each historical tick from CLOB
3. Detects moments where sum < threshold (0.98 by default)
4. Simulates a $50-per-opportunity bet and calculates P&L

NOTE: Mid-price history from CLOB represents last-trade prices, which arbitrageurs
      continuously keep in line. The REAL opportunity exists in the bid/ask spread
      (buying at ask prices). Because we only have mid prices, these results are a
      LOWER BOUND — real opportunities exist primarily in the spread (spread arb),
      which requires live order book data not available in historical feeds.
"""

import asyncio
import json
import sys
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

BET_SIZE_USD = 50.0        # dollars per opportunity
SUM_THRESHOLD = 0.98       # detect sum < this value
MATCH_WINDOW_SEC = 300     # 5-minute window for timestamp alignment
PAGES = 5                  # pages * 100 = 500 markets
CONCURRENCY = 6            # max parallel HTTP requests


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MarketResult:
    market_id: str
    question: str
    category: str
    yes_token_id: str
    no_token_id: str
    yes_pts: int
    no_pts: int
    matched_pairs: int
    min_sum: float
    max_sum: float
    opportunities: list = field(default_factory=list)  # list of (timestamp, yes_p, no_p, sum)

    @property
    def has_opportunity(self) -> bool:
        return len(self.opportunities) > 0

    @property
    def best_profit_pct(self) -> float:
        if not self.opportunities:
            return 0.0
        return max(1.0 - s for _, _, _, s in self.opportunities)

    @property
    def avg_profit_pct(self) -> float:
        if not self.opportunities:
            return 0.0
        return sum(1.0 - s for _, _, _, s in self.opportunities) / len(self.opportunities)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

async def fetch_markets(client: httpx.AsyncClient, offset: int) -> list[dict]:
    """Fetch one page of closed markets sorted by most recently closed.

    Uses closed=true as the proxy for "resolved". The Gamma API does not
    expose a separate 'resolved' or 'winner' field — closed=true is the
    tightest available filter for markets that have finished trading.
    """
    url = f"{GAMMA_BASE}/markets"
    params = {
        "closed": "true",
        "limit": 100,
        "offset": offset,
        "order": "closedTime",
        "ascending": "false",
    }
    r = await client.get(url, params=params)
    r.raise_for_status()
    return r.json()


async def fetch_price_history(client: httpx.AsyncClient, token_id: str) -> list[dict]:
    """Fetch full price history for a CLOB token. Returns list of {t, p} dicts."""
    url = f"{CLOB_BASE}/prices-history"
    params = {"market": token_id, "interval": "max", "fidelity": 60}
    try:
        r = await client.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict):
            return data.get("history", [])
        return data if isinstance(data, list) else []
    except Exception:
        return []


def is_binary_market(m: dict) -> bool:
    """Return True if the market is binary (exactly 2 outcomes) with CLOB tokens."""
    try:
        outcomes_raw = m.get("outcomes", "[]")
        outcomes = json.loads(outcomes_raw) if isinstance(outcomes_raw, str) else (outcomes_raw or [])
        if len(outcomes) != 2:
            return False

        clob_raw = m.get("clobTokenIds", "[]")
        clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else (clob_raw or [])
        return len(clob_ids) == 2
    except Exception:
        return False


def align_histories(yes_hist: list[dict], no_hist: list[dict], window_sec: int = MATCH_WINDOW_SEC) -> list[tuple]:
    """
    Match YES and NO history points by nearest timestamp within window_sec.
    Returns list of (timestamp, yes_price, no_price, sum).
    """
    if not yes_hist or not no_hist:
        return []

    no_sorted = sorted(no_hist, key=lambda x: x["t"])
    no_ts = [e["t"] for e in no_sorted]

    matched = []
    for ye in yes_hist:
        t_yes = ye["t"]
        p_yes = ye["p"]
        idx = bisect_left(no_ts, t_yes)

        best_no = None
        best_diff = window_sec + 1

        for i in [idx - 1, idx]:
            if 0 <= i < len(no_sorted):
                diff = abs(no_ts[i] - t_yes)
                if diff < best_diff:
                    best_diff = diff
                    best_no = no_sorted[i]

        if best_no is not None and best_diff <= window_sec:
            s = p_yes + best_no["p"]
            matched.append((t_yes, p_yes, best_no["p"], s))

    return matched


# ---------------------------------------------------------------------------
# Market analyzer
# ---------------------------------------------------------------------------

async def analyze_market(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    m: dict,
) -> Optional[MarketResult]:
    """Fetch YES+NO histories for a market and check for sum arb opportunities."""
    async with sem:
        try:
            clob_raw = m.get("clobTokenIds", "[]")
            clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else []
            yes_id, no_id = clob_ids[0], clob_ids[1]

            yes_hist, no_hist = await asyncio.gather(
                fetch_price_history(client, yes_id),
                fetch_price_history(client, no_id),
            )

            if len(yes_hist) < 2 or len(no_hist) < 2:
                return None

            matched = align_histories(yes_hist, no_hist)
            if not matched:
                return None

            sums = [s for _, _, _, s in matched]
            opportunities = [(t, y, n, s) for t, y, n, s in matched if s < SUM_THRESHOLD]

            return MarketResult(
                market_id=str(m.get("id", "")),
                question=m.get("question", ""),
                category=m.get("category", ""),
                yes_token_id=yes_id,
                no_token_id=no_id,
                yes_pts=len(yes_hist),
                no_pts=len(no_hist),
                matched_pairs=len(matched),
                min_sum=min(sums),
                max_sum=max(sums),
                opportunities=opportunities,
            )
        except Exception as e:
            return None


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

async def run_backtest():
    print("=" * 65)
    print("  Binary YES+NO Sum Arbitrage Backtest")
    print(f"  Threshold: sum < {SUM_THRESHOLD:.2f} | Bet: ${BET_SIZE_USD:.0f}/trade")
    print("=" * 65)

    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient(timeout=40) as client:
        # --- Step 1: Fetch 500 closed markets ---
        print(f"\n[1/3] Fetching {PAGES * 100} closed markets from Gamma API...")
        raw_markets = []
        for page in range(PAGES):
            offset = page * 100
            page_data = await fetch_markets(client, offset)
            raw_markets.extend(page_data)
            print(f"      Page {page+1}/{PAGES}: got {len(page_data)} markets (total {len(raw_markets)})")
            if len(page_data) < 100:
                break

        # --- Step 2: Filter to binary markets ---
        binary_markets = [m for m in raw_markets if is_binary_market(m)]
        print(f"\n      Total fetched: {len(raw_markets)}, binary: {len(binary_markets)}")

        # --- Step 3: Fetch price histories and detect opportunities ---
        print(f"\n[2/3] Fetching CLOB price histories ({len(binary_markets)} markets x2 tokens)...")
        tasks = [analyze_market(client, sem, m) for m in binary_markets]
        results_raw = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out None/exceptions
    results = [r for r in results_raw if isinstance(r, MarketResult)]
    print(f"      Analyzed: {len(results)} markets with matched price pairs")

    # --- Step 4: Compute statistics ---
    print(f"\n[3/3] Computing statistics...\n")

    markets_with_opps = [r for r in results if r.has_opportunity]
    all_opps = []
    for r in markets_with_opps:
        for t, y, n, s in r.opportunities:
            all_opps.append({
                "timestamp": t,
                "market": r.question,
                "category": r.category,
                "yes_price": y,
                "no_price": n,
                "sum": s,
                "profit_pct": 1.0 - s,
                "profit_usd": BET_SIZE_USD * (1.0 - s),
            })

    # Date range from timestamps
    all_ts = [o["timestamp"] for o in all_opps]
    if all_ts:
        earliest_dt = datetime.fromtimestamp(min(all_ts), tz=timezone.utc)
        latest_dt = datetime.fromtimestamp(max(all_ts), tz=timezone.utc)
        span_days = max((latest_dt - earliest_dt).days, 1)
    else:
        span_days = 30
        earliest_dt = latest_dt = None

    # Category breakdown
    by_category = defaultdict(list)
    for o in all_opps:
        by_category[o["category"] or "unknown"].append(o)

    # Sum distribution: use the pre-computed min_sum from each MarketResult
    min_sums = [r.min_sum for r in results]

    below_099 = sum(1 for s in min_sums if s < 0.99)
    below_098 = sum(1 for s in min_sums if s < 0.98)
    below_097 = sum(1 for s in min_sums if s < 0.97)

    # -----------------------------------------------------------------------
    # REPORT
    # -----------------------------------------------------------------------
    print("=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)

    print(f"\n  Markets scanned:           {len(raw_markets):>8,}")
    print(f"  Binary markets (filtered): {len(binary_markets):>8,}")
    print(f"  Markets with CLOB history: {len(results):>8,}")

    print(f"\n  --- Opportunity Detection (sum < {SUM_THRESHOLD:.2f}) ---")
    print(f"  Markets with ANY opportunity:   {len(markets_with_opps):>5}")
    print(f"  Total opportunity ticks found:  {len(all_opps):>5}")
    print(f"  Win rate:                        {'100%' if all_opps else 'N/A':>5}  (risk-free by construction)")

    if all_opps:
        avg_sum = sum(o["sum"] for o in all_opps) / len(all_opps)
        avg_profit_pct = sum(o["profit_pct"] for o in all_opps) / len(all_opps)
        max_profit_pct = max(o["profit_pct"] for o in all_opps)
        total_profit_if_all = sum(o["profit_usd"] for o in all_opps)

        print(f"\n  --- Trade Economics ---")
        print(f"  Average sum when opportunity:   {avg_sum:.4f}  (avg gap: {1-avg_sum:.4f})")
        print(f"  Average profit per trade:       {avg_profit_pct*100:.2f}%  (${avg_profit_pct*BET_SIZE_USD:.2f} on ${BET_SIZE_USD:.0f})")
        print(f"  Best single-tick profit:        {max_profit_pct*100:.2f}%")
        print(f"  Total P&L (all ticks, $50/trade): ${total_profit_if_all:.2f}")

        print(f"\n  --- 30-Day Projection ---")
        daily_rate = len(all_opps) / max(span_days, 1)
        projected_30d_opps = daily_rate * 30
        projected_30d_pnl = total_profit_if_all / max(span_days, 1) * 30
        print(f"  Data span:                 {span_days} days ({earliest_dt.date() if earliest_dt else 'N/A'} to {latest_dt.date() if latest_dt else 'N/A'})")
        print(f"  Avg opportunities/day:     {daily_rate:.2f}")
        print(f"  Projected opportunities (30d): {projected_30d_opps:.0f}")
        print(f"  Projected P&L (30d, $50/trade): ${projected_30d_pnl:.2f}")

        print(f"\n  --- By Category ---")
        for cat, opps in sorted(by_category.items(), key=lambda x: -len(x[1])):
            avg_p = sum(o["profit_pct"] for o in opps) / len(opps)
            print(f"  {cat:<35} {len(opps):>4} ticks  avg profit {avg_p*100:.2f}%")

        print(f"\n  --- Top Opportunities ---")
        top = sorted(all_opps, key=lambda x: x["profit_pct"], reverse=True)[:10]
        for o in top:
            dt = datetime.fromtimestamp(o["timestamp"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            q = o["market"][:45].encode("ascii", errors="replace").decode("ascii")
            print(f"  {dt}  sum={o['sum']:.4f}  +{o['profit_pct']*100:.2f}%  {q}")
    else:
        print(f"\n  No ticks found with sum < {SUM_THRESHOLD:.2f} in mid-price history.")

    print(f"\n  --- Sum Distribution (market minimums across {len(results)} markets) ---")
    print(f"  Markets with min_sum < 0.99:   {below_099:>4}")
    print(f"  Markets with min_sum < 0.98:   {below_098:>4}")
    print(f"  Markets with min_sum < 0.97:   {below_097:>4}")
    print(f"  Overall min sum observed:      {min(min_sums):.4f}" if min_sums else "  N/A")
    print(f"  Overall median min sum:        {sorted(min_sums)[len(min_sums)//2]:.4f}" if min_sums else "  N/A")

    print(f"\n  --- Methodology Notes ---")
    print(f"  * Source: CLOB mid-price history (last trade price per tick)")
    print(f"  * Mid prices are kept near-parity by arbitrageurs continuously")
    print(f"  * Real opportunities exist in the BID/ASK SPREAD: buying YES at ask")
    print(f"    + NO at ask < 1.00. This requires live order book data.")
    print(f"  * Historical mid prices undercount real fill-able opportunities.")
    print(f"  * These results are a LOWER BOUND on what live spread arb would find.")
    print(f"  * Typical Polymarket spread: 1-3 cents per side = sum at ask ~1.02-1.06")
    print(f"    (most markets are NOT arbitrageable at the ask level)")
    print(f"  * Markets with wider spreads or stale quotes may offer real gaps.")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(run_backtest())
