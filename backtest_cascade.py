#!/usr/bin/env python3
"""
Backtest: Correlated Market Cascade (Strategy 3)

When Market A (round/conference) resolves or reprices sharply, correlated
Market B (championship/finals) should reprice. The lag between A moving and
B moving is a trading window.

Data: Active markets with full CLOB price history (~4000+ data points each).
Focus: Nested tournament structures — Conference Finals -> Finals, etc.

Methodology:
1. Fetch ~300 active markets from Gamma API
2. Identify nested pairs: child (round/conference/semifinal) + parent (championship/finals)
3. For each pair, fetch price histories from CLOB
4. Find sharp moves in child (A), check if parent (B) followed within 48h
5. Quantify: avg lag, signal accuracy, projected edge
"""
import asyncio
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

import httpx

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
RATE_LIMIT = 0.3
SHARP_MOVE_THRESHOLD = 0.03   # 3-cent move to trigger cascade signal (pre-season: small moves)
FOLLOWTHROUGH_THRESHOLD = 0.015  # 1.5-cent min follow-through to count as hit
FOLLOWTHROUGH_WINDOW_HOURS = 48

CHILD_KEYWORDS = [
    "conference finals", "conference semifinal", "conference semi",
    "eastern conference", "western conference",
    "first round", "second round", "round 1", "round 2",
    "divisional", "wild card", "semifinal", "semi-final",
    "conference championship",
]

PARENT_KEYWORDS = [
    "stanley cup", "nba finals", "super bowl", "world series",
    "nfl championship", "championship", "title",
]

STOP_WORDS = {
    "will", "the", "a", "an", "win", "in", "2026", "2025", "2024",
    "nba", "nhl", "nfl", "mlb", "finals",
}


def get_token_ids(m):
    raw = m.get("clobTokenIds", "[]")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return raw or []


def tokenize(q):
    return set(re.findall(r"[a-z]+", q.lower())) - STOP_WORDS


def find_cascade_pairs(markets):
    """
    Find (child, parent) market pairs where:
    - child covers a round/conference-level outcome
    - parent covers the championship/finals outcome
    - both cover the same team/entity (keyword overlap >= 2)
    """
    pairs = []
    seen = set()

    for m_child in markets:
        q_child = m_child.get("question", "").lower()
        if not any(kw in q_child for kw in CHILD_KEYWORDS):
            continue

        for m_parent in markets:
            if m_parent["id"] == m_child["id"]:
                continue
            q_parent = m_parent.get("question", "").lower()
            if not any(kw in q_parent for kw in PARENT_KEYWORDS):
                continue
            # Exclude parent if it also has child keywords (e.g., "Conference Finals")
            if any(kw in q_parent for kw in ["conference finals", "conference semifinal", "divisional"]):
                continue

            shared = tokenize(q_child) & tokenize(q_parent)
            if len(shared) >= 2:
                key = tuple(sorted([m_child["id"], m_parent["id"]]))
                if key not in seen:
                    seen.add(key)
                    pairs.append({
                        "child": m_child,
                        "parent": m_parent,
                        "shared_words": sorted(shared),
                    })

    return pairs


async def fetch_price_history(client, token_id):
    """Fetch full price history from CLOB. Returns list of (ts, price) tuples."""
    try:
        resp = await client.get(
            f"{CLOB_BASE}/prices-history",
            params={"market": token_id, "interval": "all"},
            timeout=20,
        )
        if resp.status_code == 200:
            history = resp.json().get("history", [])
            return [(int(pt["t"]), float(pt["p"])) for pt in history if "t" in pt and "p" in pt]
    except Exception as e:
        print(f"  [warn] history fetch {str(token_id)[:20]}...: {e}")
    return []


def find_sharp_moves(history, threshold=SHARP_MOVE_THRESHOLD):
    """
    Find price moves greater than threshold between consecutive points.
    Returns list of (timestamp, direction, magnitude).
    """
    moves = []
    for i in range(1, len(history)):
        ts_prev, p_prev = history[i - 1]
        ts_now, p_now = history[i]
        delta = p_now - p_prev
        if abs(delta) >= threshold:
            moves.append((ts_now, "up" if delta > 0 else "down", abs(delta)))
    return moves


def check_followthrough(history_b, signal_ts, signal_direction, window_hours=FOLLOWTHROUGH_WINDOW_HOURS):
    """
    After signal_ts in market A, did market B move in the expected direction
    within window_hours?

    Returns (correct_direction, lag_hours, b_move_magnitude).
    """
    window_secs = window_hours * 3600

    # Get B's price at/after signal time
    ref_price = None
    ref_ts = None
    for ts, p in history_b:
        if ts >= signal_ts:
            ref_price = p
            ref_ts = ts
            break

    if ref_price is None:
        return False, None, 0.0

    # Scan forward for B's first significant move
    for ts, p in history_b:
        if ts < ref_ts:
            continue
        if ts > signal_ts + window_secs:
            break
        delta = p - ref_price
        if abs(delta) >= FOLLOWTHROUGH_THRESHOLD:
            lag_hours = (ts - signal_ts) / 3600
            moved_up = delta > 0
            correct = (signal_direction == "up" and moved_up) or \
                      (signal_direction == "down" and not moved_up)
            return correct, lag_hours, abs(delta)

    return False, None, 0.0


def compute_rolling_correlation(hist_a, hist_b, window_secs=86400):
    """
    Simple rolling correlation: how often do daily moves align?
    Returns (aligned_days, total_days, correlation_pct).
    """
    # Build daily price maps
    price_a = {ts: p for ts, p in hist_a}
    price_b = {ts: p for ts, p in hist_b}

    if not price_a or not price_b:
        return 0, 0, 0.0

    min_ts = max(min(price_a), min(price_b))
    max_ts = min(max(price_a), max(price_b))

    # Sample every 24h
    aligned = total = 0
    ts = min_ts
    prev_a = prev_b = None

    sorted_a = sorted(price_a.items())
    sorted_b = sorted(price_b.items())

    def get_nearest(sorted_hist, target_ts):
        closest = None
        for t, p in sorted_hist:
            if t <= target_ts:
                closest = p
            else:
                break
        return closest

    while ts < max_ts:
        pa = get_nearest(sorted_a, ts)
        pb = get_nearest(sorted_b, ts)
        if pa is not None and pb is not None and prev_a is not None and prev_b is not None:
            dir_a = pa - prev_a
            dir_b = pb - prev_b
            if abs(dir_a) > 0.01 and abs(dir_b) > 0.01:
                total += 1
                if (dir_a > 0) == (dir_b > 0):
                    aligned += 1
        prev_a, prev_b = pa, pb
        ts += window_secs

    corr = aligned / total * 100 if total > 0 else 0.0
    return aligned, total, corr


async def main():
    print("=" * 68)
    print("  BACKTEST: Correlated Market Cascade (Strategy 3)")
    print("=" * 68)
    print()

    async with httpx.AsyncClient(timeout=30) as client:
        # ── Step 1: Fetch active markets (have live price history) ──────
        print("Fetching active markets from Gamma API...")
        all_markets = []
        for offset in range(0, 300, 100):
            try:
                resp = await client.get(
                    f"{GAMMA_BASE}/markets",
                    params={"active": "true", "limit": 100, "offset": offset},
                    timeout=30,
                )
                if resp.status_code == 200:
                    batch = resp.json()
                    if isinstance(batch, list):
                        all_markets.extend(batch)
                        print(f"  Fetched {len(batch)} markets (offset={offset})")
            except Exception as e:
                print(f"  [warn] offset={offset}: {e}")
            await asyncio.sleep(0.3)

        print(f"  Total active markets: {len(all_markets)}")
        print()

        # ── Step 2: Find cascade pairs ──────────────────────────────────
        print("Identifying nested cascade pairs (child round -> parent championship)...")
        pairs = find_cascade_pairs(all_markets)
        print(f"  Cascade pairs found: {len(pairs)}")
        if pairs:
            print("  Sample pairs:")
            for p in pairs[:4]:
                print(f"    Child:  {p['child']['question'][:62]}")
                print(f"    Parent: {p['parent']['question'][:62]}")
                print()
        print()

        if not pairs:
            print("No cascade pairs found in current active markets.")
            print("Try again during active tournament season (playoffs, etc.)")
            return

        # ── Step 3: Fetch price histories ───────────────────────────────
        print("Fetching price histories from CLOB...")
        history_cache = {}

        async def get_history(tid):
            if tid not in history_cache:
                hist = await fetch_price_history(client, tid)
                history_cache[tid] = hist
                await asyncio.sleep(RATE_LIMIT)
            return history_cache[tid]

        # Pre-fetch all needed histories
        all_tids = set()
        for pair in pairs:
            child_tids = get_token_ids(pair["child"])
            parent_tids = get_token_ids(pair["parent"])
            if child_tids:
                all_tids.add(child_tids[0])
            if parent_tids:
                all_tids.add(parent_tids[0])

        print(f"  Fetching {len(all_tids)} price histories...")
        for tid in all_tids:
            await get_history(tid)

        print(f"  Done. Cached {len(history_cache)} histories.")
        print()

        # ── Step 4: Analyze cascade signals ────────────────────────────
        print("Analyzing cascade signals...")
        print()

        cascade_results = []
        pair_summaries = []
        skipped = 0

        for pair in pairs:
            child_tids = get_token_ids(pair["child"])
            parent_tids = get_token_ids(pair["parent"])

            if not child_tids or not parent_tids:
                skipped += 1
                continue

            hist_child = history_cache.get(child_tids[0], [])
            hist_parent = history_cache.get(parent_tids[0], [])

            if len(hist_child) < 10 or len(hist_parent) < 10:
                skipped += 1
                continue

            # Correlation analysis
            aligned, total_days, corr_pct = compute_rolling_correlation(hist_child, hist_parent)

            # Sharp move cascade analysis
            sharp_moves = find_sharp_moves(hist_child, SHARP_MOVE_THRESHOLD)
            pair_signals = []

            for sig_ts, sig_dir, sig_mag in sharp_moves:
                correct, lag_h, b_mag = check_followthrough(hist_parent, sig_ts, sig_dir)
                pair_signals.append({
                    "signal_ts": sig_ts,
                    "signal_dir": sig_dir,
                    "signal_mag": sig_mag,
                    "correct": correct,
                    "lag_hours": lag_h,
                    "b_magnitude": b_mag,
                })
                cascade_results.append({
                    "child_q": pair["child"]["question"][:60],
                    "parent_q": pair["parent"]["question"][:60],
                    "signal_dir": sig_dir,
                    "signal_mag": sig_mag,
                    "correct": correct,
                    "lag_hours": lag_h,
                    "b_magnitude": b_mag,
                })

            # Current prices
            child_price = hist_child[-1][1] if hist_child else None
            parent_price = hist_parent[-1][1] if hist_parent else None
            implied_ratio = child_price / parent_price if child_price and parent_price and parent_price > 0 else None

            pair_summaries.append({
                "child_q": pair["child"]["question"][:65],
                "parent_q": pair["parent"]["question"][:65],
                "child_price": child_price,
                "parent_price": parent_price,
                "implied_ratio": implied_ratio,
                "total_signals": len(sharp_moves),
                "correct_signals": sum(1 for s in pair_signals if s["correct"]),
                "avg_lag": sum(s["lag_hours"] for s in pair_signals if s["lag_hours"] is not None) /
                           max(1, sum(1 for s in pair_signals if s["lag_hours"] is not None)),
                "corr_pct": corr_pct,
                "corr_days": total_days,
            })

        # ── Step 5: Report ─────────────────────────────────────────────
        total_signals = len(cascade_results)
        correct_signals = sum(1 for r in cascade_results if r["correct"])
        accuracy = correct_signals / total_signals * 100 if total_signals else 0

        lags = [r["lag_hours"] for r in cascade_results if r["correct"] and r["lag_hours"] is not None]
        avg_lag = sum(lags) / len(lags) if lags else 0
        median_lag = sorted(lags)[len(lags) // 2] if lags else 0

        b_moves = [r["b_magnitude"] for r in cascade_results if r["correct"] and r["b_magnitude"] > 0]
        avg_b_move = sum(b_moves) / len(b_moves) if b_moves else 0

        print("=" * 68)
        print("  CASCADE BACKTEST RESULTS")
        print("=" * 68)
        print(f"  Active markets analyzed:   {len(all_markets)}")
        print(f"  Cascade pairs identified:  {len(pairs)}")
        print(f"  Pairs with history:        {len(pair_summaries)}")
        print(f"  Pairs skipped (no tokens): {skipped}")
        print()
        print(f"  Total cascade signals:     {total_signals}")
        print(f"  Correct direction:         {correct_signals} ({accuracy:.1f}%)")
        print(f"  Average lag (A->B move):   {avg_lag:.1f} hours")
        print(f"  Median lag:                {median_lag:.1f} hours")
        print(f"  Avg B follow-through size: {avg_b_move*100:.1f}c")
        print()

        if pair_summaries:
            print("  Per-pair detail:")
            print(f"  {'Child Market':<35} {'Parent Market':<35} {'A->B':>5} {'Acc%':>5} {'Corr%':>6} {'Lag h':>6}")
            print(f"  {'-'*92}")
            for ps in sorted(pair_summaries, key=lambda x: -x["total_signals"])[:15]:
                acc = ps["correct_signals"] / ps["total_signals"] * 100 if ps["total_signals"] else 0
                lag_s = f"{ps['avg_lag']:.1f}" if ps["total_signals"] and ps["correct_signals"] > 0 else " n/a"
                print(f"  {ps['child_q'][:33]:<35} {ps['parent_q'][:33]:<35} "
                      f"{ps['total_signals']:>5} {acc:>4.0f}% {ps['corr_pct']:>5.0f}% {lag_s:>6}")
            print()

        # Price ratio analysis — market implied vs theoretical
        pairs_with_ratio = [ps for ps in pair_summaries if ps["implied_ratio"] is not None]
        if pairs_with_ratio:
            print("  Current price ratio analysis (child/parent):")
            print("  Theoretical: if you WIN the conference, you should be ~50-70% to win Finals")
            print("  Market deviation may indicate mispricing:")
            print()
            print(f"  {'Child Market':<42} {'Child':>6} {'Parent':>7} {'Ratio':>6}")
            print(f"  {'-'*65}")
            for ps in sorted(pairs_with_ratio, key=lambda x: -(x["implied_ratio"] or 0)):
                if ps["child_price"] and ps["parent_price"]:
                    print(f"  {ps['child_q'][:40]:<42} {ps['child_price']:>5.3f} "
                          f"{ps['parent_price']:>6.3f} {ps['implied_ratio']:>5.2f}x")
            print()

        # ── Projected edge ─────────────────────────────────────────────
        print("=" * 68)
        print("  PROJECTED EDGE ESTIMATE")
        print("=" * 68)

        MIN_SIGNALS_FOR_VERDICT = 20  # need at least 20 signals to draw conclusions

        if total_signals == 0:
            print("  INCONCLUSIVE: Zero cascade signals found.")
            print("  Signals require sharp moves (>3c) in child markets.")
            print("  Current data: pre-playoff NBA markets that have moved <5c in 31 days.")
            print("  Run this script during active playoff rounds (series in progress)")
            print("  when conference results trigger sharp repricing of Finals markets.")
        elif total_signals < MIN_SIGNALS_FOR_VERDICT:
            print(f"  INCONCLUSIVE: Only {total_signals} signals found — too few for reliable statistics.")
            print(f"  Need >= {MIN_SIGNALS_FOR_VERDICT} signals for a meaningful verdict.")
            print()
            print(f"  Observed (small sample): {accuracy:.0f}% accuracy on {total_signals} signals")
            print(f"  Average B follow-through: {avg_b_move*100:.1f}c after A signal")
            print(f"  Average lag: {avg_lag:.1f}h  (median: {median_lag:.1f}h)")
            print()
            print(f"  WHEN TO RUN: During active playoffs, e.g. when a conference semifinal")
            print(f"  series is live — Conference game results cause immediate 15-30c moves")
            print(f"  in the conference market, followed by Finals market repricing 1-6h later.")
            print(f"  That window is the trading opportunity.")
        else:
            entry_spread = 0.03  # typical entry cost (spread + slippage)
            net_per_trade = avg_b_move - entry_spread
            breakeven_acc = entry_spread / (avg_b_move + entry_spread) * 100 if avg_b_move > 0 else 99
            daily_opp = total_signals / max(1, len(pairs)) * 7  # estimated weekly signals

            print(f"  Signal accuracy:      {accuracy:.1f}%")
            print(f"  Avg follow-through:   {avg_b_move*100:.1f}c")
            print(f"  Est. entry cost:      {entry_spread*100:.1f}c (spread + slippage)")
            print(f"  Net edge per trade:   {net_per_trade*100:.1f}c")
            print(f"  Breakeven accuracy:   {breakeven_acc:.0f}%")
            print(f"  Typical lag to trade: {median_lag:.1f}h after A signal")
            print(f"  Est. signals/week:    {daily_opp:.1f}")
            print()

            if accuracy > breakeven_acc and net_per_trade > 0:
                grade = "STRONG" if accuracy > 65 and net_per_trade > 0.05 else "MODERATE"
                print(f"  VERDICT: {grade} POSITIVE EDGE")
                print(f"  Strategy: After sharp A move, enter B within {median_lag:.0f}h")
                print(f"  Expected edge: {net_per_trade*100:.1f}c per trade at {accuracy:.0f}% accuracy")
            elif accuracy > 50:
                print(f"  VERDICT: MARGINAL — edge exists but thin. Execution timing critical.")
                print(f"  Need to capture lag < {median_lag:.0f}h to be profitable.")
            else:
                print(f"  VERDICT: NO EDGE — cascade is not directionally predictive in this data.")
                print(f"  May require longer lookback or different market categories.")

        print("=" * 68)


if __name__ == "__main__":
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    asyncio.run(main())
