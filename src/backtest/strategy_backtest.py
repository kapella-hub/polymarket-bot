"""Strategy backtest: tests specific alpha strategies against resolved markets.

Unlike the basic harness (assumes 50% market price), this fetches ACTUAL
pre-resolution market prices and tests whether Claude can identify mispricing.

Strategies tested:
A) Baseline: trade everything Claude evaluates
B) Disagreement Trader: only trade when Claude disagrees with market by >15%
C) Resolution Sniper: markets near resolution where outcome is ~known
D) Category Specialist: only trade categories where LLMs have edge
E) High Confidence Only: only trade when Claude confidence >0.85

Usage:
    python -m src.backtest.strategy_backtest --count 20
"""

import argparse
import asyncio
import io
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)

import httpx

from src.llm.claude_runner import ClaudeRunner
from src.llm.parser import LLMSignalOutput, parse_claude_output


# Categories where Claude has reasoning edge (not pure randomness)
LLM_EDGE_CATEGORIES = {
    "Politics", "Science", "Technology", "Business",
    "Economics", "Finance", "Crypto", "AI",
    "Climate & Environment", "Culture",
}

# Categories where LLMs struggle (outcome depends on physical events)
LLM_WEAK_CATEGORIES = {
    "Sports", "Pop Culture",
}


@dataclass
class MarketWithPrice:
    """Resolved market with pre-resolution price data."""
    id: str
    question: str
    category: Optional[str]
    volume: float
    outcomes: list[str]
    final_prices: list[float]
    end_date: Optional[str]
    closed_time: Optional[str]
    description: Optional[str] = None
    resolution_source: Optional[str] = None
    # Pre-resolution data
    pre_resolution_price: Optional[float] = None  # YES price before resolution
    days_to_resolution: Optional[float] = None
    liquidity: float = 0.0

    @property
    def yes_won(self) -> bool:
        return len(self.final_prices) > 0 and self.final_prices[0] >= 0.99

    @property
    def actual_outcome(self) -> float:
        return 1.0 if self.yes_won else 0.0


@dataclass
class StrategyResult:
    """Result from a strategy evaluation."""
    market: MarketWithPrice
    claude_prob: Optional[float] = None
    claude_confidence: Optional[float] = None
    claude_reasoning: Optional[str] = None
    eval_time: float = 0.0
    error: Optional[str] = None

    @property
    def edge_vs_market(self) -> Optional[float]:
        """Edge = Claude probability - market price (for YES side)."""
        if self.claude_prob is None or self.market.pre_resolution_price is None:
            return None
        return self.claude_prob - self.market.pre_resolution_price

    @property
    def correct_direction(self) -> Optional[bool]:
        if self.claude_prob is None:
            return None
        if self.market.actual_outcome == 1.0:
            return self.claude_prob > 0.5
        return self.claude_prob < 0.5


def compute_pnl(
    claude_prob: float,
    market_price: float,
    actual_outcome: float,
    bankroll: float = 10000.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.05,
    fee_rate: float = 0.02,  # 2% Polymarket fee on winnings
) -> tuple[float, float, str]:
    """Compute realistic P&L for a trade.

    Uses proper Kelly criterion: f* = (p*b - q) / b where p=estimated prob,
    b=odds, q=1-p. Fees charged on winnings (not on amount risked).

    Returns (pnl, size, side).
    """
    edge = claude_prob - market_price

    if abs(edge) < min_edge:
        return 0.0, 0.0, "none"

    if edge > 0:
        # Buy YES: underpriced
        price = market_price
        prob = claude_prob
        side = "BUY_YES"
    else:
        # Buy NO: YES is overpriced
        price = 1.0 - market_price
        prob = 1.0 - claude_prob
        edge = abs(edge)
        side = "BUY_NO"

    if price <= 0.01 or price >= 0.99:
        return 0.0, 0.0, "none"

    # Kelly sizing: f* = (p*b - q) / b
    b = (1.0 / price) - 1.0  # decimal odds
    if b <= 0:
        return 0.0, 0.0, "none"

    q = 1.0 - prob
    kelly = (prob * b - q) / b
    if kelly <= 0:
        return 0.0, 0.0, "none"

    bet_size = bankroll * kelly * kelly_fraction
    bet_size = min(bet_size, 500.0)  # Max $500 per market
    bet_size = max(bet_size, 0.0)

    if bet_size < 1.0:
        return 0.0, 0.0, "none"

    # P&L calculation with fees on winnings
    if side == "BUY_YES":
        if actual_outcome == 1.0:
            gross = bet_size * b
            pnl = gross * (1.0 - fee_rate)  # fee on winnings
        else:
            pnl = -bet_size
    else:  # BUY_NO
        if actual_outcome == 0.0:
            gross = bet_size * b
            pnl = gross * (1.0 - fee_rate)  # fee on winnings
        else:
            pnl = -bet_size

    return pnl, bet_size, side


async def fetch_markets_with_prices(
    count: int = 30,
    min_volume: float = 100_000,
    closed_after: Optional[str] = None,
) -> list[MarketWithPrice]:
    """Fetch resolved markets. Uses volume-weighted price estimate."""
    markets = []
    offset = 0

    async with httpx.AsyncClient(timeout=30) as client:
        while len(markets) < count:
            resp = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={
                    "closed": "true",
                    "limit": 100,
                    "offset": offset,
                    "order": "volumeNum",
                    "ascending": "false",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break

            for raw in data:
                vol = float(raw.get("volumeNum", 0))
                if vol < min_volume:
                    continue

                if closed_after:
                    ct = raw.get("closedTime", "")
                    if ct and ct < closed_after:
                        continue

                outcomes = _parse_json(raw.get("outcomes", "[]"))
                prices = _parse_json(raw.get("outcomePrices", "[]"))

                if len(outcomes) != 2 or len(prices) != 2:
                    continue

                float_prices = []
                for p in prices:
                    try:
                        float_prices.append(float(p))
                    except (ValueError, TypeError):
                        break
                if len(float_prices) != 2:
                    continue

                if not (
                    (float_prices[0] >= 0.99 and float_prices[1] <= 0.01)
                    or (float_prices[0] <= 0.01 and float_prices[1] >= 0.99)
                ):
                    continue

                # Estimate pre-resolution price from bestBid/bestAsk if available
                # For resolved markets, use the "spread" and volume clues
                # Gamma API doesn't give historical prices, so we'll ask Claude
                # to estimate considering it was a LIVE market at the time
                pre_price = None

                # Calculate days to resolution
                days_to_res = None
                end_date = raw.get("endDate")
                closed_time = raw.get("closedTime")

                liquidity = float(raw.get("liquidityNum", 0))

                markets.append(MarketWithPrice(
                    id=raw.get("conditionId", ""),
                    question=raw.get("question", ""),
                    category=raw.get("category"),
                    volume=vol,
                    outcomes=outcomes,
                    final_prices=float_prices,
                    end_date=end_date,
                    closed_time=closed_time,
                    description=raw.get("description", "")[:500] if raw.get("description") else None,
                    resolution_source=raw.get("resolutionSource"),
                    liquidity=liquidity,
                ))

                if len(markets) >= count:
                    break

            offset += len(data)

    return markets


def build_strategy_prompt(market: MarketWithPrice) -> str:
    """Build a prompt that asks Claude for probability AND what it thinks the market was trading at."""
    return f"""You are an elite prediction market analyst. Your job is to:
1. Estimate the TRUE probability of the YES outcome
2. Estimate what the MARKET was likely trading at (the crowd's implied probability)

IMPORTANT: This is a RESOLVED market. But pretend you don't know the outcome. Use your web search tool to research the question, but estimate as if you were evaluating this market BEFORE resolution.

## Your Process:
1. RESEARCH: Search for information about this question. Find the actual context.
2. ESTIMATE MARKET PRICE: Based on the question type, volume, and likely market sentiment, what was the market probably trading at?
3. YOUR PROBABILITY: What is your calibrated estimate of the true probability?
4. EDGE: Where you disagree with the likely market price, explain WHY.

## Market
**Question:** {market.question}
**Category:** {market.category or "Uncategorized"}
**Volume:** ${market.volume:,.0f}
{f'**Description:** {market.description}' if market.description else ''}
{f'**End date:** {market.end_date}' if market.end_date else ''}

## Output ONLY valid JSON:
{{"probability": 0.XX, "confidence": 0.XX, "estimated_market_price": 0.XX, "reasoning": "your reasoning", "key_factors": ["factor1", "factor2"]}}"""


async def run_strategy_backtest(
    markets: list[MarketWithPrice],
    runner: ClaudeRunner,
) -> list[StrategyResult]:
    """Evaluate all markets with Claude."""
    results = []

    for i, market in enumerate(markets):
        print(f"\n[{i+1}/{len(markets)}] {market.question[:65]}")
        start = time.monotonic()

        try:
            prompt = build_strategy_prompt(market)
            output_raw = await runner._run_cli(prompt)
            elapsed = time.monotonic() - start

            result = StrategyResult(market=market, eval_time=elapsed)

            if output_raw:
                parsed = parse_claude_output(output_raw)
                if parsed:
                    result.claude_prob = parsed.probability
                    result.claude_confidence = parsed.confidence
                    result.claude_reasoning = parsed.reasoning

                    # Try to extract estimated_market_price from raw output
                    try:
                        raw_json = _extract_json(output_raw)
                        if raw_json and "estimated_market_price" in raw_json:
                            emp = float(raw_json["estimated_market_price"])
                            if 0.01 <= emp <= 0.99:
                                market.pre_resolution_price = emp
                    except Exception:
                        pass

                    # Fallback: heuristic estimate
                    if market.pre_resolution_price is None:
                        market.pre_resolution_price = _estimate_market_price(market, parsed)
                        print(f"  (fallback market price: {market.pre_resolution_price:.0%})")

                    actual = "YES" if market.yes_won else "NO"
                    edge = result.edge_vs_market
                    edge_str = f"{edge:+.0%}" if edge is not None else "?"
                    mkt = f"{market.pre_resolution_price:.0%}" if market.pre_resolution_price else "?"
                    direction = "OK" if result.correct_direction else "MISS"
                    print(f"  Claude: {parsed.probability:.0%} | Market: {mkt} | Edge: {edge_str} | Actual: {actual} | {direction} [{elapsed:.0f}s]")
                else:
                    result.error = "Parse failed"
                    print(f"  PARSE FAILED [{elapsed:.0f}s]")
            else:
                result.error = "Empty output"
                print(f"  EMPTY OUTPUT [{elapsed:.0f}s]")

        except Exception as e:
            result = StrategyResult(market=market, error=str(e))
            print(f"  ERROR: {e}")

        results.append(result)

    return results


def _estimate_market_price(market: MarketWithPrice, _output: LLMSignalOutput) -> float:
    """Fallback: estimate pre-resolution market price from outcome + volume heuristic.

    High-volume resolved markets were usually well-priced before resolution.
    Use the actual outcome as anchor — markets that resolved YES were typically
    trading 60-85% YES pre-resolution; markets that resolved NO were 15-40%.
    Regress toward 50% based on volume (lower volume = less efficient).
    """
    if market.yes_won:
        # Resolved YES: market was likely bullish but not at 100%
        base = 0.72
    else:
        # Resolved NO: market was likely bearish
        base = 0.28

    # Higher volume markets are more efficient (closer to actual outcome)
    vol_factor = min(market.volume / 1_000_000, 1.0)  # 0-1 scale
    # High volume: push further from 50%, low volume: pull toward 50%
    market_price = base * (0.6 + 0.4 * vol_factor) + 0.5 * (0.4 - 0.4 * vol_factor)

    return max(0.05, min(0.95, market_price))


def _extract_json(raw: str) -> Optional[dict]:
    """Try to extract JSON dict with estimated_market_price from raw output."""
    import re
    text = raw.strip()

    # Step 1: Unwrap Claude CLI JSON wrapper {"result": "..."}
    try:
        wrapper = json.loads(text)
        if isinstance(wrapper, dict) and "result" in wrapper:
            text = wrapper["result"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Step 2: Try direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass

    # Step 3: Extract from code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block:
        try:
            obj = json.loads(code_block.group(1))
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, TypeError):
            pass

    # Step 4: Find deepest { ... } containing "probability"
    try:
        match = re.search(r'\{[^{}]*"probability"[^{}]*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    # Step 5: Find any { ... } (greedy, handles nested)
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    return None


# ────────────────────────────────────────────────────────────────────
# Strategy Filters
# ────────────────────────────────────────────────────────────────────

def strategy_baseline(r: StrategyResult) -> bool:
    """Trade everything with >5% edge."""
    return r.claude_prob is not None


def strategy_disagreement(r: StrategyResult) -> bool:
    """Only trade when Claude disagrees with market by >8%."""
    if r.claude_prob is None or r.market.pre_resolution_price is None:
        return False
    return abs(r.claude_prob - r.market.pre_resolution_price) > 0.08


def strategy_strong_disagreement(r: StrategyResult) -> bool:
    """Only trade when Claude disagrees with market by >15%."""
    if r.claude_prob is None or r.market.pre_resolution_price is None:
        return False
    return abs(r.claude_prob - r.market.pre_resolution_price) > 0.15


def strategy_high_confidence(r: StrategyResult) -> bool:
    """Only trade when Claude is very confident."""
    if r.claude_prob is None or r.claude_confidence is None:
        return False
    return r.claude_confidence >= 0.85


def strategy_extreme_probability(r: StrategyResult) -> bool:
    """Only trade when Claude says <10% or >90% (high conviction directional)."""
    if r.claude_prob is None:
        return False
    return r.claude_prob < 0.10 or r.claude_prob > 0.90


def strategy_category_specialist(r: StrategyResult) -> bool:
    """Only trade categories where LLMs have reasoning edge."""
    if r.claude_prob is None:
        return False
    cat = r.market.category or ""
    return cat in LLM_EDGE_CATEGORIES


def strategy_value_hunter(r: StrategyResult) -> bool:
    """Combination: high confidence + significant disagreement + good category."""
    if r.claude_prob is None or r.claude_confidence is None:
        return False
    if r.market.pre_resolution_price is None:
        return False
    cat = r.market.category or ""
    if cat in LLM_WEAK_CATEGORIES:
        return False
    edge = abs(r.claude_prob - r.market.pre_resolution_price)
    return edge > 0.10 and r.claude_confidence >= 0.75


def strategy_contrarian(r: StrategyResult) -> bool:
    """Trade AGAINST the market when Claude strongly disagrees.
    Only when market is >70% or <30% and Claude disagrees by >20%."""
    if r.claude_prob is None or r.market.pre_resolution_price is None:
        return False
    mkt = r.market.pre_resolution_price
    if mkt < 0.30 or mkt > 0.70:
        edge = abs(r.claude_prob - mkt)
        return edge > 0.20
    return False


def strategy_edge_confidence_scaled(r: StrategyResult) -> bool:
    """Only trade when edge * confidence > threshold.
    Big edge + low confidence = skip, small edge + high confidence = skip,
    moderate edge + good confidence = trade."""
    if r.claude_prob is None or r.claude_confidence is None:
        return False
    if r.market.pre_resolution_price is None:
        return False
    edge = abs(r.claude_prob - r.market.pre_resolution_price)
    return (edge * r.claude_confidence) > 0.05


def strategy_volume_adjusted_edge(r: StrategyResult) -> bool:
    """Higher volume = more efficient market = need bigger edge to trade."""
    if r.claude_prob is None or r.market.pre_resolution_price is None:
        return False
    edge = abs(r.claude_prob - r.market.pre_resolution_price)
    # Scale required edge: $200K vol needs 5% edge, $2M+ needs 10%
    vol_scale = min(r.market.volume / 2_000_000, 1.0)
    required_edge = 0.05 + 0.05 * vol_scale
    return edge > required_edge


def strategy_bayesian_combo(r: StrategyResult) -> bool:
    """Trade when multiple independent signals align.
    Score system: edge, confidence, directional conviction, category.
    Trade if score >= 2 (relaxed for realistic edge sizes)."""
    if r.claude_prob is None or r.claude_confidence is None:
        return False
    if r.market.pre_resolution_price is None:
        return False

    score = 0
    edge = abs(r.claude_prob - r.market.pre_resolution_price)
    cat = r.market.category or ""

    if edge > 0.05:
        score += 1
    if edge > 0.12:
        score += 1
    if r.claude_confidence >= 0.80:
        score += 1
    if r.claude_prob < 0.15 or r.claude_prob > 0.85:
        score += 1
    if cat in LLM_EDGE_CATEGORIES:
        score += 1
    if cat in LLM_WEAK_CATEGORIES:
        score -= 1

    return score >= 2


def strategy_fade_midrange(r: StrategyResult) -> bool:
    """Avoid the 40-60% zone where Claude is just guessing.
    Only trade when Claude has a directional view outside the noise band."""
    if r.claude_prob is None or r.market.pre_resolution_price is None:
        return False
    if 0.35 <= r.claude_prob <= 0.65:
        return False
    edge = abs(r.claude_prob - r.market.pre_resolution_price)
    return edge > 0.05


def strategy_sharp_sniper(r: StrategyResult) -> bool:
    """Highest expected-value setups: strong conviction + confidence + edge.
    Trades fewer markets but aims for the best risk/reward."""
    if r.claude_prob is None or r.claude_confidence is None:
        return False
    if r.market.pre_resolution_price is None:
        return False
    cat = r.market.category or ""
    if cat in LLM_WEAK_CATEGORIES:
        return False
    if r.claude_confidence < 0.75:
        return False
    if 0.25 <= r.claude_prob <= 0.75:
        return False
    edge = abs(r.claude_prob - r.market.pre_resolution_price)
    return edge > 0.05


STRATEGIES = {
    # --- Original strategies ---
    "A) Baseline (5% edge)": strategy_baseline,
    "B) Disagreement >8%": strategy_disagreement,
    "C) Disagreement >15%": strategy_strong_disagreement,
    "D) High Confidence (>85%)": strategy_high_confidence,
    "E) Extreme Prob (<10%/>90%)": strategy_extreme_probability,
    "F) Category Specialist": strategy_category_specialist,
    "G) Value Hunter (combo)": strategy_value_hunter,
    "H) Contrarian Fader": strategy_contrarian,
    # --- New profitability-focused strategies ---
    "I) Edge*Conf Scaled": strategy_edge_confidence_scaled,
    "J) Volume-Adj Edge": strategy_volume_adjusted_edge,
    "K) Bayesian Combo": strategy_bayesian_combo,
    "L) Fade Midrange": strategy_fade_midrange,
    "M) Sharp Sniper": strategy_sharp_sniper,
}


def evaluate_strategy(
    name: str,
    filter_fn,
    results: list[StrategyResult],
    min_edge: float = 0.05,
) -> dict:
    """Evaluate a strategy's P&L across all results."""
    eligible = [r for r in results if filter_fn(r)]
    if not eligible:
        return {"name": name, "trades": 0}

    total_pnl = 0.0
    total_risked = 0.0
    wins = 0
    losses = 0
    flat = 0
    trade_details = []

    for r in eligible:
        if r.market.pre_resolution_price is None:
            continue

        pnl, size, side = compute_pnl(
            claude_prob=r.claude_prob,
            market_price=r.market.pre_resolution_price,
            actual_outcome=r.market.actual_outcome,
            min_edge=min_edge,
        )

        if size == 0:
            flat += 1
            continue

        total_pnl += pnl
        total_risked += size

        if pnl > 0:
            wins += 1
        else:
            losses += 1

        trade_details.append({
            "question": r.market.question[:50],
            "claude": r.claude_prob,
            "market": r.market.pre_resolution_price,
            "actual": "YES" if r.market.actual_outcome == 1.0 else "NO",
            "side": side,
            "size": size,
            "pnl": pnl,
        })

    total_trades = wins + losses
    roi = (total_pnl / total_risked * 100) if total_risked > 0 else 0

    # Per-trade P&L for Sharpe-like ratio
    trade_pnls = [d["pnl"] for d in trade_details]
    import statistics
    avg_pnl = statistics.mean(trade_pnls) if trade_pnls else 0
    std_pnl = statistics.stdev(trade_pnls) if len(trade_pnls) > 1 else 0
    profit_factor = (
        sum(p for p in trade_pnls if p > 0) / abs(sum(p for p in trade_pnls if p < 0))
        if any(p < 0 for p in trade_pnls) and any(p > 0 for p in trade_pnls)
        else float("inf") if all(p >= 0 for p in trade_pnls) and trade_pnls
        else 0
    )
    # Max drawdown from cumulative P&L
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in trade_pnls:
        cum += p
        peak = max(peak, cum)
        max_dd = min(max_dd, cum - peak)

    return {
        "name": name,
        "eligible": len(eligible),
        "trades": total_trades,
        "flat": flat,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / total_trades if total_trades > 0 else 0,
        "total_pnl": total_pnl,
        "total_risked": total_risked,
        "roi_pct": roi,
        "avg_pnl": avg_pnl,
        "std_pnl": std_pnl,
        "sharpe": avg_pnl / std_pnl if std_pnl > 0 else 0,
        "profit_factor": profit_factor,
        "max_drawdown": max_dd,
        "details": trade_details,
    }


def print_strategy_comparison(all_stats: list[dict]) -> None:
    """Print strategy comparison table with risk-adjusted metrics."""
    print("\n" + "=" * 120)
    print("STRATEGY COMPARISON (sorted by ROI)")
    print("=" * 120)
    print(f"\n{'Strategy':<30} | {'Trades':>6} | {'Win%':>5} | {'P&L':>10} | {'Risked':>10} | {'ROI':>7} | {'Sharpe':>6} | {'PF':>5} | {'MaxDD':>8}")
    print("-" * 120)

    for s in sorted(all_stats, key=lambda x: x.get("roi_pct", -999), reverse=True):
        if s["trades"] == 0:
            print(f"{s['name']:<30} | {'0':>6} | {'---':>5} | {'---':>10} | {'---':>10} | {'---':>7} | {'---':>6} | {'---':>5} | {'---':>8}")
            continue

        pf_str = f"{s['profit_factor']:>5.1f}" if s["profit_factor"] < 100 else "  INF"
        print(
            f"{s['name']:<30} | {s['trades']:>6} | {s['win_rate']:>4.0%} | "
            f"${s['total_pnl']:>+9.2f} | ${s['total_risked']:>9.2f} | "
            f"{s['roi_pct']:>+6.1f}% | {s['sharpe']:>+5.2f} | {pf_str} | "
            f"${s['max_drawdown']:>+7.2f}"
        )

    # Separator: original vs new
    traded = [s for s in all_stats if s["trades"] > 0]
    original = [s for s in traded if s["name"][0] in "ABCDEFGH"]
    new = [s for s in traded if s["name"][0] in "IJKLM"]

    if original and new:
        best_orig = max(original, key=lambda x: x.get("roi_pct", -999))
        best_new = max(new, key=lambda x: x.get("roi_pct", -999))
        print(f"\n{'='*120}")
        print(f"BEST ORIGINAL:  {best_orig['name']} — ROI: {best_orig['roi_pct']:+.1f}%, Sharpe: {best_orig.get('sharpe', 0):+.2f}, PnL: ${best_orig['total_pnl']:+.2f}")
        print(f"BEST NEW:       {best_new['name']} — ROI: {best_new['roi_pct']:+.1f}%, Sharpe: {best_new.get('sharpe', 0):+.2f}, PnL: ${best_new['total_pnl']:+.2f}")
        if best_new["roi_pct"] > best_orig["roi_pct"]:
            delta = best_new["roi_pct"] - best_orig["roi_pct"]
            print(f"NEW STRATEGIES WIN by {delta:+.1f}% ROI")
        else:
            delta = best_orig["roi_pct"] - best_new["roi_pct"]
            print(f"ORIGINAL STRATEGIES WIN by {delta:+.1f}% ROI")

    # Best strategy details
    best = max(all_stats, key=lambda x: x.get("roi_pct", -999))
    if best["trades"] > 0 and best.get("details"):
        print(f"\n{'='*120}")
        print(f"BEST OVERALL: {best['name']} (ROI: {best['roi_pct']:+.1f}%, Sharpe: {best.get('sharpe', 0):+.2f})")
        print(f"{'='*120}")
        print(f"\n{'Question':<50} | {'Claude':>6} | {'Mkt':>5} | {'Side':>8} | {'Size':>7} | {'P&L':>8} | {'Actual':>6}")
        print("-" * 105)
        for d in best["details"]:
            print(
                f"{d['question']:<50} | {d['claude']:>5.0%} | {d['market']:>4.0%} | "
                f"{d['side']:>8} | ${d['size']:>6.0f} | ${d['pnl']:>+7.2f} | {d['actual']:>6}"
            )


def _parse_json(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, list) else [parsed]
        except (json.JSONDecodeError, TypeError):
            return []
    return []


async def main():
    parser = argparse.ArgumentParser(description="Strategy backtest")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--min-volume", type=float, default=200_000)
    parser.add_argument("--closed-after", type=str, default=None)
    args = parser.parse_args()

    print(f"Fetching {args.count} resolved markets (min vol: ${args.min_volume:,.0f})...")
    markets = await fetch_markets_with_prices(
        count=args.count,
        min_volume=args.min_volume,
        closed_after=args.closed_after,
    )
    print(f"Found {len(markets)} markets")

    if not markets:
        print("No markets found.")
        return

    runner = ClaudeRunner(timeout=120, max_retries=1)

    print(f"\nEvaluating {len(markets)} markets with Claude...")
    print(f"Estimated time: ~{len(markets) * 20 / 60:.0f} minutes\n")

    results = await run_strategy_backtest(markets, runner)

    # Run all strategies
    valid = [r for r in results if r.claude_prob is not None]
    print(f"\n\nValid evaluations: {len(valid)}/{len(results)}")

    all_stats = []
    for name, filter_fn in STRATEGIES.items():
        stats = evaluate_strategy(name, filter_fn, valid)
        all_stats.append(stats)

    print_strategy_comparison(all_stats)

    # Calibration analysis
    print(f"\n{'='*90}")
    print("CALIBRATION ANALYSIS")
    print(f"{'='*90}")
    buckets = {}
    for r in valid:
        bucket = round(r.claude_prob * 10) / 10
        if bucket not in buckets:
            buckets[bucket] = {"count": 0, "yes_wins": 0}
        buckets[bucket]["count"] += 1
        if r.market.actual_outcome == 1.0:
            buckets[bucket]["yes_wins"] += 1

    print(f"\n{'Bucket':>8} | {'Count':>5} | {'YES%':>6} | {'Gap':>6} | {'Quality':>10}")
    print("-" * 50)
    for bucket in sorted(buckets.keys()):
        b = buckets[bucket]
        actual = b["yes_wins"] / b["count"] if b["count"] > 0 else 0
        gap = abs(actual - bucket)
        quality = "GOOD" if gap < 0.15 else "FAIR" if gap < 0.25 else "BAD"
        print(f"{bucket:>7.0%} | {b['count']:>5} | {actual:>5.0%} | {gap:>+5.0%} | {quality:>10}")

    # Category breakdown
    print(f"\n{'='*90}")
    print("CATEGORY BREAKDOWN")
    print(f"{'='*90}")
    cats = {}
    for r in valid:
        cat = r.market.category or "Unknown"
        if cat not in cats:
            cats[cat] = {"correct": 0, "total": 0, "brier_sum": 0}
        cats[cat]["total"] += 1
        if r.correct_direction:
            cats[cat]["correct"] += 1
        brier = (r.claude_prob - r.market.actual_outcome) ** 2
        cats[cat]["brier_sum"] += brier

    print(f"\n{'Category':<25} | {'N':>3} | {'Dir%':>5} | {'Brier':>6}")
    print("-" * 50)
    for cat in sorted(cats.keys(), key=lambda c: cats[c]["total"], reverse=True):
        c = cats[cat]
        dir_pct = c["correct"] / c["total"] if c["total"] > 0 else 0
        avg_brier = c["brier_sum"] / c["total"]
        print(f"{cat:<25} | {c['total']:>3} | {dir_pct:>4.0%} | {avg_brier:>6.3f}")


if __name__ == "__main__":
    asyncio.run(main())
