"""Backtest harness: evaluate Claude against resolved Polymarket markets.

Fetches resolved markets, runs Claude CLI on each, scores predictions
against actual outcomes, and calculates simulated P&L.

Usage:
    python -m src.backtest.harness --count 20 --min-volume 100000
"""

import argparse
import asyncio
import io
import json
import sys
import time

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import httpx

from src.llm.claude_runner import ClaudeRunner
from src.llm.parser import LLMSignalOutput


@dataclass
class ResolvedMarket:
    """A resolved market with known outcome."""

    id: str
    question: str
    category: Optional[str]
    volume: float
    outcomes: list[str]
    final_prices: list[float]  # [1, 0] or [0, 1]
    end_date: Optional[str]
    closed_time: Optional[str]
    description: Optional[str] = None
    resolution_source: Optional[str] = None

    @property
    def winning_outcome(self) -> str:
        """Which outcome won (e.g., 'Yes' or 'No')."""
        for i, price in enumerate(self.final_prices):
            if price >= 0.99:
                return self.outcomes[i] if i < len(self.outcomes) else f"Outcome {i}"
        return "Unknown"

    @property
    def yes_won(self) -> bool:
        """Did the YES outcome win?"""
        return len(self.final_prices) > 0 and self.final_prices[0] >= 0.99


@dataclass
class BacktestResult:
    """Result of evaluating one market."""

    market: ResolvedMarket
    claude_probability: Optional[float] = None  # Claude's YES probability
    claude_confidence: Optional[float] = None
    claude_reasoning: Optional[str] = None
    actual_outcome: float = 0.0  # 1.0 if YES, 0.0 if NO
    error: Optional[str] = None
    eval_time_seconds: float = 0.0

    @property
    def brier_score(self) -> Optional[float]:
        """Brier score: (prediction - outcome)^2. Lower is better."""
        if self.claude_probability is None:
            return None
        return (self.claude_probability - self.actual_outcome) ** 2

    @property
    def edge(self) -> Optional[float]:
        """Edge if we assumed 50% market price."""
        if self.claude_probability is None:
            return None
        if self.actual_outcome == 1.0:
            return self.claude_probability - 0.5  # Positive = correct direction
        else:
            return 0.5 - self.claude_probability  # Positive = correct direction

    @property
    def correct_direction(self) -> Optional[bool]:
        """Did Claude lean the right way?"""
        if self.claude_probability is None:
            return None
        if self.actual_outcome == 1.0:
            return self.claude_probability > 0.5
        else:
            return self.claude_probability < 0.5

    @property
    def simulated_pnl(self) -> Optional[float]:
        """Simulated P&L per $100 bet at 50% market price.

        If Claude says 70% YES and YES wins:
          We buy YES at $0.50, it resolves to $1.00 = +$100
          Bet size = $100 * kelly_fraction
          kelly = (0.70 - 0.50) / (1/0.50 - 1) = 0.20 / 1.0 = 0.20
          Bet = $100 * 0.20 * 0.25 (quarter kelly) = $5
          PnL = $5 * (1.0/0.50 - 1) = $5 = +$5

        If Claude says 70% YES and NO wins:
          PnL = -$5
        """
        if self.claude_probability is None:
            return None

        assumed_market = 0.50
        prob = self.claude_probability
        edge = abs(prob - assumed_market)

        if edge < 0.05:  # Below minimum edge threshold
            return 0.0

        # Direction
        if prob > assumed_market:
            # Buy YES
            odds = (1.0 / assumed_market) - 1.0
            kelly = edge / odds if odds > 0 else 0
            bet = 100.0 * kelly * 0.25  # Quarter kelly on $100 bankroll
            if self.actual_outcome == 1.0:
                return bet * odds  # Win
            else:
                return -bet  # Lose
        else:
            # Buy NO (prob < 0.5, so NO is underpriced)
            no_market = 1.0 - assumed_market
            no_edge = (1.0 - prob) - no_market
            odds = (1.0 / no_market) - 1.0
            kelly = no_edge / odds if odds > 0 else 0
            bet = 100.0 * kelly * 0.25
            if self.actual_outcome == 0.0:
                return bet * odds  # Win (NO resolved)
            else:
                return -bet  # Lose


async def fetch_resolved_markets(
    count: int = 50,
    min_volume: float = 100_000,
    closed_after: Optional[str] = None,
) -> list[ResolvedMarket]:
    """Fetch resolved markets from Gamma API."""
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
            data = resp.json()
            if not data:
                break

            for raw in data:
                vol = float(raw.get("volumeNum", 0))
                if vol < min_volume:
                    continue

                # Date filter: skip markets closed before cutoff
                if closed_after:
                    closed_time = raw.get("closedTime", "")
                    if closed_time and closed_time < closed_after:
                        continue

                outcomes = _parse_json(raw.get("outcomes", "[]"))
                prices = _parse_json(raw.get("outcomePrices", "[]"))

                # Only include binary markets with clear resolution
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

                # Must be fully resolved (one price at 1, other at 0)
                if not (
                    (float_prices[0] >= 0.99 and float_prices[1] <= 0.01)
                    or (float_prices[0] <= 0.01 and float_prices[1] >= 0.99)
                ):
                    continue

                markets.append(
                    ResolvedMarket(
                        id=raw.get("conditionId", ""),
                        question=raw.get("question", ""),
                        category=raw.get("category"),
                        volume=vol,
                        outcomes=outcomes,
                        final_prices=float_prices,
                        end_date=raw.get("endDate"),
                        closed_time=raw.get("closedTime"),
                        description=raw.get("description", "")[:500] if raw.get("description") else None,
                        resolution_source=raw.get("resolutionSource"),
                    )
                )

                if len(markets) >= count:
                    break

            offset += len(data)

    return markets


def build_backtest_prompt(market: ResolvedMarket) -> str:
    """Build a prompt for Claude to evaluate a resolved market.

    We ask Claude to estimate probability WITHOUT telling it the outcome.
    """
    return f"""You are a prediction market probability analyst. Estimate the probability of the YES outcome.

IMPORTANT: Output ONLY valid JSON. No other text.

## Market

**Question:** {market.question}

**Possible outcomes:** {', '.join(market.outcomes)}

{f'**Description:** {market.description}' if market.description else ''}
{f'**Resolution date:** {market.end_date}' if market.end_date else ''}

Analyze this question and estimate the probability that the first outcome ("{market.outcomes[0]}") is correct.

Output format:
{{"probability": 0.XX, "confidence": 0.XX, "reasoning": "...", "key_factors": ["...", "..."]}}"""


async def run_backtest(
    markets: list[ResolvedMarket],
    runner: Optional[ClaudeRunner] = None,
) -> list[BacktestResult]:
    """Run Claude evaluation on each resolved market."""
    runner = runner or ClaudeRunner(timeout=90, max_retries=1)
    results = []

    for i, market in enumerate(markets):
        print(f"\n[{i+1}/{len(markets)}] {market.question[:70]}...")

        start = time.monotonic()
        prompt = build_backtest_prompt(market)

        try:
            output = await runner.evaluate(prompt)
            elapsed = time.monotonic() - start

            result = BacktestResult(
                market=market,
                actual_outcome=1.0 if market.yes_won else 0.0,
                eval_time_seconds=elapsed,
            )

            if output:
                result.claude_probability = output.probability
                result.claude_confidence = output.confidence
                result.claude_reasoning = output.reasoning
                direction = "CORRECT" if result.correct_direction else "WRONG"
                print(
                    f"  Claude: {output.probability:.0%} YES | "
                    f"Actual: {'YES' if market.yes_won else 'NO'} | "
                    f"{direction} | "
                    f"Brier: {result.brier_score:.3f} | "
                    f"{elapsed:.1f}s"
                )
            else:
                result.error = "Failed to parse output"
                print(f"  FAILED to get output ({elapsed:.1f}s)")

        except Exception as e:
            result = BacktestResult(
                market=market,
                actual_outcome=1.0 if market.yes_won else 0.0,
                error=str(e),
            )
            print(f"  ERROR: {e}")

        results.append(result)

    return results


def print_summary(results: list[BacktestResult]) -> None:
    """Print backtest summary statistics."""
    valid = [r for r in results if r.claude_probability is not None]
    failed = len(results) - len(valid)

    if not valid:
        print("\nNo valid results to summarize.")
        return

    # Brier scores
    brier_scores = [r.brier_score for r in valid if r.brier_score is not None]
    avg_brier = sum(brier_scores) / len(brier_scores)

    # Direction accuracy
    correct = sum(1 for r in valid if r.correct_direction)
    accuracy = correct / len(valid)

    # P&L
    pnls = [r.simulated_pnl for r in valid if r.simulated_pnl is not None]
    total_pnl = sum(pnls)
    winning_trades = sum(1 for p in pnls if p > 0)
    losing_trades = sum(1 for p in pnls if p < 0)
    flat_trades = sum(1 for p in pnls if p == 0)

    # Calibration buckets
    buckets = {}
    for r in valid:
        bucket = round(r.claude_probability * 10) / 10  # Round to nearest 0.1
        if bucket not in buckets:
            buckets[bucket] = {"count": 0, "yes_wins": 0}
        buckets[bucket]["count"] += 1
        if r.actual_outcome == 1.0:
            buckets[bucket]["yes_wins"] += 1

    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)
    print(f"\nMarkets evaluated: {len(valid)} ({failed} failed)")
    print(f"Avg eval time:     {sum(r.eval_time_seconds for r in valid) / len(valid):.1f}s")

    print(f"\n--- Accuracy ---")
    print(f"Direction correct: {correct}/{len(valid)} ({accuracy:.0%})")
    print(f"Avg Brier score:   {avg_brier:.4f}  (0=perfect, 0.25=coin flip)")

    # Brier score context
    if avg_brier < 0.15:
        quality = "Excellent"
    elif avg_brier < 0.20:
        quality = "Good"
    elif avg_brier < 0.25:
        quality = "Fair (barely better than coin flip)"
    else:
        quality = "Poor (worse than coin flip)"
    print(f"Signal quality:    {quality}")

    print(f"\n--- Simulated P&L (per $100 bankroll, quarter-Kelly, 50% assumed market) ---")
    print(f"Total P&L:         ${total_pnl:+.2f}")
    print(f"Winning trades:    {winning_trades}")
    print(f"Losing trades:     {losing_trades}")
    print(f"Flat (no edge):    {flat_trades}")
    if winning_trades + losing_trades > 0:
        win_rate = winning_trades / (winning_trades + losing_trades)
        print(f"Win rate:          {win_rate:.0%}")

    print(f"\n--- Calibration ---")
    print(f"{'Bucket':>8} | {'Count':>5} | {'YES wins':>8} | {'Actual %':>8} | {'Gap':>6}")
    print("-" * 50)
    for bucket in sorted(buckets.keys()):
        b = buckets[bucket]
        actual_pct = b["yes_wins"] / b["count"] if b["count"] > 0 else 0
        gap = actual_pct - bucket
        print(
            f"{bucket:>7.0%} | {b['count']:>5} | {b['yes_wins']:>8} | "
            f"{actual_pct:>7.0%} | {gap:>+5.0%}"
        )

    print("\n--- Individual Results ---")
    print(f"{'Question':<50} | {'Claude':>6} | {'Actual':>6} | {'Dir':>5} | {'PnL':>7}")
    print("-" * 85)
    for r in valid:
        q = r.market.question[:48]
        actual = "YES" if r.actual_outcome == 1.0 else "NO"
        direction = "OK" if r.correct_direction else "MISS"
        pnl = r.simulated_pnl or 0
        print(f"{q:<50} | {r.claude_probability:>5.0%} | {actual:>6} | {direction:>5} | ${pnl:>+6.2f}")


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
    parser = argparse.ArgumentParser(description="Backtest Claude against resolved Polymarket markets")
    parser.add_argument("--count", type=int, default=20, help="Number of markets to evaluate")
    parser.add_argument("--min-volume", type=float, default=100_000, help="Minimum market volume in USD")
    parser.add_argument("--closed-after", type=str, default=None, help="Only markets closed after this date (e.g., 2025-06)")
    args = parser.parse_args()

    print(f"Fetching {args.count} resolved markets (min volume: ${args.min_volume:,.0f})...")
    markets = await fetch_resolved_markets(count=args.count, min_volume=args.min_volume, closed_after=args.closed_after)
    print(f"Found {len(markets)} resolved markets")

    if not markets:
        print("No markets found. Try lowering --min-volume.")
        return

    print(f"\nRunning Claude CLI evaluation on {len(markets)} markets...")
    print("(This will take ~{:.0f} minutes at ~15s per market)".format(len(markets) * 15 / 60))

    results = await run_backtest(markets)
    print_summary(results)


if __name__ == "__main__":
    asyncio.run(main())
