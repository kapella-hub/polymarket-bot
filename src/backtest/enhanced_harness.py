"""Enhanced backtest: compares naive vs full innovation stack.

Runs both approaches on the same resolved markets and compares:
- Naive: single Claude call (baseline)
- Enhanced: superforecaster protocol + cross-platform intel + multi-framing

Usage:
    python -m src.backtest.enhanced_harness --count 10
"""

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from src.backtest.harness import (
    BacktestResult,
    ResolvedMarket,
    build_backtest_prompt,
    fetch_resolved_markets,
)
from src.enrichment.cross_platform import CrossPlatformScraper
from src.llm.claude_runner import ClaudeRunner
from src.llm.power_prompt import build_power_prompt
from src.markets.models import MarketInfo, Outcome


@dataclass
class ComparisonResult:
    """Side-by-side comparison of naive vs enhanced on one market."""

    market: ResolvedMarket
    actual_outcome: float  # 1.0 = YES, 0.0 = NO

    # Naive baseline
    naive_prob: Optional[float] = None
    naive_brier: Optional[float] = None
    naive_correct: Optional[bool] = None
    naive_time: float = 0.0

    # Enhanced (superforecaster + cross-platform + multi-framing)
    enhanced_prob: Optional[float] = None
    enhanced_brier: Optional[float] = None
    enhanced_correct: Optional[bool] = None
    enhanced_time: float = 0.0

    # Components
    base_rate: Optional[float] = None
    cross_platform_avg: Optional[float] = None
    disagreement_score: Optional[float] = None
    framing_analyst: Optional[float] = None
    framing_expert: Optional[float] = None
    framing_contrarian: Optional[float] = None


async def run_naive(
    market: ResolvedMarket,
    runner: ClaudeRunner,
) -> tuple[Optional[float], float]:
    """Run naive single-call evaluation. Returns (probability, time)."""
    start = time.monotonic()
    prompt = build_backtest_prompt(market)
    output = await runner.evaluate(prompt)
    elapsed = time.monotonic() - start
    return (output.probability if output else None, elapsed)


def _market_to_info(market: ResolvedMarket) -> MarketInfo:
    """Convert ResolvedMarket to MarketInfo for prompt building."""
    return MarketInfo(
        id=market.id,
        question=market.question,
        category=market.category,
        volume=market.volume,
        outcomes=[
            Outcome(name=market.outcomes[0], clob_token_id="", price=None),
            Outcome(name=market.outcomes[1], clob_token_id="", price=None),
        ] if len(market.outcomes) >= 2 else [],
        description=market.description,
        resolution_source=market.resolution_source,
    )


async def run_enhanced(
    market: ResolvedMarket,
    runner: ClaudeRunner,
    scraper: CrossPlatformScraper,
) -> ComparisonResult:
    """Run the refactored enhanced pipeline: cross-platform data + power prompt.

    1 data gather + 1 Claude call (vs 6 calls before).
    """
    actual = 1.0 if market.yes_won else 0.0
    result = ComparisonResult(market=market, actual_outcome=actual)

    start = time.monotonic()

    # 1. Gather cross-platform intelligence (independent signals)
    cross_intel = await scraper.gather(market.question, market.category)
    result.cross_platform_avg = cross_intel.average_probability

    # 2. Build power prompt with all context in one call
    market_info = _market_to_info(market)
    prompt = build_power_prompt(market_info, cross_intel)

    # 3. Single Claude evaluation with full context
    output = await runner.evaluate(prompt)

    result.enhanced_time = time.monotonic() - start

    if output:
        result.enhanced_prob = output.probability
        result.enhanced_brier = (output.probability - actual) ** 2
        result.enhanced_correct = (
            (output.probability > 0.5 and actual == 1.0)
            or (output.probability < 0.5 and actual == 0.0)
        )

    return result


async def main():
    parser = argparse.ArgumentParser(
        description="A/B test naive vs enhanced prediction pipeline"
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--min-volume", type=float, default=500_000)
    parser.add_argument("--closed-after", type=str, default=None, help="Only markets closed after this date (e.g., 2025-06)")
    args = parser.parse_args()

    print(f"Fetching {args.count} resolved markets...")
    markets = await fetch_resolved_markets(count=args.count, min_volume=args.min_volume, closed_after=args.closed_after)
    print(f"Found {len(markets)} markets\n")

    if not markets:
        return

    runner = ClaudeRunner(timeout=90, max_retries=1)
    scraper = CrossPlatformScraper()

    results: list[ComparisonResult] = []

    for i, market in enumerate(markets):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(markets)}] {market.question[:65]}")
        actual = "YES" if market.yes_won else "NO"
        print(f"  Actual outcome: {actual}")

        result = ComparisonResult(
            market=market,
            actual_outcome=1.0 if market.yes_won else 0.0,
        )

        # Run naive
        print("  Running naive baseline...", end="", flush=True)
        naive_prob, naive_time = await run_naive(market, runner)
        result.naive_prob = naive_prob
        result.naive_time = naive_time
        if naive_prob is not None:
            result.naive_brier = (naive_prob - result.actual_outcome) ** 2
            result.naive_correct = (
                (naive_prob > 0.5 and result.actual_outcome == 1.0)
                or (naive_prob < 0.5 and result.actual_outcome == 0.0)
            )
            direction = "OK" if result.naive_correct else "MISS"
            print(f" {naive_prob:.0%} ({direction}) [{naive_time:.0f}s]")
        else:
            print(" FAILED")

        # Run enhanced (power prompt: 1 data gather + 1 Claude call)
        print("  Running power prompt...", end="", flush=True)
        enhanced = await run_enhanced(market, runner, scraper)
        result.enhanced_prob = enhanced.enhanced_prob
        result.enhanced_brier = enhanced.enhanced_brier
        result.enhanced_correct = enhanced.enhanced_correct
        result.enhanced_time = enhanced.enhanced_time
        result.cross_platform_avg = enhanced.cross_platform_avg

        if result.enhanced_prob is not None:
            direction = "OK" if result.enhanced_correct else "MISS"
            xplat = f" | xplat={result.cross_platform_avg:.0%}" if result.cross_platform_avg is not None else ""
            print(f" {result.enhanced_prob:.0%} ({direction}) [{result.enhanced_time:.0f}s]{xplat}")
        else:
            print(" FAILED")

        results.append(result)

    await scraper.close()

    # Print comparison summary
    print("\n" + "=" * 70)
    print("A/B COMPARISON: NAIVE vs ENHANCED")
    print("=" * 70)

    naive_valid = [r for r in results if r.naive_prob is not None]
    enhanced_valid = [r for r in results if r.enhanced_prob is not None]

    if naive_valid:
        naive_briers = [r.naive_brier for r in naive_valid if r.naive_brier is not None]
        naive_correct = sum(1 for r in naive_valid if r.naive_correct)
        print(f"\n  NAIVE BASELINE:")
        print(f"    Direction accuracy: {naive_correct}/{len(naive_valid)} ({naive_correct/len(naive_valid):.0%})")
        print(f"    Avg Brier score:    {sum(naive_briers)/len(naive_briers):.4f}")
        print(f"    Avg eval time:      {sum(r.naive_time for r in naive_valid)/len(naive_valid):.0f}s")

    if enhanced_valid:
        enhanced_briers = [r.enhanced_brier for r in enhanced_valid if r.enhanced_brier is not None]
        enhanced_correct = sum(1 for r in enhanced_valid if r.enhanced_correct)
        print(f"\n  ENHANCED PIPELINE:")
        print(f"    Direction accuracy: {enhanced_correct}/{len(enhanced_valid)} ({enhanced_correct/len(enhanced_valid):.0%})")
        print(f"    Avg Brier score:    {sum(enhanced_briers)/len(enhanced_briers):.4f}")
        print(f"    Avg eval time:      {sum(r.enhanced_time for r in enhanced_valid)/len(enhanced_valid):.0f}s")

    if naive_valid and enhanced_valid:
        naive_avg_brier = sum(r.naive_brier for r in naive_valid if r.naive_brier is not None) / len(naive_valid)
        enhanced_avg_brier = sum(r.enhanced_brier for r in enhanced_valid if r.enhanced_brier is not None) / len(enhanced_valid)
        improvement = ((naive_avg_brier - enhanced_avg_brier) / naive_avg_brier * 100) if naive_avg_brier > 0 else 0

        print(f"\n  IMPROVEMENT:")
        print(f"    Brier score change: {improvement:+.1f}%")
        print(f"    {'ENHANCED WINS' if enhanced_avg_brier < naive_avg_brier else 'NAIVE WINS'}")

    # Per-market comparison table
    print(f"\n  {'Question':<40} | {'Naive':>6} | {'Enhanced':>8} | {'Actual':>6} | {'Winner':>7}")
    print("  " + "-" * 80)
    for r in results:
        q = r.market.question[:38]
        actual = "YES" if r.actual_outcome == 1.0 else "NO"
        n = f"{r.naive_prob:.0%}" if r.naive_prob is not None else "FAIL"
        e = f"{r.enhanced_prob:.0%}" if r.enhanced_prob is not None else "FAIL"

        winner = ""
        if r.naive_brier is not None and r.enhanced_brier is not None:
            if r.enhanced_brier < r.naive_brier:
                winner = "ENH"
            elif r.naive_brier < r.enhanced_brier:
                winner = "NAIVE"
            else:
                winner = "TIE"

        print(f"  {q:<40} | {n:>6} | {e:>8} | {actual:>6} | {winner:>7}")


if __name__ == "__main__":
    asyncio.run(main())
