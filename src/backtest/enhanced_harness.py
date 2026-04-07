"""Enhanced backtest: compares naive vs full innovation stack.

Runs both approaches on the same resolved markets and compares:
- Naive: single Claude call (baseline)
- Enhanced: superforecaster protocol + cross-platform intel + multi-framing

Usage:
    python -m src.backtest.enhanced_harness --count 10
"""

import argparse
import asyncio
import io
import sys
import time
from dataclasses import dataclass
from typing import Optional

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from src.backtest.harness import (
    BacktestResult,
    ResolvedMarket,
    build_backtest_prompt,
    fetch_resolved_markets,
)
from src.enrichment.cross_platform import CrossPlatformScraper
from src.llm.claude_runner import ClaudeRunner
from src.llm.multi_framing import MultiFramingAnalyzer
from src.llm.superforecaster import SuperforecasterProtocol


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


async def run_enhanced(
    market: ResolvedMarket,
    protocol: SuperforecasterProtocol,
    scraper: CrossPlatformScraper,
    framing: MultiFramingAnalyzer,
) -> ComparisonResult:
    """Run the full enhanced pipeline."""
    actual = 1.0 if market.yes_won else 0.0
    result = ComparisonResult(market=market, actual_outcome=actual)

    start = time.monotonic()

    # 1. Cross-platform intelligence (run concurrently with step 1 of superforecaster)
    cross_task = asyncio.create_task(scraper.gather(market.question, market.category))

    # 2. Multi-framing analysis (run concurrently)
    framing_task = asyncio.create_task(framing.evaluate(market.question))

    # 3. Superforecaster protocol (sequential 3-step chain)
    cross_intel = await cross_task
    cross_context = cross_intel.format_for_prompt()
    result.cross_platform_avg = cross_intel.average_probability

    forecast = await protocol.evaluate(
        question=market.question,
        description=market.description or "",
        cross_platform_context=cross_context,
    )

    framing_result = await framing_task

    result.enhanced_time = time.monotonic() - start

    # Populate component details
    if forecast:
        result.base_rate = forecast.base_rate

    if framing_result.has_results:
        result.disagreement_score = framing_result.disagreement_score
        for f in framing_result.framings:
            if f.framing == "analyst":
                result.framing_analyst = f.probability
            elif f.framing == "expert":
                result.framing_expert = f.probability
            elif f.framing == "contrarian":
                result.framing_contrarian = f.probability

    # Combine: weighted average of superforecaster and multi-framing consensus
    probs = []
    weights = []

    if forecast and forecast.probability is not None:
        probs.append(forecast.probability)
        weights.append(forecast.confidence * 2)  # Superforecaster gets 2x weight

    if framing_result.consensus_probability is not None:
        probs.append(framing_result.consensus_probability)
        # Lower weight when disagreement is high (signal is noisier)
        framing_weight = 1.0 * (1.0 - framing_result.disagreement_score * 0.5)
        weights.append(framing_weight)

    if probs and sum(weights) > 0:
        result.enhanced_prob = sum(p * w for p, w in zip(probs, weights)) / sum(weights)
        result.enhanced_prob = max(0.0, min(1.0, result.enhanced_prob))
        result.enhanced_brier = (result.enhanced_prob - actual) ** 2
        result.enhanced_correct = (
            (result.enhanced_prob > 0.5 and actual == 1.0)
            or (result.enhanced_prob < 0.5 and actual == 0.0)
        )

    return result


async def main():
    parser = argparse.ArgumentParser(
        description="A/B test naive vs enhanced prediction pipeline"
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--min-volume", type=float, default=500_000)
    args = parser.parse_args()

    print(f"Fetching {args.count} resolved markets...")
    markets = await fetch_resolved_markets(count=args.count, min_volume=args.min_volume)
    print(f"Found {len(markets)} markets\n")

    if not markets:
        return

    runner = ClaudeRunner(timeout=90, max_retries=1)
    protocol = SuperforecasterProtocol(runner)
    scraper = CrossPlatformScraper()
    framing_analyzer = MultiFramingAnalyzer(runner)

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

        # Run enhanced
        print("  Running enhanced pipeline...", end="", flush=True)
        enhanced = await run_enhanced(market, protocol, scraper, framing_analyzer)
        result.enhanced_prob = enhanced.enhanced_prob
        result.enhanced_brier = enhanced.enhanced_brier
        result.enhanced_correct = enhanced.enhanced_correct
        result.enhanced_time = enhanced.enhanced_time
        result.base_rate = enhanced.base_rate
        result.cross_platform_avg = enhanced.cross_platform_avg
        result.disagreement_score = enhanced.disagreement_score
        result.framing_analyst = enhanced.framing_analyst
        result.framing_expert = enhanced.framing_expert
        result.framing_contrarian = enhanced.framing_contrarian

        if result.enhanced_prob is not None:
            direction = "OK" if result.enhanced_correct else "MISS"
            print(f" {result.enhanced_prob:.0%} ({direction}) [{result.enhanced_time:.0f}s]")
        else:
            print(" FAILED")

        # Show components
        parts = []
        if result.base_rate is not None:
            parts.append(f"base_rate={result.base_rate:.0%}")
        if result.cross_platform_avg is not None:
            parts.append(f"xplat={result.cross_platform_avg:.0%}")
        if result.framing_analyst is not None:
            parts.append(f"analyst={result.framing_analyst:.0%}")
        if result.framing_expert is not None:
            parts.append(f"expert={result.framing_expert:.0%}")
        if result.framing_contrarian is not None:
            parts.append(f"contrarian={result.framing_contrarian:.0%}")
        if result.disagreement_score is not None:
            parts.append(f"disagree={result.disagreement_score:.2f}")
        if parts:
            print(f"  Components: {' | '.join(parts)}")

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
