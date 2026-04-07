"""Power prompt: single comprehensive Claude call with all context.

Learned from A/B testing: one well-informed call beats six fragmented ones.
This prompt embeds base-rate reasoning, cross-platform data, and adversarial
checking into a single evaluation, giving Claude full context to reason with.
"""

from datetime import datetime, timezone
from typing import Optional

from src.enrichment.cross_platform import CrossPlatformIntel
from src.markets.models import MarketInfo


def build_power_prompt(
    market: MarketInfo,
    cross_platform: Optional[CrossPlatformIntel] = None,
    calibration_note: str = "",
) -> str:
    """Build a single comprehensive evaluation prompt.

    Combines:
    - Market question and context
    - Cross-platform intelligence (independent signals)
    - Base rate reasoning instructions
    - Adversarial self-check instructions
    - Calibration adjustments from backtest data
    """

    # Cross-platform section — only include if we have high-confidence matches
    xplat_block = ""
    if cross_platform and cross_platform.has_data:
        # Only include cross-platform data if disagreement between sources
        # is reasonable (< 40%). Huge disagreement likely means bad match.
        if cross_platform.max_disagreement is None or cross_platform.max_disagreement < 0.4:
            xplat_block = f"""
## Cross-Platform Intelligence (other prediction platforms)
{cross_platform.format_for_prompt()}

CAUTION: These may not be exact matches to this market question. Only use
them if the questions are truly the same. If the match seems wrong, ignore
this section entirely and form your own estimate."""

    # Calibration section
    cal_block = ""
    if calibration_note:
        cal_block = f"""
## Calibration Note
{calibration_note}"""

    # Time remaining
    time_block = ""
    if market.end_date:
        now = datetime.now(timezone.utc)
        delta = market.end_date - now
        days = delta.days
        hours = (delta.seconds // 3600) if days >= 0 else 0
        if days > 0:
            time_block = f"**Time remaining:** {days} days, {hours} hours"
        elif days == 0:
            time_block = f"**Time remaining:** {hours} hours"
        else:
            time_block = "**Status:** Resolution date has passed"

    yes_price = market.yes_price
    price_block = ""
    if yes_price is not None:
        price_block = f"**Current Polymarket price (YES):** ${yes_price:.2f} (implying {yes_price:.0%} probability)"

    return f"""You are an elite prediction market analyst. Estimate the TRUE probability of the YES outcome.

## Your Process (do all of this internally, then output ONLY JSON):

1. ANCHOR: What is the base rate for events like this? What reference class does it belong to? Start from the outside view.

2. UPDATE: What specific evidence moves the probability AWAY from the base rate? List factors for and against.

3. CROSS-CHECK: If cross-platform data is provided below, compare your estimate against it. But ONLY if the matched question is truly the same — text-search matches can be wrong. Trust your own analysis over a bad match.

4. ADVERSARIAL CHECK: What is your biggest source of overconfidence? What scenario are you underweighting? Adjust if needed.

5. FINAL: Output your calibrated probability.

## Rules
- Do NOT anchor to the Polymarket price. Form your own estimate first.
- Weight cross-platform data heavily — these are independent signals from informed traders.
- When uncertain, stay closer to the base rate. Don't overfit to narratives.
- A 70% estimate should be right ~70% of the time. Be calibrated, not confident.

## Market
**Question:** {market.question}

{price_block}
{time_block}

**Category:** {market.category or "Uncategorized"}
**Volume:** ${market.volume:,.0f}
**Liquidity:** ${market.liquidity:,.0f}

{f'**Description:** {market.description}' if market.description else ''}
{f'**Resolution source:** {market.resolution_source}' if market.resolution_source else ''}
{xplat_block}
{cal_block}

## Output ONLY valid JSON — no other text:
{{"probability": 0.XX, "confidence": 0.XX, "reasoning": "your full reasoning in 2-3 sentences", "key_factors": ["factor 1", "factor 2", "factor 3"]}}"""
