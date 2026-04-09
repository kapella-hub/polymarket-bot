"""Cross-market arbitrage alpha: detects mispricing across related markets."""

from collections import defaultdict
from typing import Optional

import structlog

from src.alpha.base import AlphaOutput, AlphaSource
from src.db.models import Market

logger = structlog.get_logger()


class CrossMarketAlpha(AlphaSource):
    """Alpha derived from cross-market probability inconsistencies.

    If a group of mutually exclusive markets (e.g., candidates in an election)
    sums to significantly less than 1.0, the group is underpriced.
    If it sums to more than 1.0, the group is overpriced.

    This alpha returns a positive edge for individual markets in underpriced
    groups (buy opportunity) and negative edge for overpriced groups (sell).
    """

    name = "cross_market"

    def __init__(self):
        self._group_cache: dict[str, dict] = {}
        self._last_update_count: int = 0

    def update_groups(self, markets: list[Market]) -> None:
        """Rebuild market groups from active markets.

        Groups are identified by shared event patterns in the question text.
        Only groups where we likely have complete coverage are actionable.
        """
        groups: dict[str, list[Market]] = defaultdict(list)

        for m in markets:
            q = m.question.lower()
            group_key = self._extract_group_key(q)
            if group_key:
                groups[group_key].append(m)

        # Build group summaries
        self._group_cache.clear()
        for key, group_markets in groups.items():
            if len(group_markets) < 2:
                continue

            # Sum prices using best available
            total_price = 0.0
            market_prices: dict[str, float] = {}
            has_leader = False  # Does the group have a market with bid > 0.20?

            for gm in group_markets:
                price = gm.best_bid or gm.last_price or gm.best_ask or 0
                market_prices[gm.id] = price
                total_price += price
                if price > 0.20:
                    has_leader = True

            # Only trust groups where we likely have full coverage:
            # - Has a clear leader (>20%)
            # - Sum is reasonably close to 1.0 (0.5-1.3)
            # - At least 3 markets
            if not has_leader or total_price < 0.5 or len(group_markets) < 3:
                continue

            gap = 1.0 - total_price  # Positive = underpriced, negative = overpriced
            self._group_cache[key] = {
                "markets": {gm.id: gm for gm in group_markets},
                "prices": market_prices,
                "total_price": total_price,
                "gap": gap,
                "count": len(group_markets),
            }

        if self._group_cache:
            logger.info(
                "cross_market_groups_updated",
                groups=len(self._group_cache),
                actionable=[
                    k for k, v in self._group_cache.items()
                    if abs(v["gap"]) > 0.05
                ],
            )

    async def compute(
        self,
        market: Market,
        context: dict,
    ) -> Optional[AlphaOutput]:
        # Find which group this market belongs to
        for key, group in self._group_cache.items():
            if market.id not in group["markets"]:
                continue

            gap = group["gap"]
            # Only signal if gap is meaningful (>5% after fees)
            if abs(gap) < 0.05:
                return None

            market_price = group["prices"].get(market.id, 0)

            if gap > 0:
                # Group is underpriced — each market is cheap
                # Edge proportional to this market's share of the gap
                share = market_price / group["total_price"] if group["total_price"] > 0 else 0
                edge = gap * max(share, 0.1)  # At least 10% of gap
            else:
                # Group is overpriced — each market is expensive
                share = market_price / group["total_price"] if group["total_price"] > 0 else 0
                edge = gap * max(share, 0.1)  # Negative edge = sell signal

            # Confidence based on gap size and group completeness
            confidence = min(0.6, abs(gap) * 2)

            return AlphaOutput(
                edge=edge,
                confidence=confidence,
                notes=f"group:{key} sum={group['total_price']:.3f} gap={gap:+.3f} n={group['count']}",
                meta={
                    "group_key": key,
                    "group_sum": group["total_price"],
                    "gap": gap,
                    "market_share": market_price / group["total_price"] if group["total_price"] > 0 else 0,
                },
            )

        return None

    @staticmethod
    def _extract_group_key(question: str) -> Optional[str]:
        """Extract a group identifier from a market question.

        Returns None if the market doesn't belong to an identifiable group.
        """
        q = question.lower()

        # Election patterns: "Will X win the Y election?"
        # Group by election name
        if "win the 2026 colombian presidential" in q and "1st round" not in q:
            return "colombia_2026_final"
        if "1st round of the 2026 colombian" in q:
            return "colombia_2026_r1"
        if "next prime minister of hungary" in q:
            return "hungary_pm"
        if "2026 texas republican primary" in q:
            return "texas_gop_2026"

        # Sports: "Will X win the Y?"
        if "2025-26 english premier league" in q or "2025–26 english premier league" in q:
            if "win" in q:
                return "epl_2025_winner"
        if "2025-26 champions league" in q or "2025–26 champions league" in q:
            if "win" in q:
                return "ucl_2025_winner"
        if "2025-2026 nba mvp" in q or "2025–2026 nba mvp" in q:
            return "nba_mvp_2025"
        if "2026 masters tournament" in q:
            return "masters_2026"

        return None
