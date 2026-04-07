"""Configurable market filters for discovery pipeline."""

from datetime import datetime, timezone
from typing import Optional

import structlog

from src.config import settings
from src.markets.models import MarketInfo

logger = structlog.get_logger()


class MarketFilter:
    """Filters markets based on configurable criteria."""

    def __init__(
        self,
        min_volume: Optional[float] = None,
        min_liquidity: Optional[float] = None,
        min_spread: Optional[float] = None,
        max_end_date_days: Optional[int] = None,
        min_end_date_hours: Optional[int] = None,
        category_whitelist: Optional[list[str]] = None,
        category_blacklist: Optional[list[str]] = None,
    ):
        self.min_volume = min_volume or settings.min_volume_usd
        self.min_liquidity = min_liquidity or settings.min_liquidity_usd
        self.min_spread = min_spread or settings.min_spread
        self.max_end_date_days = max_end_date_days or settings.max_end_date_days
        self.min_end_date_hours = min_end_date_hours or settings.min_end_date_hours
        self.category_whitelist = category_whitelist
        self.category_blacklist = category_blacklist

    def apply(self, markets: list[MarketInfo]) -> list[MarketInfo]:
        """Filter markets and return those passing all criteria."""
        passed = []
        for m in markets:
            reason = self._check(m)
            if reason is None:
                passed.append(m)
            else:
                logger.debug("market_filtered", market_id=m.id, reason=reason)
        logger.info(
            "markets_filtered",
            total=len(markets),
            passed=len(passed),
            rejected=len(markets) - len(passed),
        )
        return passed

    def _check(self, m: MarketInfo) -> Optional[str]:
        """Return rejection reason, or None if market passes."""
        if not m.active:
            return "inactive"

        if m.volume < self.min_volume:
            return f"low_volume ({m.volume:.0f} < {self.min_volume:.0f})"

        if m.liquidity < self.min_liquidity:
            return f"low_liquidity ({m.liquidity:.0f} < {self.min_liquidity:.0f})"

        if m.spread is not None and m.spread < self.min_spread:
            return f"tight_spread ({m.spread:.4f} < {self.min_spread:.4f})"

        now = datetime.now(timezone.utc)
        if m.end_date:
            hours_until_end = (m.end_date - now).total_seconds() / 3600
            if hours_until_end < self.min_end_date_hours:
                return f"too_close_to_resolution ({hours_until_end:.1f}h)"
            days_until_end = hours_until_end / 24
            if days_until_end > self.max_end_date_days:
                return f"too_far_out ({days_until_end:.0f}d)"

        if self.category_whitelist and m.category:
            if m.category.lower() not in [c.lower() for c in self.category_whitelist]:
                return f"category_not_whitelisted ({m.category})"

        if self.category_blacklist and m.category:
            if m.category.lower() in [c.lower() for c in self.category_blacklist]:
                return f"category_blacklisted ({m.category})"

        if len(m.outcomes) < 2:
            return "insufficient_outcomes"

        return None
