"""Gamma API client for market discovery."""

from datetime import datetime, timezone
from typing import Optional

import httpx
import structlog

from src.config import settings
from src.markets.models import MarketInfo, Outcome

logger = structlog.get_logger()


class GammaAPIClient:
    """Fetches active markets from the Polymarket Gamma API."""

    def __init__(self, base_url: Optional[str] = None):
        self._base_url = base_url or settings.gamma_api_host
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=30.0,
            headers={"Accept": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def fetch_active_markets(
        self,
        limit: int = 500,
        offset: int = 0,
    ) -> list[MarketInfo]:
        """Fetch active markets from Gamma API with pagination."""
        all_markets: list[MarketInfo] = []
        current_offset = offset

        while True:
            params = {
                "active": "true",
                "closed": "false",
                "limit": min(limit, 100),  # Gamma API caps at 100 per request
                "offset": current_offset,
            }

            try:
                resp = await self._client.get("/markets", params=params)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as e:
                logger.error("gamma_api_error", error=str(e), offset=current_offset)
                break

            if not data:
                break

            for raw in data:
                market = self._parse_market(raw)
                if market:
                    all_markets.append(market)

            if len(data) < params["limit"]:
                break  # Last page

            current_offset += len(data)
            if len(all_markets) >= limit:
                break

        logger.info("markets_discovered", count=len(all_markets))
        return all_markets

    def _parse_market(self, raw: dict) -> Optional[MarketInfo]:
        """Parse a raw Gamma API market response into MarketInfo."""
        try:
            condition_id = raw.get("conditionId", raw.get("condition_id", ""))
            if not condition_id:
                return None

            # Parse outcomes and token IDs
            outcomes = []
            clob_token_ids = raw.get("clobTokenIds", [])
            outcome_names = raw.get("outcomes", [])
            outcome_prices = raw.get("outcomePrices", [])

            for i, token_id in enumerate(clob_token_ids):
                name = outcome_names[i] if i < len(outcome_names) else f"Outcome {i}"
                price = (
                    float(outcome_prices[i])
                    if i < len(outcome_prices) and outcome_prices[i]
                    else None
                )
                outcomes.append(Outcome(name=name, clob_token_id=str(token_id), price=price))

            if len(outcomes) < 2:
                return None  # Need at least YES/NO

            # Parse end date
            end_date = None
            end_str = raw.get("endDate") or raw.get("end_date_iso")
            if end_str:
                try:
                    end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    pass

            # Parse prices
            best_bid = _safe_float(raw.get("bestBid"))
            best_ask = _safe_float(raw.get("bestAsk"))
            spread = None
            if best_bid is not None and best_ask is not None:
                spread = best_ask - best_bid

            return MarketInfo(
                id=condition_id,
                question=raw.get("question", ""),
                category=raw.get("category"),
                end_date=end_date,
                volume=_safe_float(raw.get("volume")) or 0.0,
                liquidity=_safe_float(raw.get("liquidity")) or 0.0,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                outcomes=outcomes,
                description=raw.get("description"),
                resolution_source=raw.get("resolutionSource"),
                tags=raw.get("tags"),
                active=raw.get("active", True),
            )
        except Exception as e:
            logger.warning("market_parse_error", error=str(e), raw_id=raw.get("id"))
            return None


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None
