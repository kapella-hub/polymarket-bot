"""Cross-platform intelligence scraper.

Fetches probability estimates from other prediction platforms
so Claude can act as a meta-analyst instead of a raw estimator.

Supported sources:
- Metaculus (community forecasts — well-calibrated)
- Manifold Markets (another prediction market)
- Polymarket itself (current market price as reference)

Claude sees all sources and evaluates DISAGREEMENTS between them,
rather than generating a probability from nothing.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Optional

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class PlatformEstimate:
    """A probability estimate from a prediction platform."""

    source: str
    probability: Optional[float]  # 0.0-1.0
    url: Optional[str] = None
    confidence: Optional[str] = None  # "high volume", "few traders", etc.
    raw_data: dict = field(default_factory=dict)


@dataclass
class CrossPlatformIntel:
    """Aggregated intelligence from multiple platforms."""

    estimates: list[PlatformEstimate] = field(default_factory=list)
    search_query: str = ""

    @property
    def has_data(self) -> bool:
        return any(e.probability is not None for e in self.estimates)

    @property
    def average_probability(self) -> Optional[float]:
        probs = [e.probability for e in self.estimates if e.probability is not None]
        return sum(probs) / len(probs) if probs else None

    @property
    def max_disagreement(self) -> Optional[float]:
        probs = [e.probability for e in self.estimates if e.probability is not None]
        return max(probs) - min(probs) if len(probs) >= 2 else None

    def format_for_prompt(self) -> str:
        """Format as context block for Claude prompt injection."""
        if not self.has_data:
            return ""

        lines = []
        for e in self.estimates:
            if e.probability is not None:
                conf = f" ({e.confidence})" if e.confidence else ""
                lines.append(f"- **{e.source}**: {e.probability:.0%}{conf}")

        disagreement = self.max_disagreement
        if disagreement is not None and disagreement > 0.1:
            lines.append(
                f"\n**Notable disagreement: {disagreement:.0%} spread between sources.** "
                "Analyze why these sources disagree and which is most likely correct."
            )

        return "\n".join(lines)


class CrossPlatformScraper:
    """Scrapes prediction data from multiple platforms."""

    def __init__(self):
        self._client = httpx.AsyncClient(timeout=15, follow_redirects=True)

    async def close(self) -> None:
        await self._client.aclose()

    async def gather(self, question: str, category: Optional[str] = None) -> CrossPlatformIntel:
        """Gather estimates from all available platforms for a market question."""
        intel = CrossPlatformIntel(search_query=question)

        # Run all scrapers concurrently
        results = await asyncio.gather(
            self._search_metaculus(question),
            self._search_manifold(question),
            return_exceptions=True,
        )

        for result in results:
            if isinstance(result, PlatformEstimate):
                intel.estimates.append(result)
            elif isinstance(result, Exception):
                logger.debug("cross_platform_scrape_error", error=str(result))

        if intel.has_data:
            logger.info(
                "cross_platform_gathered",
                sources=len([e for e in intel.estimates if e.probability is not None]),
                avg_prob=f"{intel.average_probability:.0%}" if intel.average_probability else "N/A",
                disagreement=f"{intel.max_disagreement:.0%}" if intel.max_disagreement else "N/A",
            )

        return intel

    async def _search_metaculus(self, question: str) -> PlatformEstimate:
        """Search Metaculus for matching questions."""
        try:
            # Use first few words as search query
            search_terms = " ".join(question.split()[:8])
            resp = await self._client.get(
                "https://www.metaculus.com/api2/questions/",
                params={
                    "search": search_terms,
                    "limit": 3,
                    "status": "open",
                    "order_by": "-activity",
                    "type": "forecast",
                },
            )

            if resp.status_code != 200:
                return PlatformEstimate(source="Metaculus", probability=None)

            data = resp.json()
            results = data.get("results", [])

            if not results:
                return PlatformEstimate(source="Metaculus", probability=None)

            # Take the best matching result
            best = results[0]
            community_prediction = best.get("community_prediction", {})

            # Metaculus stores predictions differently based on question type
            prob = None
            if isinstance(community_prediction, dict):
                prob = community_prediction.get("full", {}).get("q2")  # Median
            elif isinstance(community_prediction, (int, float)):
                prob = float(community_prediction)

            forecasters = best.get("number_of_forecasters", 0)
            confidence = f"{forecasters} forecasters" if forecasters else "unknown volume"

            return PlatformEstimate(
                source="Metaculus",
                probability=prob,
                url=f"https://www.metaculus.com/questions/{best.get('id', '')}",
                confidence=confidence,
                raw_data={"title": best.get("title", ""), "forecasters": forecasters},
            )

        except Exception as e:
            logger.debug("metaculus_search_error", error=str(e))
            return PlatformEstimate(source="Metaculus", probability=None)

    async def _search_manifold(self, question: str) -> PlatformEstimate:
        """Search Manifold Markets for matching questions."""
        try:
            search_terms = " ".join(question.split()[:8])
            resp = await self._client.get(
                "https://api.manifold.markets/v0/search-markets",
                params={
                    "term": search_terms,
                    "limit": 3,
                    "filter": "open",
                    "sort": "most-popular",
                },
            )

            if resp.status_code != 200:
                return PlatformEstimate(source="Manifold", probability=None)

            data = resp.json()
            if not data:
                return PlatformEstimate(source="Manifold", probability=None)

            best = data[0]
            prob = best.get("probability")
            volume = best.get("volume", 0)
            confidence = f"${volume:,.0f} volume" if volume else "unknown volume"

            return PlatformEstimate(
                source="Manifold",
                probability=prob,
                url=best.get("url"),
                confidence=confidence,
                raw_data={"question": best.get("question", ""), "volume": volume},
            )

        except Exception as e:
            logger.debug("manifold_search_error", error=str(e))
            return PlatformEstimate(source="Manifold", probability=None)
