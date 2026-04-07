"""Breaking news monitor for temporal edge.

Polls news sources every 60 seconds, matches headlines to active markets,
and triggers immediate re-evaluation when market-moving news is detected.

The edge: markets take 10-30 minutes to fully reprice after breaking news.
A bot that detects and trades within 2 minutes captures the gap.
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

import httpx
import structlog

from src.config import settings

logger = structlog.get_logger()


@dataclass
class NewsItem:
    """A news article that may be relevant to a market."""

    title: str
    source: str
    url: str
    published_at: Optional[datetime] = None
    description: Optional[str] = None

    @property
    def fingerprint(self) -> str:
        """Unique identifier to avoid re-processing."""
        return hashlib.md5(f"{self.title}:{self.url}".encode()).hexdigest()


@dataclass
class NewsMatch:
    """A news item matched to a specific market."""

    news: NewsItem
    market_id: str
    market_question: str
    relevance_keywords: list[str] = field(default_factory=list)


class NewsMonitor:
    """Polls news APIs and matches headlines to active markets.

    Uses free/public RSS-style APIs:
    - GNews API (free tier: 100 req/day)
    - Google News RSS (no API key needed)
    """

    def __init__(
        self,
        on_match: Optional[Callable] = None,
        poll_interval: int = 60,
    ):
        self._client = httpx.AsyncClient(timeout=15, follow_redirects=True)
        self._on_match = on_match  # Callback when news matches a market
        self._poll_interval = poll_interval
        self._seen: set[str] = set()  # Fingerprints of already-processed news
        self._market_keywords: dict[str, list[str]] = {}  # market_id -> keywords

    async def close(self) -> None:
        await self._client.aclose()

    def register_markets(self, markets: list[dict]) -> None:
        """Register active markets and extract keywords for matching.

        Args:
            markets: List of dicts with 'id' and 'question' keys.
        """
        self._market_keywords.clear()
        for m in markets:
            keywords = self._extract_keywords(m["question"])
            if keywords:
                self._market_keywords[m["id"]] = keywords

        logger.info(
            "news_monitor_markets_registered",
            count=len(self._market_keywords),
        )

    async def poll_once(self) -> list[NewsMatch]:
        """Poll news sources and return matches against registered markets."""
        news_items = await self._fetch_news()
        new_items = [n for n in news_items if n.fingerprint not in self._seen]

        if not new_items:
            return []

        # Mark as seen
        for item in new_items:
            self._seen.add(item.fingerprint)

        # Match against markets
        matches = []
        for item in new_items:
            for market_id, keywords in self._market_keywords.items():
                matched_kw = self._match_keywords(item, keywords)
                if matched_kw:
                    matches.append(
                        NewsMatch(
                            news=item,
                            market_id=market_id,
                            market_question="",  # Caller can fill in
                            relevance_keywords=matched_kw,
                        )
                    )

        if matches:
            logger.info(
                "news_matches_found",
                new_articles=len(new_items),
                matches=len(matches),
                markets_affected=len(set(m.market_id for m in matches)),
            )

        # Trim seen set to prevent unbounded growth
        if len(self._seen) > 10000:
            self._seen = set(list(self._seen)[-5000:])

        return matches

    async def _fetch_news(self) -> list[NewsItem]:
        """Fetch latest news from Google News RSS."""
        items = []

        try:
            # Google News top stories RSS (no API key needed)
            resp = await self._client.get(
                "https://news.google.com/rss",
                headers={"User-Agent": "PolymarketBot/1.0"},
            )
            if resp.status_code == 200:
                items.extend(self._parse_rss(resp.text, "Google News"))
        except Exception as e:
            logger.debug("google_news_fetch_error", error=str(e))

        try:
            # Google News search for prediction-market-relevant topics
            for topic in ["politics", "economy", "sports", "crypto"]:
                resp = await self._client.get(
                    f"https://news.google.com/rss/search?q={topic}&hl=en-US",
                    headers={"User-Agent": "PolymarketBot/1.0"},
                )
                if resp.status_code == 200:
                    items.extend(self._parse_rss(resp.text, f"Google News ({topic})"))
        except Exception as e:
            logger.debug("google_news_topic_error", error=str(e))

        return items

    def _parse_rss(self, xml_text: str, source: str) -> list[NewsItem]:
        """Parse RSS XML into NewsItem objects (simple regex parsing)."""
        import re

        items = []
        # Extract <item> blocks
        for match in re.finditer(
            r"<item>(.*?)</item>", xml_text, re.DOTALL
        ):
            block = match.group(1)

            title_match = re.search(r"<title>(.*?)</title>", block, re.DOTALL)
            link_match = re.search(r"<link>(.*?)</link>", block, re.DOTALL)

            if title_match and link_match:
                title = title_match.group(1).strip()
                # Clean CDATA
                title = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", title)
                url = link_match.group(1).strip()

                desc_match = re.search(
                    r"<description>(.*?)</description>", block, re.DOTALL
                )
                desc = None
                if desc_match:
                    desc = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", desc_match.group(1))
                    desc = re.sub(r"<[^>]+>", "", desc).strip()[:300]

                items.append(
                    NewsItem(
                        title=title,
                        source=source,
                        url=url,
                        description=desc,
                    )
                )

        return items[:20]  # Cap per source

    def _extract_keywords(self, question: str) -> list[str]:
        """Extract searchable keywords from a market question."""
        # Remove common question words
        stop_words = {
            "will", "the", "be", "is", "are", "was", "were", "do", "does",
            "did", "has", "have", "had", "can", "could", "would", "should",
            "a", "an", "of", "in", "on", "at", "to", "for", "by", "with",
            "from", "this", "that", "it", "or", "and", "not", "no", "yes",
            "before", "after", "win", "won", "lose", "get", "any", "other",
        }

        words = question.lower().split()
        # Keep proper nouns and significant terms (2+ chars, not stop words)
        keywords = [
            w.strip("?.,!\"'()[]")
            for w in words
            if w.strip("?.,!\"'()[]") not in stop_words and len(w.strip("?.,!\"'()[]")) > 2
        ]

        return keywords[:8]  # Top 8 keywords

    def _match_keywords(self, news: NewsItem, keywords: list[str]) -> list[str]:
        """Check if a news item matches market keywords. Returns matched keywords."""
        text = f"{news.title} {news.description or ''}".lower()
        matched = [kw for kw in keywords if kw in text]

        # Require at least 2 keyword matches to reduce false positives
        if len(matched) >= 2:
            return matched
        return []
