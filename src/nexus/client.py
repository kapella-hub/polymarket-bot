"""NexusStack integration client for the trading bot.

Wraps NexusCortex, NexusSentinel, and NexusRelay into a single interface
that the bot calls at key lifecycle points:

- Cortex: self-calibration memory (learn from outcomes, recall past accuracy)
- Sentinel: trading event monitoring (trades, risk events, health)
- Relay: multi-bot coordination (position sharing, evaluation sharing)

All calls are fire-and-forget with error handling — NexusStack being
unavailable should never block trading operations.
"""

import json
import os
from typing import Optional

import httpx
import structlog

from src.config import settings

logger = structlog.get_logger()

# NexusStack service URLs — defaults match VPS deployment
CORTEX_URL = os.environ.get("NEXUS_CORTEX_URL", "http://localhost:8100")
SENTINEL_URL = os.environ.get("NEXUS_SENTINEL_URL", "http://localhost:8060")
RELAY_URL = os.environ.get("NEXUS_RELAY_URL", "http://localhost:8050")


class NexusClient:
    """Unified client for NexusStack services."""

    def __init__(
        self,
        cortex_url: str = CORTEX_URL,
        sentinel_url: str = SENTINEL_URL,
        relay_url: str = RELAY_URL,
        agent_id: str = "polymarket-bot",
    ):
        self._cortex = cortex_url
        self._sentinel = sentinel_url
        self._relay = relay_url
        self._agent_id = agent_id
        self._client = httpx.AsyncClient(timeout=10)
        self._enabled = True

    async def close(self) -> None:
        await self._client.aclose()

    # ─── Cortex: Self-Calibration Memory ─────────────────────────────

    async def learn_evaluation(
        self,
        market_id: str,
        question: str,
        estimated_prob: float,
        market_price: float,
        confidence: float,
        category: Optional[str] = None,
    ) -> None:
        """Record a market evaluation for future calibration."""
        await self._cortex_learn(
            action=f"Evaluated market: {question[:100]}",
            outcome=(
                f"Estimated {estimated_prob:.0%} YES probability "
                f"(market price: {market_price:.0%}, "
                f"edge: {estimated_prob - market_price:+.0%}, "
                f"confidence: {confidence:.0%})"
            ),
            tags=["evaluation", category or "uncategorized", market_id[:16]],
            domain="trading",
        )

    async def learn_resolution(
        self,
        market_id: str,
        question: str,
        estimated_prob: float,
        actual_outcome: str,
        pnl: float,
        category: Optional[str] = None,
    ) -> None:
        """Record a market resolution for calibration learning.

        This is where the bot learns from its mistakes and successes.
        """
        outcome_val = 1.0 if actual_outcome.lower() in ("yes", "true", "1") else 0.0
        error = estimated_prob - outcome_val
        direction = "correct" if (estimated_prob > 0.5) == (outcome_val == 1.0) else "wrong"

        resolution = None
        if abs(error) > 0.2:
            resolution = (
                f"Large calibration error ({error:+.0%}). "
                f"Review whether base rate anchoring or "
                f"{'overconfidence' if abs(error) > 0.3 else 'underconfidence'} "
                f"was the issue for {category or 'this'} category markets."
            )

        await self._cortex_learn(
            action=f"Market resolved: {question[:100]}",
            outcome=(
                f"Resolved {actual_outcome}. "
                f"My estimate was {estimated_prob:.0%} ({direction}). "
                f"Error: {error:+.0%}. PnL: ${pnl:+.2f}. "
                f"Category: {category or 'unknown'}."
            ),
            resolution=resolution,
            tags=["resolution", direction, category or "uncategorized"],
            domain="trading",
        )

    async def recall_calibration(
        self,
        question: str,
        category: Optional[str] = None,
    ) -> Optional[str]:
        """Recall past performance on similar markets for calibration.

        Returns a calibration note string to inject into the prompt,
        or None if no relevant memories exist.
        """
        task = (
            f"Evaluating prediction market: {question[:100]}. "
            f"Category: {category or 'unknown'}. "
            f"What was my past accuracy on similar markets?"
        )
        return await self._cortex_recall(task, tags=["resolution", category or "uncategorized"])

    async def learn_trade(
        self,
        market_id: str,
        question: str,
        side: str,
        size: float,
        price: float,
        strategy: str,
        edge: float,
    ) -> None:
        """Record a trade execution."""
        await self._cortex_learn(
            action=f"Traded {side} ${size:.2f} on: {question[:80]}",
            outcome=(
                f"Price: {price:.4f}, Edge: {edge:+.0%}, "
                f"Strategy: {strategy}"
            ),
            tags=["trade", side, strategy],
            domain="trading",
        )

    # ─── Sentinel: Trading Event Monitoring ──────────────────────────

    async def event_trade_executed(
        self,
        market_id: str,
        side: str,
        size: float,
        price: float,
        edge: float,
    ) -> None:
        """Push trade execution event to Sentinel."""
        await self._sentinel_push(
            event_type="trade.executed",
            summary=f"{side.upper()} ${size:.2f} at {price:.4f} (edge: {edge:+.0%}) market:{market_id[:16]}",
            severity="info",
            tags=["trade", side],
        )

    async def event_risk_triggered(
        self,
        reason: str,
        details: str = "",
    ) -> None:
        """Push risk limit trigger event to Sentinel."""
        await self._sentinel_push(
            event_type="risk.triggered",
            summary=f"Risk limit: {reason}. {details}",
            severity="warning",
            tags=["risk", reason.split("_")[0]],
        )

    async def event_kill_switch(self, activated: bool) -> None:
        """Push kill switch event to Sentinel."""
        action = "activated" if activated else "deactivated"
        await self._sentinel_push(
            event_type=f"killswitch.{action}",
            summary=f"Kill switch {action}",
            severity="critical" if activated else "info",
            tags=["killswitch"],
        )

    async def event_signal_stale(self, stale_count: int, total_count: int) -> None:
        """Push signal staleness warning to Sentinel."""
        if stale_count > total_count * 0.5:
            await self._sentinel_push(
                event_type="signal.stale",
                summary=f"{stale_count}/{total_count} market signals are stale",
                severity="warning",
                tags=["signal", "staleness"],
            )

    async def event_bot_started(self, mode: str) -> None:
        await self._sentinel_push(
            event_type="bot.started",
            summary=f"Polymarket bot started in {mode} mode",
            severity="info",
            tags=["lifecycle"],
        )

    async def event_bot_stopped(self) -> None:
        await self._sentinel_push(
            event_type="bot.stopped",
            summary="Polymarket bot stopped",
            severity="info",
            tags=["lifecycle"],
        )

    async def event_drawdown(self, drawdown_pct: float, threshold: float) -> None:
        await self._sentinel_push(
            event_type="risk.drawdown",
            summary=f"Portfolio drawdown {drawdown_pct:.1%} (threshold: {threshold:.1%})",
            severity="error" if drawdown_pct > threshold else "warning",
            tags=["risk", "drawdown"],
        )

    # ─── Relay: Multi-Bot Coordination ───────────────────────────────

    async def broadcast_position_update(
        self,
        market_id: str,
        outcome: str,
        size: float,
        exposure_usd: float,
    ) -> None:
        """Broadcast position update for cross-bot coordination."""
        await self._relay_broadcast(
            channel="positions",
            content=json.dumps({
                "bot": self._agent_id,
                "market_id": market_id,
                "outcome": outcome,
                "size": size,
                "exposure_usd": exposure_usd,
            }),
            tags=["position"],
        )

    async def broadcast_evaluation(
        self,
        market_id: str,
        question: str,
        probability: float,
        confidence: float,
    ) -> None:
        """Share a market evaluation with other bot instances."""
        await self._relay_broadcast(
            channel="evaluations",
            content=json.dumps({
                "bot": self._agent_id,
                "market_id": market_id,
                "question": question[:100],
                "probability": probability,
                "confidence": confidence,
            }),
            tags=["evaluation"],
        )

    async def post_daily_summary(self, summary: dict) -> None:
        """Post daily P&L summary to the bulletin board."""
        await self._relay_post(
            content=json.dumps({
                "type": "daily_summary",
                "bot": self._agent_id,
                **summary,
            }),
            tags=["summary", "daily"],
        )

    # ─── Internal HTTP helpers ───────────────────────────────────────

    async def _cortex_learn(
        self, action: str, outcome: str,
        resolution: Optional[str] = None,
        tags: Optional[list[str]] = None,
        domain: str = "general",
    ) -> None:
        if not self._enabled:
            return
        try:
            await self._client.post(
                f"{self._cortex}/memory/learn",
                json={
                    "action": action,
                    "outcome": outcome,
                    "resolution": resolution,
                    "tags": tags,
                    "domain": domain,
                    "agent_id": self._agent_id,
                },
            )
        except Exception as e:
            logger.debug("nexus_cortex_learn_failed", error=str(e))

    async def _cortex_recall(
        self, task: str, tags: Optional[list[str]] = None,
    ) -> Optional[str]:
        if not self._enabled:
            return None
        try:
            resp = await self._client.post(
                f"{self._cortex}/memory/recall",
                json={
                    "task": task,
                    "tags": tags,
                    "top_k": 5,
                    "agent_id": self._agent_id,
                },
            )
            data = resp.json()
            result = data.get("result", "")
            # Only return if there are actual memories (not "0 results")
            if "0 results" in result or not result.strip():
                return None
            return result
        except Exception as e:
            logger.debug("nexus_cortex_recall_failed", error=str(e))
            return None

    async def _sentinel_push(
        self, event_type: str, summary: str,
        severity: str = "info", tags: Optional[list[str]] = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            await self._client.post(
                f"{self._sentinel}/events",
                json={
                    "source": self._agent_id,
                    "event_type": event_type,
                    "summary": summary,
                    "severity": severity,
                    "tags": tags,
                },
            )
        except Exception as e:
            logger.debug("nexus_sentinel_push_failed", error=str(e))

    async def _relay_broadcast(
        self, channel: str, content: str,
        tags: Optional[list[str]] = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            await self._client.post(
                f"{self._relay}/broadcast",
                json={
                    "channel": channel,
                    "content": content,
                    "sender": self._agent_id,
                    "tags": tags,
                },
            )
        except Exception as e:
            logger.debug("nexus_relay_broadcast_failed", error=str(e))

    async def _relay_post(
        self, content: str, tags: Optional[list[str]] = None,
    ) -> None:
        if not self._enabled:
            return
        try:
            await self._client.post(
                f"{self._relay}/post",
                json={
                    "content": content,
                    "author": self._agent_id,
                    "tags": tags,
                },
            )
        except Exception as e:
            logger.debug("nexus_relay_post_failed", error=str(e))
