"""Superforecaster Protocol: multi-step structured reasoning.

Replaces naive single-prompt estimation with a 3-step chain
that mimics how elite forecasters actually think (per Tetlock's research):

Step 1 — Base Rate: Anchor to historical frequency of similar events.
Step 2 — Evidence Update: Identify specific factors that move away from base rate.
Step 3 — Adversarial Check: Argue the opposite, catch overconfidence, produce final estimate.

Each step feeds its output into the next, building layered reasoning.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import structlog

from src.llm.claude_runner import ClaudeRunner
from src.llm.parser import LLMSignalOutput, parse_claude_output

logger = structlog.get_logger()


@dataclass
class ForecastChainResult:
    """Full output from the 3-step superforecaster chain."""

    # Final estimate (from step 3)
    probability: float
    confidence: float
    reasoning: str

    # Per-step outputs
    base_rate: Optional[float] = None
    base_rate_reasoning: Optional[str] = None
    evidence_adjusted: Optional[float] = None
    evidence_factors: list[str] = field(default_factory=list)
    adversarial_arguments: list[str] = field(default_factory=list)
    final_adjustment: Optional[str] = None

    # Metadata
    total_time_seconds: float = 0.0
    steps_completed: int = 0

    def to_signal_output(self) -> LLMSignalOutput:
        """Convert to standard LLMSignalOutput for the signal store."""
        return LLMSignalOutput(
            probability=self.probability,
            confidence=self.confidence,
            reasoning=self.reasoning,
            key_factors=self.evidence_factors[:5],
        )


# --- Step prompts ---

STEP1_BASE_RATE = """You are a calibrated forecaster. Your job is to estimate a BASE RATE — the historical frequency of events similar to this one.

## Market Question
{question}

{description}

## Instructions
1. Identify the REFERENCE CLASS: what category of events is this? (e.g., "incumbent party winning reelection", "Fed rate cuts during expansion", "underdog NBA team winning finals")
2. Find the BASE RATE: historically, how often do events in this reference class happen? Use specific numbers if possible.
3. State your base rate estimate as a probability.

Output ONLY valid JSON:
{{"base_rate": 0.XX, "reference_class": "description of the reference class", "historical_reasoning": "why this base rate"}}"""

STEP2_EVIDENCE = """You are a calibrated forecaster doing an EVIDENCE UPDATE. You've been given a base rate. Now identify specific factors that move the probability AWAY from that base rate.

## Market Question
{question}

{description}

## Base Rate Analysis (from previous step)
Reference class: {reference_class}
Base rate: {base_rate:.0%}
Reasoning: {base_rate_reasoning}

{cross_platform_context}

## Instructions
1. List 3-5 specific factors that make this event MORE likely than the base rate
2. List 3-5 specific factors that make this event LESS likely than the base rate
3. Weigh these factors and produce an UPDATED probability
4. Each factor should shift the probability by a specific amount (e.g., "+8% because...")

Output ONLY valid JSON:
{{"updated_probability": 0.XX, "factors_for": ["factor (+X%): reasoning", ...], "factors_against": ["factor (-X%): reasoning", ...], "net_adjustment": "+/-X% from base rate"}}"""

STEP3_ADVERSARIAL = """You are a calibrated forecaster doing a FINAL ADVERSARIAL CHECK. You've estimated a probability. Now try to BREAK your own estimate.

## Market Question
{question}

## Your Current Estimate: {current_estimate:.0%}

Based on:
- Base rate: {base_rate:.0%} ({reference_class})
- Evidence update: {evidence_summary}

## Instructions
1. Argue the OPPOSITE of your estimate. If you said 70%, argue why it should be 30%.
2. Identify your biggest source of OVERCONFIDENCE. What are you most likely wrong about?
3. Consider: are you anchoring too heavily on the base rate? Or not enough?
4. Produce a FINAL probability that accounts for this adversarial check.
5. Rate your CONFIDENCE (0.0-1.0) in this final estimate. Be honest — lower confidence when evidence is thin.

Output ONLY valid JSON:
{{"probability": 0.XX, "confidence": 0.XX, "adversarial_arguments": ["argument 1", "argument 2", "argument 3"], "adjustment_from_step2": "why you moved (or didn't move) from your step 2 estimate", "reasoning": "final 1-2 sentence summary of your position", "key_factors": ["factor 1", "factor 2", "factor 3"]}}"""


class SuperforecasterProtocol:
    """3-step structured reasoning chain for probability estimation."""

    def __init__(self, runner: Optional[ClaudeRunner] = None):
        self._runner = runner or ClaudeRunner(timeout=90, max_retries=1)

    async def evaluate(
        self,
        question: str,
        description: str = "",
        cross_platform_context: str = "",
    ) -> Optional[ForecastChainResult]:
        """Run the full 3-step superforecaster chain."""
        start = time.monotonic()
        result = ForecastChainResult(
            probability=0.5,
            confidence=0.0,
            reasoning="",
        )

        # --- Step 1: Base Rate ---
        step1 = await self._run_step1(question, description)
        if step1 is None:
            logger.warning("superforecaster_step1_failed")
            return None

        result.base_rate = step1.get("base_rate", 0.5)
        result.base_rate_reasoning = step1.get("historical_reasoning", "")
        result.steps_completed = 1

        logger.info(
            "superforecaster_step1",
            base_rate=result.base_rate,
            reference_class=step1.get("reference_class", ""),
        )

        # --- Step 2: Evidence Update ---
        step2 = await self._run_step2(
            question,
            description,
            step1,
            cross_platform_context,
        )
        if step2 is None:
            # Fall back to base rate
            result.probability = result.base_rate
            result.confidence = 0.3
            result.reasoning = f"Base rate only: {result.base_rate_reasoning}"
            result.total_time_seconds = time.monotonic() - start
            return result

        result.evidence_adjusted = step2.get("updated_probability", result.base_rate)
        result.evidence_factors = (
            step2.get("factors_for", []) + step2.get("factors_against", [])
        )
        result.steps_completed = 2

        logger.info(
            "superforecaster_step2",
            base_rate=result.base_rate,
            updated=result.evidence_adjusted,
            adjustment=step2.get("net_adjustment", ""),
        )

        # --- Step 3: Adversarial Check ---
        evidence_summary = step2.get("net_adjustment", "no significant adjustment")
        step3 = await self._run_step3(
            question,
            result.evidence_adjusted,
            result.base_rate,
            step1.get("reference_class", ""),
            evidence_summary,
        )
        if step3 is None:
            # Fall back to step 2 estimate
            result.probability = result.evidence_adjusted
            result.confidence = 0.5
            result.reasoning = f"Evidence-adjusted (no adversarial check): {evidence_summary}"
            result.total_time_seconds = time.monotonic() - start
            return result

        result.probability = max(0.0, min(1.0, step3.get("probability", result.evidence_adjusted)))
        result.confidence = max(0.0, min(1.0, step3.get("confidence", 0.5)))
        result.reasoning = step3.get("reasoning", "")
        result.adversarial_arguments = step3.get("adversarial_arguments", [])
        result.final_adjustment = step3.get("adjustment_from_step2", "")
        result.steps_completed = 3

        # Merge key_factors from step 3 into evidence_factors
        step3_factors = step3.get("key_factors", [])
        if step3_factors:
            result.evidence_factors = step3_factors

        result.total_time_seconds = time.monotonic() - start

        logger.info(
            "superforecaster_complete",
            base_rate=result.base_rate,
            evidence=result.evidence_adjusted,
            final=result.probability,
            confidence=result.confidence,
            time=f"{result.total_time_seconds:.1f}s",
        )

        return result

    async def _run_step1(self, question: str, description: str) -> Optional[dict]:
        prompt = STEP1_BASE_RATE.format(
            question=question,
            description=f"**Description:** {description}" if description else "",
        )
        return await self._call_and_parse(prompt, "step1")

    async def _run_step2(
        self, question: str, description: str, step1: dict, cross_platform_context: str,
    ) -> Optional[dict]:
        ctx_block = ""
        if cross_platform_context:
            ctx_block = f"## Cross-Platform Intelligence\n{cross_platform_context}"

        prompt = STEP2_EVIDENCE.format(
            question=question,
            description=f"**Description:** {description}" if description else "",
            reference_class=step1.get("reference_class", "unknown"),
            base_rate=step1.get("base_rate", 0.5),
            base_rate_reasoning=step1.get("historical_reasoning", ""),
            cross_platform_context=ctx_block,
        )
        return await self._call_and_parse(prompt, "step2")

    async def _run_step3(
        self, question: str, current_estimate: float, base_rate: float,
        reference_class: str, evidence_summary: str,
    ) -> Optional[dict]:
        prompt = STEP3_ADVERSARIAL.format(
            question=question,
            current_estimate=current_estimate,
            base_rate=base_rate,
            reference_class=reference_class,
            evidence_summary=evidence_summary,
        )
        return await self._call_and_parse(prompt, "step3")

    async def _call_and_parse(self, prompt: str, step_name: str) -> Optional[dict]:
        """Run Claude CLI and parse raw JSON output.

        Uses _run_cli directly (not evaluate) because each step
        has a different JSON schema — we parse the dict ourselves
        rather than validating against LLMSignalOutput.
        """
        raw = await self._runner._run_cli(prompt)
        if raw is None:
            logger.warning("superforecaster_no_output", step=step_name)
            return None
        result = _extract_json(raw)
        if result is None:
            logger.warning("superforecaster_parse_failed", step=step_name, preview=raw[:200])
        return result


def _extract_json(text: str) -> Optional[dict]:
    """Extract JSON from Claude output, handling various formats."""
    import re

    if not text:
        return None

    # Try wrapper format
    try:
        wrapper = json.loads(text)
        if isinstance(wrapper, dict):
            if "result" in wrapper:
                text = wrapper["result"]
            else:
                return wrapper
    except (json.JSONDecodeError, TypeError):
        pass

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError):
        pass

    # Code block
    code_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_match:
        try:
            return json.loads(code_match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # First { ... }
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    return None
