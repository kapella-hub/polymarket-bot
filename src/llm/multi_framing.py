"""Multi-framing disagreement analysis.

Runs 3 independent Claude evaluations with different cognitive framings:
- Analyst: data-driven, base rates, statistics
- Expert: domain knowledge, insider perspective
- Contrarian: what is the consensus missing?

The DISAGREEMENT between framings is the signal:
- All agree → market likely agrees too (small edge)
- Strong disagreement → genuine uncertainty (potential edge)
- One outlier → that framing may have found something others missed
"""

import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Optional

import structlog

from src.llm.claude_runner import ClaudeRunner
from src.llm.parser import LLMSignalOutput, parse_claude_output

logger = structlog.get_logger()


FRAMING_ANALYST = """You are a DATA ANALYST evaluating a prediction market. Focus ONLY on:
- Historical base rates and statistical frequencies
- Quantitative data, polls, and measurable indicators
- Reference classes and sample sizes
- Bayesian reasoning with explicit priors

Do NOT use gut feelings, narratives, or qualitative reasoning. Numbers only.

## Market Question
{question}

{context}

Output ONLY valid JSON:
{{"probability": 0.XX, "confidence": 0.XX, "reasoning": "data-driven explanation", "key_factors": ["stat 1", "stat 2", "stat 3"]}}"""


FRAMING_EXPERT = """You are a DOMAIN EXPERT evaluating a prediction market. Focus ONLY on:
- Deep domain knowledge and insider understanding
- Qualitative factors that statistics might miss
- Political dynamics, personal motivations, institutional behavior
- What people "in the know" would understand

Do NOT rely on base rates alone. Use your expertise to see beyond the numbers.

## Market Question
{question}

{context}

Output ONLY valid JSON:
{{"probability": 0.XX, "confidence": 0.XX, "reasoning": "expert analysis", "key_factors": ["insight 1", "insight 2", "insight 3"]}}"""


FRAMING_CONTRARIAN = """You are a CONTRARIAN ANALYST evaluating a prediction market. Focus ONLY on:
- What is the CONSENSUS wrong about?
- What scenario is everyone ignoring or underweighting?
- What would be surprising but is actually plausible?
- Historical examples of consensus being spectacularly wrong

Deliberately push AGAINST the obvious answer. Find the non-obvious angle.

## Market Question
{question}

{context}

Output ONLY valid JSON:
{{"probability": 0.XX, "confidence": 0.XX, "reasoning": "contrarian thesis", "key_factors": ["overlooked factor 1", "overlooked factor 2", "overlooked factor 3"]}}"""


@dataclass
class FramingResult:
    """Result from one cognitive framing."""

    framing: str  # "analyst", "expert", "contrarian"
    probability: Optional[float] = None
    confidence: Optional[float] = None
    reasoning: str = ""
    key_factors: list[str] = field(default_factory=list)


@dataclass
class DisagreementAnalysis:
    """Analysis of disagreement across framings."""

    framings: list[FramingResult] = field(default_factory=list)

    # Synthesized output
    consensus_probability: Optional[float] = None  # Weighted average
    disagreement_score: float = 0.0  # 0 = perfect agreement, 1 = max disagreement
    edge_signal: str = "none"  # "none", "weak", "moderate", "strong"
    outlier_framing: Optional[str] = None  # Which framing disagrees most
    synthesis_notes: str = ""

    @property
    def has_results(self) -> bool:
        return any(f.probability is not None for f in self.framings)

    def to_signal_output(self) -> Optional[LLMSignalOutput]:
        if self.consensus_probability is None:
            return None
        avg_conf = statistics.mean(
            [f.confidence for f in self.framings if f.confidence is not None]
        ) if any(f.confidence is not None for f in self.framings) else 0.5

        # Reduce confidence when framings disagree strongly
        adjusted_confidence = avg_conf * (1.0 - self.disagreement_score * 0.5)

        return LLMSignalOutput(
            probability=self.consensus_probability,
            confidence=max(0.05, adjusted_confidence),
            reasoning=self.synthesis_notes,
            key_factors=[f"{f.framing}: {f.reasoning[:60]}" for f in self.framings if f.probability is not None],
        )


class MultiFramingAnalyzer:
    """Runs 3 cognitive framings and analyzes their disagreement."""

    def __init__(self, runner: Optional[ClaudeRunner] = None):
        self._runner = runner or ClaudeRunner(timeout=90, max_retries=1)

    async def evaluate(
        self,
        question: str,
        context: str = "",
    ) -> DisagreementAnalysis:
        """Run all 3 framings and analyze disagreement."""
        analysis = DisagreementAnalysis()

        # Run all 3 framings concurrently (they're independent)
        tasks = [
            self._run_framing("analyst", FRAMING_ANALYST, question, context),
            self._run_framing("expert", FRAMING_EXPERT, question, context),
            self._run_framing("contrarian", FRAMING_CONTRARIAN, question, context),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, FramingResult):
                analysis.framings.append(result)

        if not analysis.has_results:
            return analysis

        # Compute consensus and disagreement
        valid = [f for f in analysis.framings if f.probability is not None]
        if not valid:
            return analysis

        probs = [f.probability for f in valid]
        confs = [f.confidence for f in valid if f.confidence is not None]

        # Confidence-weighted average
        if confs and len(confs) == len(probs):
            total_weight = sum(confs)
            if total_weight > 0:
                analysis.consensus_probability = sum(
                    p * c for p, c in zip(probs, confs)
                ) / total_weight
            else:
                analysis.consensus_probability = statistics.mean(probs)
        else:
            analysis.consensus_probability = statistics.mean(probs)

        # Disagreement score (standard deviation normalized to 0-1)
        if len(probs) >= 2:
            stdev = statistics.stdev(probs)
            analysis.disagreement_score = min(1.0, stdev * 4)  # Scale: 0.25 stdev = max

        # Identify outlier
        if len(probs) >= 3:
            mean = statistics.mean(probs)
            deviations = [(f, abs(f.probability - mean)) for f in valid]
            outlier = max(deviations, key=lambda x: x[1])
            if outlier[1] > 0.1:  # At least 10% away from mean
                analysis.outlier_framing = outlier[0].framing

        # Classify edge signal strength
        if analysis.disagreement_score < 0.1:
            analysis.edge_signal = "none"  # All agree, market probably does too
        elif analysis.disagreement_score < 0.25:
            analysis.edge_signal = "weak"
        elif analysis.disagreement_score < 0.5:
            analysis.edge_signal = "moderate"
        else:
            analysis.edge_signal = "strong"  # Major disagreement = potential edge

        # Build synthesis notes
        parts = []
        for f in valid:
            parts.append(f"{f.framing}: {f.probability:.0%}")
        parts.append(f"disagreement: {analysis.edge_signal}")
        if analysis.outlier_framing:
            parts.append(f"outlier: {analysis.outlier_framing}")
        analysis.synthesis_notes = " | ".join(parts)

        logger.info(
            "multi_framing_complete",
            consensus=f"{analysis.consensus_probability:.0%}",
            disagreement=f"{analysis.disagreement_score:.2f}",
            edge_signal=analysis.edge_signal,
            outlier=analysis.outlier_framing,
            framings={f.framing: f"{f.probability:.0%}" for f in valid},
        )

        return analysis

    async def _run_framing(
        self, name: str, template: str, question: str, context: str,
    ) -> FramingResult:
        """Run one cognitive framing."""
        prompt = template.format(
            question=question,
            context=f"## Additional Context\n{context}" if context else "",
        )

        output = await self._runner.evaluate(prompt)
        if output is None:
            return FramingResult(framing=name)

        return FramingResult(
            framing=name,
            probability=output.probability,
            confidence=output.confidence,
            reasoning=output.reasoning,
            key_factors=output.key_factors,
        )
