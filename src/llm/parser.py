"""Parse and validate Claude CLI output into structured signals."""

import json
import re
from typing import Optional

import structlog
from pydantic import BaseModel, Field, field_validator

logger = structlog.get_logger()


class LLMSignalOutput(BaseModel):
    """Structured output from Claude's market probability analysis."""

    probability: float = Field(ge=0.0, le=1.0, description="Estimated true probability")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the estimate")
    reasoning: str = Field(description="Brief explanation of the estimate")
    key_factors: list[str] = Field(default_factory=list, description="Factors driving the estimate")

    @field_validator("probability", "confidence")
    @classmethod
    def clamp_to_range(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


def parse_claude_output(raw_output: str) -> Optional[LLMSignalOutput]:
    """Parse Claude CLI output into a validated LLMSignalOutput.

    Handles multiple output formats:
    1. Direct JSON object
    2. JSON wrapped in ```json``` code blocks
    3. JSON embedded in prose text
    4. Claude CLI --output-format json wrapper
    """
    if not raw_output or not raw_output.strip():
        return None

    text = raw_output.strip()

    # Try 1: Parse as Claude CLI JSON wrapper {"result": "..."}
    try:
        wrapper = json.loads(text)
        if isinstance(wrapper, dict) and "result" in wrapper:
            text = wrapper["result"]
    except (json.JSONDecodeError, TypeError):
        pass

    # Try 2: Direct JSON parse
    parsed = _try_parse_json(text)
    if parsed:
        return _validate(parsed)

    # Try 3: Extract from ```json code block
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if code_block:
        parsed = _try_parse_json(code_block.group(1))
        if parsed:
            return _validate(parsed)

    # Try 4: Find first { ... } in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        parsed = _try_parse_json(brace_match.group(0))
        if parsed:
            return _validate(parsed)

    logger.warning("llm_output_unparseable", output_preview=text[:200])
    return None


def _try_parse_json(text: str) -> Optional[dict]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, TypeError):
        return None


def _validate(data: dict) -> Optional[LLMSignalOutput]:
    try:
        return LLMSignalOutput(**data)
    except Exception as e:
        logger.warning("llm_output_validation_failed", error=str(e), data=data)
        return None
