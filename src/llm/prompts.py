"""Prompt template loading and rendering."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

from src.markets.models import MarketInfo

# Template directory relative to project root
_PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"

_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)

# Load system prompt once
_system_prompt = (_PROMPTS_DIR / "system.txt").read_text(encoding="utf-8")


def render_evaluation_prompt(
    market: MarketInfo,
    enrichment_context: Optional[str] = None,
) -> str:
    """Render a full evaluation prompt for a market."""
    template = _env.get_template("generic.j2")

    # Compute time remaining
    time_remaining = ""
    if market.end_date:
        now = datetime.now(timezone.utc)
        delta = market.end_date - now
        days = delta.days
        hours = delta.seconds // 3600
        if days > 0:
            time_remaining = f"{days} days, {hours} hours"
        else:
            time_remaining = f"{hours} hours"

    return template.render(
        system_prompt=_system_prompt,
        question=market.question,
        yes_price=market.yes_price or 0.5,
        no_price=market.no_price,
        category=market.category,
        end_date=market.end_date.strftime("%Y-%m-%d %H:%M UTC") if market.end_date else None,
        time_remaining=time_remaining,
        description=market.description,
        resolution_source=market.resolution_source,
        volume=market.volume,
        liquidity=market.liquidity,
        enrichment_context=enrichment_context,
    )
