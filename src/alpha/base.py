"""Alpha source interface for prediction market signals."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from src.db.models import Market


@dataclass
class AlphaOutput:
    """Output from an alpha source.

    edge: estimated_probability - market_price (positive = underpriced YES)
    confidence: 0.0-1.0 how confident the alpha is in this signal
    notes: human-readable explanation
    """

    edge: float
    confidence: float
    notes: str = ""
    meta: dict = field(default_factory=dict)


class AlphaSource(ABC):
    """Abstract base for prediction market alpha sources."""

    name: str = "base"

    @abstractmethod
    async def compute(
        self,
        market: Market,
        context: dict,
    ) -> Optional[AlphaOutput]:
        """Compute alpha signal for a market.

        Args:
            market: The market to evaluate.
            context: Dict with order_book, signals, positions, etc.

        Returns:
            AlphaOutput if the alpha has a signal, None if no opinion.
        """
