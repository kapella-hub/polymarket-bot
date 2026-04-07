"""Strategy mode definitions."""

from dataclasses import dataclass
from src.db.models import StrategyMode


@dataclass
class TradeDecision:
    """Output of the ensemble: what to do with a market."""

    market_id: str
    strategy: StrategyMode
    side: str  # "buy" or "sell"
    token_id: str  # Which token to trade (YES or NO)
    edge: float  # Combined edge estimate
    confidence: float  # Combined confidence
    suggested_size: float  # Pre-risk-check suggested size in USDC
    notes: str = ""
