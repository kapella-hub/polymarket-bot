"""Market domain models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Outcome:
    name: str
    clob_token_id: str
    price: Optional[float] = None


@dataclass
class MarketInfo:
    """Parsed market data from Gamma API."""

    id: str  # condition_id
    question: str
    category: Optional[str] = None
    end_date: Optional[datetime] = None
    volume: float = 0.0
    liquidity: float = 0.0
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None
    outcomes: list[Outcome] = field(default_factory=list)
    description: Optional[str] = None
    resolution_source: Optional[str] = None
    tags: Optional[dict] = None
    active: bool = True

    @property
    def outcome_yes(self) -> Optional[Outcome]:
        for o in self.outcomes:
            if o.name.lower() in ("yes", "true", "1"):
                return o
        return self.outcomes[0] if self.outcomes else None

    @property
    def outcome_no(self) -> Optional[Outcome]:
        for o in self.outcomes:
            if o.name.lower() in ("no", "false", "0"):
                return o
        return self.outcomes[1] if len(self.outcomes) > 1 else None

    @property
    def yes_price(self) -> Optional[float]:
        o = self.outcome_yes
        return o.price if o else None

    @property
    def no_price(self) -> Optional[float]:
        o = self.outcome_no
        return o.price if o else None
