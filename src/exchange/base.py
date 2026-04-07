"""Abstract exchange adapter interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OrderBookEntry:
    price: float
    size: float


@dataclass
class OrderBook:
    bids: list[OrderBookEntry] = field(default_factory=list)
    asks: list[OrderBookEntry] = field(default_factory=list)
    timestamp: Optional[float] = None

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def midpoint(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


@dataclass
class OrderResult:
    order_id: str
    success: bool
    message: str = ""


@dataclass
class TradeRecord:
    trade_id: str
    order_id: str
    token_id: str
    side: str  # "buy" or "sell"
    price: float
    size: float
    fee: float
    timestamp: float


class ExchangeAdapter(ABC):
    """Abstract interface for prediction market exchanges."""

    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection and authenticate."""

    @abstractmethod
    async def get_order_book(self, token_id: str) -> OrderBook:
        """Get the order book for a token."""

    @abstractmethod
    async def get_midpoint(self, token_id: str) -> float:
        """Get the midpoint price for a token."""

    @abstractmethod
    async def get_spread(self, token_id: str) -> float:
        """Get the bid-ask spread for a token."""

    @abstractmethod
    async def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size: float,
    ) -> OrderResult:
        """Place a GTC limit order. Returns order result."""

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order. Returns True if successful."""

    @abstractmethod
    async def cancel_all(self) -> int:
        """Cancel all open orders. Returns count cancelled."""

    @abstractmethod
    async def get_open_orders(self) -> list[dict]:
        """Get all open orders."""

    @abstractmethod
    async def get_trades(self) -> list[TradeRecord]:
        """Get recent trade history for reconciliation."""
