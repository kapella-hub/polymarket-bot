#!/usr/bin/env python3
"""
Backtest framework for Polymarket trading strategies.

Replays signals from the DB, simulates trades under different strategy
configurations, and computes P&L metrics for comparison.

Usage:
    python scripts/backtest.py
"""

import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    signal_id: int
    market_id: str
    question: str
    category: Optional[str]
    model_prob: float
    confidence: float
    recorded_mkt_price: float
    best_bid: Optional[float]
    best_ask: Optional[float]
    last_price: Optional[float]
    volume: float
    liquidity: float
    end_date: Optional[datetime]
    evaluated_at: datetime
    key_factors: Optional[str]


@dataclass
class MarketState:
    market_id: str
    question: str
    category: Optional[str]
    best_bid: Optional[float]
    best_ask: Optional[float]
    last_price: Optional[float]
    volume: float
    liquidity: float
    end_date: Optional[datetime]


@dataclass
class Position:
    market_id: str
    side: str  # "buy_yes" or "buy_no"
    entry_price: float
    size_usd: float
    token_qty: float
    entry_time: datetime
    signal: Signal


@dataclass
class Trade:
    market_id: str
    side: str
    entry_price: float
    exit_price: float
    size_usd: float
    pnl: float
    hold_time_hours: float
    strategy: str


@dataclass
class StrategyResult:
    name: str
    trades: list[Trade] = field(default_factory=list)
    total_pnl: float = 0.0
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    max_drawdown: float = 0.0
    total_exposure: float = 0.0
    signals_evaluated: int = 0
    signals_traded: int = 0

    @property
    def win_rate(self) -> float:
        return self.winners / self.total_trades if self.total_trades > 0 else 0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.total_trades if self.total_trades > 0 else 0

    @property
    def roi_pct(self) -> float:
        return (self.total_pnl / self.total_exposure * 100) if self.total_exposure > 0 else 0

    @property
    def selectivity(self) -> float:
        return self.signals_traded / self.signals_evaluated if self.signals_evaluated > 0 else 0


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _parse_float(val: str) -> Optional[float]:
    if not val or val == "":
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _parse_dt(val: str) -> Optional[datetime]:
    if not val or val == "":
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def load_signals(path: str = "backtest_data.csv") -> list[Signal]:
    signals = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            signals.append(Signal(
                signal_id=int(row["signal_id"]),
                market_id=row["market_id"],
                question=row["question"],
                category=row["category"] or None,
                model_prob=float(row["model_prob"]),
                confidence=float(row["confidence"]),
                recorded_mkt_price=float(row["recorded_mkt_price"]),
                best_bid=_parse_float(row["best_bid"]),
                best_ask=_parse_float(row["best_ask"]),
                last_price=_parse_float(row["last_price"]),
                volume=float(row["volume"]),
                liquidity=float(row["liquidity"]),
                end_date=_parse_dt(row.get("end_date", "")),
                evaluated_at=_parse_dt(row["evaluated_at"]),
                key_factors=row.get("key_factors"),
            ))
    return signals


def load_markets(path: str = "backtest_markets.csv") -> dict[str, MarketState]:
    markets = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row["market_id"]
            markets[mid] = MarketState(
                market_id=mid,
                question=row["question"],
                category=row["category"] or None,
                best_bid=_parse_float(row["best_bid"]),
                best_ask=_parse_float(row["best_ask"]),
                last_price=_parse_float(row["last_price"]),
                volume=float(row["volume"]),
                liquidity=float(row["liquidity"]),
                end_date=_parse_dt(row.get("end_date", "")),
            )
    return markets


# ---------------------------------------------------------------------------
# Price Resolution (matches production logic)
# ---------------------------------------------------------------------------

def resolve_price(sig: Signal) -> Optional[float]:
    """Best available price, matching production resolve_price()."""
    if sig.best_bid is not None and sig.best_bid > 0:
        return sig.best_bid
    if sig.best_ask is not None and sig.best_ask > 0:
        return sig.best_ask
    if sig.last_price is not None and sig.last_price > 0:
        return sig.last_price
    return None


def current_market_price(market: MarketState) -> Optional[float]:
    """Current best price for mark-to-market."""
    if market.best_bid is not None and market.best_bid > 0:
        return market.best_bid
    if market.best_ask is not None and market.best_ask > 0:
        return market.best_ask
    if market.last_price is not None and market.last_price > 0:
        return market.last_price
    return None


# ---------------------------------------------------------------------------
# Strategy Definitions
# ---------------------------------------------------------------------------

# Each strategy is a function: (signal, market_state) -> Optional[trade_params]
# trade_params = (side, size_usd, entry_price)

FEES_PCT = 0.02  # 2% round-trip (Polymarket ~1% each side for taker)
SLIPPAGE_PCT = 0.01  # 1% slippage estimate


def kelly_size(edge: float, price: float, bankroll: float, fraction: float = 0.25, cap: float = 500.0) -> float:
    """Quarter-Kelly position sizing."""
    if price <= 0 or price >= 1:
        return 0
    b = (1.0 / price) - 1.0  # decimal odds
    prob = min(0.99, max(0.01, price + edge))
    q = 1.0 - prob
    kelly = (prob * b - q) / b if b > 0 else 0
    kelly = max(kelly, 0)
    size = kelly * fraction * bankroll
    return min(size, cap)


# ---- Strategy 0: Baseline (current bot logic, all markets) ----

def strategy_baseline(sig: Signal, mkt: Optional[MarketState], bankroll: float) -> Optional[tuple]:
    """Current bot: trade any signal with edge > 5%, no liquidity filter."""
    price = resolve_price(sig)
    if price is None:
        price = 0.5  # OLD behavior
    edge = sig.model_prob - price
    if abs(edge) < 0.05:
        return None
    if edge > 0:
        side = "buy_yes"
        entry = price
    else:
        side = "buy_no"
        entry = 1.0 - price
        edge = abs(edge)
    size = kelly_size(edge, entry, bankroll)
    if size < 1.0:
        return None
    return (side, size, entry)


# ---- Strategy 1: Liquidity-only (require real order book) ----

def strategy_liquidity_only(sig: Signal, mkt: Optional[MarketState], bankroll: float) -> Optional[tuple]:
    """Only trade markets with real bids (tradeable)."""
    if sig.best_bid is None or sig.best_bid <= 0:
        return None
    price = sig.best_bid
    edge = sig.model_prob - price
    if abs(edge) < 0.05:
        return None
    if edge > 0:
        side = "buy_yes"
        entry = sig.best_ask if sig.best_ask else price * 1.02  # Buy at ask
    else:
        side = "buy_no"
        entry = 1.0 - price
        edge = abs(edge)
    size = kelly_size(edge, entry, bankroll)
    if size < 1.0:
        return None
    return (side, size, entry)


# ---- Strategy 2: High confidence + liquidity ----

def strategy_high_confidence(sig: Signal, mkt: Optional[MarketState], bankroll: float) -> Optional[tuple]:
    """Only trade when model is confident AND market is liquid."""
    if sig.best_bid is None or sig.best_bid <= 0:
        return None
    if sig.confidence < 0.6:
        return None
    price = sig.best_bid
    edge = sig.model_prob - price
    if abs(edge) < 0.03:  # Tighter threshold since we trust the signal more
        return None
    # Require edge * confidence product
    if abs(edge) * sig.confidence < 0.04:
        return None
    if edge > 0:
        side = "buy_yes"
        entry = sig.best_ask if sig.best_ask else price * 1.02
    else:
        side = "buy_no"
        entry = 1.0 - price
        edge = abs(edge)
    size = kelly_size(edge, entry, bankroll, fraction=0.35)  # More aggressive sizing
    if size < 1.0:
        return None
    return (side, size, entry)


# ---- Strategy 3: Contrarian (fade large model-market disagreements) ----

def strategy_contrarian(sig: Signal, mkt: Optional[MarketState], bankroll: float) -> Optional[tuple]:
    """When model strongly disagrees with market, trust the MARKET and fade the model.
    Rationale: on liquid markets, the market is usually right. The model's disagreement
    often corrects toward the market, not away from it."""
    if sig.best_bid is None or sig.best_bid <= 0:
        return None
    price = sig.best_bid
    edge = sig.model_prob - price
    # Only act on large disagreements (>8%)
    if abs(edge) < 0.08:
        return None
    # FADE the model: if model says higher, sell (buy NO); if model says lower, buy YES
    # This is contrarian — betting the market is right and model will correct
    if edge > 0:
        # Model says YES is underpriced — contrarian says market is right, sell YES (buy NO)
        side = "buy_no"
        entry = 1.0 - price
        contrarian_edge = abs(edge) * 0.5  # Assume partial correction
    else:
        # Model says YES is overpriced — contrarian says market is right, buy YES
        side = "buy_yes"
        entry = sig.best_ask if sig.best_ask else price * 1.02
        contrarian_edge = abs(edge) * 0.5
    size = kelly_size(contrarian_edge, entry, bankroll, fraction=0.15)  # Conservative
    if size < 1.0:
        return None
    return (side, size, entry)


# ---- Strategy 4: Spread capture (pseudo market-making) ----

def strategy_spread_capture(sig: Signal, mkt: Optional[MarketState], bankroll: float) -> Optional[tuple]:
    """Capture bid-ask spread on liquid markets.
    Buy at bid, expect to sell at ask. Profit = spread - fees.
    Only works on markets with meaningful spread and depth."""
    if sig.best_bid is None or sig.best_ask is None:
        return None
    if sig.best_bid <= 0 or sig.best_ask <= 0:
        return None
    spread = sig.best_ask - sig.best_bid
    if spread < 0.03:  # Need at least 3 cent spread to cover fees
        return None
    if sig.liquidity < 50000:  # Need meaningful depth
        return None
    # Entry at bid + small improvement
    entry = sig.best_bid + 0.005
    # Expected exit at ask - small improvement
    expected_exit = sig.best_ask - 0.005
    expected_profit_pct = (expected_exit - entry) / entry
    if expected_profit_pct < FEES_PCT + SLIPPAGE_PCT:
        return None
    # Fixed size for spread capture, not Kelly (it's not directional)
    size = min(200.0, bankroll * 0.05)
    return ("buy_yes", size, entry)


# ---- Strategy 5: Near-resolution focus ----

def strategy_near_resolution(sig: Signal, mkt: Optional[MarketState], bankroll: float) -> Optional[tuple]:
    """Focus on markets 1-14 days from resolution where model has edge.
    Near resolution = more information, more accurate model, faster payoff."""
    if sig.best_bid is None or sig.best_bid <= 0:
        return None
    if sig.end_date is None:
        return None
    now = sig.evaluated_at or datetime.now(timezone.utc)
    days_to_end = (sig.end_date - now).total_seconds() / 86400
    if days_to_end < 1 or days_to_end > 14:
        return None
    price = sig.best_bid
    edge = sig.model_prob - price
    if abs(edge) < 0.04:  # Lower threshold near resolution
        return None
    # Scale confidence by proximity to resolution
    time_boost = max(1.0, 2.0 - days_to_end / 7)  # Up to 2x near resolution
    if abs(edge) * sig.confidence * time_boost < 0.03:
        return None
    if edge > 0:
        side = "buy_yes"
        entry = sig.best_ask if sig.best_ask else price * 1.02
    else:
        side = "buy_no"
        entry = 1.0 - price
        edge = abs(edge)
    size = kelly_size(edge, entry, bankroll, fraction=0.30)
    if size < 1.0:
        return None
    return (side, size, entry)


# ---- Strategy 6: Combined best-of ----

def strategy_combined(sig: Signal, mkt: Optional[MarketState], bankroll: float) -> Optional[tuple]:
    """Combined strategy:
    1. Require liquidity (real bid)
    2. If near resolution + confident: aggressive sizing
    3. If spread is wide enough: spread capture overlay
    4. Otherwise: require high edge*confidence product
    """
    if sig.best_bid is None or sig.best_bid <= 0:
        return None

    price = sig.best_bid
    edge = sig.model_prob - price

    # Check for spread capture opportunity first
    if sig.best_ask is not None and sig.best_ask > 0:
        spread = sig.best_ask - sig.best_bid
        if spread >= 0.03 and sig.liquidity >= 50000:
            entry = sig.best_bid + 0.005
            expected_exit = sig.best_ask - 0.005
            profit_pct = (expected_exit - entry) / entry
            if profit_pct > FEES_PCT + SLIPPAGE_PCT:
                size = min(150.0, bankroll * 0.04)
                return ("buy_yes", size, entry)

    # Near-resolution boost
    time_boost = 1.0
    if sig.end_date is not None:
        now = sig.evaluated_at or datetime.now(timezone.utc)
        days_to_end = (sig.end_date - now).total_seconds() / 86400
        if 1 <= days_to_end <= 14:
            time_boost = max(1.0, 2.0 - days_to_end / 7)

    # Edge threshold: base 5%, reduced by time_boost and confidence
    effective_threshold = 0.05 / (time_boost * max(sig.confidence, 0.3))
    effective_threshold = max(effective_threshold, 0.02)  # Floor

    if abs(edge) < effective_threshold:
        return None

    # Edge * confidence gate
    if abs(edge) * sig.confidence < 0.03:
        return None

    if edge > 0:
        side = "buy_yes"
        entry = sig.best_ask if sig.best_ask else price * 1.02
    else:
        side = "buy_no"
        entry = 1.0 - price
        edge = abs(edge)

    fraction = 0.25 * time_boost * min(sig.confidence, 1.0)
    fraction = min(fraction, 0.50)  # Cap
    size = kelly_size(edge, entry, bankroll, fraction=fraction)
    if size < 1.0:
        return None
    return (side, size, entry)


# ---------------------------------------------------------------------------
# Backtest Engine
# ---------------------------------------------------------------------------

def run_backtest(
    signals: list[Signal],
    markets: dict[str, MarketState],
    strategy_fn,
    strategy_name: str,
    initial_bankroll: float = 10000.0,
    max_position_per_market: float = 500.0,
    dedup_interval_hours: float = 2.0,
) -> StrategyResult:
    """
    Replay signals through a strategy function and compute P&L.

    For each signal:
    1. Strategy decides whether to trade and at what size
    2. We simulate entry at the ask (for buys)
    3. Mark-to-market against current prices

    Dedup: only one trade per market per dedup_interval_hours to avoid
    the same signal generating hundreds of trades.
    """
    result = StrategyResult(name=strategy_name)
    positions: dict[str, Position] = {}
    bankroll = initial_bankroll
    peak_value = initial_bankroll
    last_trade_time: dict[str, datetime] = {}

    for sig in signals:
        result.signals_evaluated += 1
        mkt = markets.get(sig.market_id)

        # Dedup: skip if we traded this market recently
        last = last_trade_time.get(sig.market_id)
        if last and sig.evaluated_at:
            hours_since = (sig.evaluated_at - last).total_seconds() / 3600
            if hours_since < dedup_interval_hours:
                continue

        # Skip if already holding a position in this market
        if sig.market_id in positions:
            continue

        # Ask strategy
        params = strategy_fn(sig, mkt, bankroll)
        if params is None:
            continue

        side, size_usd, entry_price = params

        # Cap position size
        size_usd = min(size_usd, max_position_per_market, bankroll * 0.2)
        if size_usd < 1.0 or entry_price <= 0:
            continue

        # Apply slippage to entry
        if side == "buy_yes":
            entry_with_slippage = entry_price * (1 + SLIPPAGE_PCT)
        else:
            entry_with_slippage = entry_price * (1 + SLIPPAGE_PCT)

        token_qty = size_usd / entry_with_slippage
        fee = size_usd * FEES_PCT

        # Deduct cost from bankroll
        bankroll -= (size_usd + fee)
        if bankroll < 0:
            bankroll += (size_usd + fee)  # Can't afford it
            continue

        positions[sig.market_id] = Position(
            market_id=sig.market_id,
            side=side,
            entry_price=entry_with_slippage,
            size_usd=size_usd,
            token_qty=token_qty,
            entry_time=sig.evaluated_at,
            signal=sig,
        )
        last_trade_time[sig.market_id] = sig.evaluated_at
        result.signals_traded += 1
        result.total_exposure += size_usd

    # Mark-to-market all positions against current prices
    for mid, pos in positions.items():
        mkt = markets.get(mid)
        if mkt is None:
            # Market not found, assume flat
            exit_price = pos.entry_price
        else:
            current = current_market_price(mkt)
            if current is None:
                exit_price = pos.entry_price  # Can't price, assume flat
            elif pos.side == "buy_yes":
                exit_price = current
            else:
                exit_price = 1.0 - current

        # P&L = (exit - entry) * token_qty - fees
        gross_pnl = (exit_price - pos.entry_price) * pos.token_qty
        exit_fee = abs(exit_price * pos.token_qty) * (FEES_PCT / 2)  # Half the round-trip
        net_pnl = gross_pnl - exit_fee

        hold_hours = 0
        if pos.entry_time:
            now = datetime.now(timezone.utc)
            hold_hours = (now - pos.entry_time).total_seconds() / 3600

        trade = Trade(
            market_id=mid,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_usd=pos.size_usd,
            pnl=net_pnl,
            hold_time_hours=hold_hours,
            strategy=strategy_name,
        )
        result.trades.append(trade)
        result.total_pnl += net_pnl
        bankroll += pos.size_usd + net_pnl  # Return capital + P&L

        if net_pnl > 0:
            result.winners += 1
        elif net_pnl < 0:
            result.losers += 1

        # Track drawdown
        portfolio_value = bankroll + sum(
            p.size_usd for p in positions.values() if p.market_id != mid
        )
        peak_value = max(peak_value, portfolio_value)
        dd = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0
        result.max_drawdown = max(result.max_drawdown, dd)

    result.total_trades = len(result.trades)
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_results(results: list[StrategyResult]):
    """Print comparison table."""
    print("\n" + "=" * 120)
    print(f"{'STRATEGY BACKTEST COMPARISON':^120}")
    print(f"{'(3 days of data, mark-to-market against current prices)':^120}")
    print("=" * 120)

    header = f"{'Strategy':<28} {'Trades':>6} {'Win%':>6} {'P&L ($)':>10} {'ROI%':>8} {'AvgP&L':>8} {'MaxDD%':>7} {'Exposure':>10} {'Select%':>8}"
    print(header)
    print("-" * 120)

    for r in sorted(results, key=lambda x: x.total_pnl, reverse=True):
        print(
            f"{r.name:<28} "
            f"{r.total_trades:>6} "
            f"{r.win_rate * 100:>5.1f}% "
            f"{r.total_pnl:>+10.2f} "
            f"{r.roi_pct:>+7.2f}% "
            f"{r.avg_pnl:>+8.2f} "
            f"{r.max_drawdown * 100:>6.1f}% "
            f"{r.total_exposure:>10.0f} "
            f"{r.selectivity * 100:>7.1f}%"
        )

    print("-" * 120)

    # Per-trade detail for top strategy
    best = max(results, key=lambda x: x.total_pnl)
    print(f"\n--- Best strategy: {best.name} ---")
    print(f"{'Market':<65} {'Side':<8} {'Entry':>7} {'Exit':>7} {'Size':>8} {'P&L':>9}")
    print("-" * 110)
    for t in sorted(best.trades, key=lambda x: x.pnl, reverse=True):
        # Get question from signal
        label = t.market_id[:60]
        print(
            f"{label:<65} "
            f"{t.side:<8} "
            f"{t.entry_price:>7.4f} "
            f"{t.exit_price:>7.4f} "
            f"${t.size_usd:>7.0f} "
            f"{t.pnl:>+9.2f}"
        )

    # Also print worst strategy details
    worst = min(results, key=lambda x: x.total_pnl)
    if worst.name != best.name:
        print(f"\n--- Worst strategy: {worst.name} ---")
        print(f"{'Market':<65} {'Side':<8} {'Entry':>7} {'Exit':>7} {'Size':>8} {'P&L':>9}")
        print("-" * 110)
        for t in sorted(worst.trades, key=lambda x: x.pnl)[:10]:
            label = t.market_id[:60]
            print(
                f"{label:<65} "
                f"{t.side:<8} "
                f"{t.entry_price:>7.4f} "
                f"{t.exit_price:>7.4f} "
                f"${t.size_usd:>7.0f} "
                f"{t.pnl:>+9.2f}"
            )


# ---------------------------------------------------------------------------
# Cross-market analysis (not a trading strategy per se, but informative)
# ---------------------------------------------------------------------------

def analyze_cross_market(signals: list[Signal], markets: dict[str, MarketState]):
    """Find groups of related markets and check probability consistency."""
    print("\n" + "=" * 120)
    print(f"{'CROSS-MARKET ANALYSIS':^120}")
    print("=" * 120)

    # Group by likely event (shared keywords in question)
    from collections import defaultdict
    groups = defaultdict(list)
    for mid, mkt in markets.items():
        # Extract event identifier from question
        q = mkt.question.lower()
        if "nba mvp" in q:
            groups["NBA MVP 2025-26"].append(mkt)
        elif "premier league" in q:
            groups["EPL 2025-26"].append(mkt)
        elif "champions league" in q:
            groups["Champions League 2025-26"].append(mkt)
        elif "hungary" in q and "prime minister" in q:
            groups["Hungary PM"].append(mkt)
        elif "texas republican" in q:
            groups["Texas GOP Primary 2026"].append(mkt)
        elif "colombian presidential" in q:
            groups["Colombia 2026"].append(mkt)
        elif "masters" in q:
            groups["2026 Masters"].append(mkt)

    for group_name, group_markets in sorted(groups.items()):
        if len(group_markets) < 2:
            continue
        print(f"\n{group_name} ({len(group_markets)} markets):")
        total_prob = 0
        for mkt in sorted(group_markets, key=lambda m: -(m.best_bid or m.last_price or 0)):
            price = mkt.best_bid or mkt.last_price or mkt.best_ask or 0
            total_prob += price
            print(f"  {mkt.question[:65]:<68} price={price:.3f}")
        print(f"  {'TOTAL PROBABILITY':<68} = {total_prob:.3f}")
        if total_prob > 1.05:
            print(f"  ** OVERPRICED by {(total_prob - 1) * 100:.1f}% — arbitrage: sell all **")
        elif total_prob < 0.90:
            print(f"  ** UNDERPRICED by {(1 - total_prob) * 100:.1f}% — arbitrage: buy all **")
        else:
            print(f"  (within normal range)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    base = Path(__file__).parent.parent
    print("Loading data...")
    signals = load_signals(str(base / "backtest_data.csv"))
    markets = load_markets(str(base / "backtest_markets.csv"))
    print(f"  {len(signals)} signals, {len(markets)} markets")

    strategies = [
        (strategy_baseline, "0-Baseline (old bot)"),
        (strategy_liquidity_only, "1-Liquidity Only"),
        (strategy_high_confidence, "2-High Confidence+Liquid"),
        (strategy_contrarian, "3-Contrarian (fade model)"),
        (strategy_spread_capture, "4-Spread Capture"),
        (strategy_near_resolution, "5-Near Resolution"),
        (strategy_combined, "6-Combined Best-Of"),
    ]

    results = []
    for fn, name in strategies:
        print(f"Running: {name}...")
        r = run_backtest(signals, markets, fn, name)
        results.append(r)

    print_results(results)
    analyze_cross_market(signals, markets)

    print("\n" + "=" * 120)
    print("NOTES:")
    print("- P&L is mark-to-market: entry at signal time, exit at current market price")
    print("- Includes 2% round-trip fees + 1% slippage estimate")
    print("- 3 days of data — results are directional, not statistically significant")
    print("- Dedup: max 1 trade per market per 2 hours")
    print("- Position cap: $500/market, 20% bankroll per trade")
    print("=" * 120)


if __name__ == "__main__":
    main()
