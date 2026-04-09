#!/usr/bin/env python3
"""
Structural strategy backtest — edges that DON'T require predicting outcomes.

These strategies profit from market STRUCTURE, not from being smarter about
the underlying question:

1. Market Making — earn the bid-ask spread by providing liquidity on both sides
2. Cross-Outcome Arbitrage — buy all outcomes when sum < $1.00 for guaranteed profit
3. LLM Bias Calibration — the model is consistently bearish; use deviations from
   that bias as the signal (not the absolute prediction)
4. Inventory-Neutral Spread — place matching YES+NO limit orders to earn spread
   with zero directional exposure

Key insight: Polymarket has 0% MAKER fees. This makes market making and
spread strategies viable even on tight 2-cent spreads.
"""

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data Models (reused from backtest.py)
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


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def _pf(val):
    if not val or val == "": return None
    try: return float(val)
    except: return None

def _pdt(val):
    if not val or val == "": return None
    try: return datetime.fromisoformat(val.replace("Z", "+00:00"))
    except: return None

def load_signals(path):
    signals = []
    with open(path) as f:
        for row in csv.DictReader(f):
            signals.append(Signal(
                signal_id=int(row["signal_id"]), market_id=row["market_id"],
                question=row["question"], category=row["category"] or None,
                model_prob=float(row["model_prob"]), confidence=float(row["confidence"]),
                recorded_mkt_price=float(row["recorded_mkt_price"]),
                best_bid=_pf(row["best_bid"]), best_ask=_pf(row["best_ask"]),
                last_price=_pf(row["last_price"]), volume=float(row["volume"]),
                liquidity=float(row["liquidity"]), end_date=_pdt(row.get("end_date","")),
                evaluated_at=_pdt(row["evaluated_at"]), key_factors=row.get("key_factors"),
            ))
    return signals

def load_markets(path):
    markets = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            mid = row["market_id"]
            markets[mid] = MarketState(
                market_id=mid, question=row["question"],
                category=row["category"] or None,
                best_bid=_pf(row["best_bid"]), best_ask=_pf(row["best_ask"]),
                last_price=_pf(row["last_price"]), volume=float(row["volume"]),
                liquidity=float(row["liquidity"]),
                end_date=_pdt(row.get("end_date","")),
            )
    return markets


# ===================================================================
# STRATEGY 1: MARKET MAKING SIMULATION
# ===================================================================

@dataclass
class MMResult:
    market_id: str
    question: str
    num_quotes: int = 0
    spreads_earned: float = 0.0
    inventory_pnl: float = 0.0
    total_pnl: float = 0.0
    max_inventory: float = 0.0
    avg_spread: float = 0.0


def simulate_market_making(signals: list[Signal], markets: dict) -> list[MMResult]:
    """Simulate market making on each liquid market.

    Strategy: At each observation, we quote:
      - Bid at best_bid + 0.005 (improve the bid by half a cent)
      - Ask at best_ask - 0.005 (improve the ask by half a cent)

    Fill model (conservative):
      - Our bid fills if the next observation's price <= our bid
      - Our ask fills if the next observation's price >= our ask
      - In reality, fills would be faster, but this is conservative

    Maker fee: 0% (Polymarket)
    """
    # Group signals by market, ordered by time
    by_market: dict[str, list[Signal]] = defaultdict(list)
    for sig in signals:
        if sig.best_bid and sig.best_bid > 0 and sig.best_ask and sig.best_ask > 0:
            by_market[sig.market_id].append(sig)

    results = []
    for mid, sigs in by_market.items():
        if len(sigs) < 5:
            continue
        sigs.sort(key=lambda s: s.evaluated_at)

        res = MMResult(market_id=mid, question=sigs[0].question[:60])
        inventory = 0.0  # Positive = long YES, negative = short YES (long NO)
        inventory_cost = 0.0
        total_spread_earned = 0.0
        max_inv = 0.0
        spread_sum = 0.0
        quote_size_usd = 100.0  # Quote $100 on each side
        max_inventory_usd = 300.0  # Max $300 directional exposure

        for i in range(len(sigs) - 1):
            s = sigs[i]
            s_next = sigs[i + 1]

            spread = s.best_ask - s.best_bid
            if spread < 0.015:  # Need at least 1.5 cent spread
                continue

            our_bid = s.best_bid + 0.005
            our_ask = s.best_ask - 0.005
            our_spread = our_ask - our_bid
            if our_spread <= 0:
                continue

            spread_sum += spread
            res.num_quotes += 1

            # Check fills against next observation
            next_price = s_next.best_bid or s_next.last_price or s_next.best_ask
            if next_price is None:
                continue

            # Bid fill: price dropped to our bid level
            if next_price <= our_bid and inventory * quote_size_usd < max_inventory_usd:
                # We bought YES at our_bid
                qty = quote_size_usd / our_bid
                inventory += qty
                inventory_cost += quote_size_usd
                # No maker fee

            # Ask fill: price rose to our ask level
            if next_price >= our_ask and inventory > 0:
                # We sold YES at our_ask
                sell_qty = min(inventory, quote_size_usd / our_ask)
                sell_value = sell_qty * our_ask
                buy_cost = sell_qty * (inventory_cost / inventory if inventory > 0 else our_bid)
                spread_pnl = sell_value - buy_cost
                total_spread_earned += spread_pnl
                inventory -= sell_qty
                if inventory > 0:
                    inventory_cost *= (1 - sell_qty / (inventory + sell_qty))
                else:
                    inventory_cost = 0

            max_inv = max(max_inv, abs(inventory * (s.best_bid or 0.5)))

        # Mark remaining inventory to market
        mkt = markets.get(mid)
        if mkt and inventory > 0:
            current_price = mkt.best_bid or mkt.last_price or mkt.best_ask or 0
            inventory_value = inventory * current_price
            inventory_pnl = inventory_value - inventory_cost
        else:
            inventory_pnl = 0

        res.spreads_earned = total_spread_earned
        res.inventory_pnl = inventory_pnl
        res.total_pnl = total_spread_earned + inventory_pnl
        res.max_inventory = max_inv
        res.avg_spread = spread_sum / res.num_quotes if res.num_quotes > 0 else 0

        results.append(res)

    return results


# ===================================================================
# STRATEGY 2: CROSS-OUTCOME ARBITRAGE
# ===================================================================

@dataclass
class ArbOpportunity:
    group_name: str
    markets: list[tuple[str, str, float, float]]  # (market_id, question, bid_price, ask_price)
    total_ask: float  # Cost to buy all at ask
    total_bid: float  # Value to sell all at bid
    guaranteed_payout: float  # Always $1.00 for mutually exclusive
    gross_profit_buy_all: float  # Payout - cost
    net_profit_after_fees: float  # After taker fees
    roi_pct: float
    capital_required: float
    is_complete: bool  # Do we have all outcomes?


def find_cross_outcome_arbs(markets: dict) -> list[ArbOpportunity]:
    """Find groups of mutually exclusive markets where prices don't sum to 1.0."""

    # Group markets by event
    groups: dict[str, list[MarketState]] = defaultdict(list)
    for mid, mkt in markets.items():
        q = mkt.question.lower()
        if "next prime minister of hungary" in q:
            groups["hungary_pm"].append(mkt)
        elif "2026 texas republican primary" in q:
            groups["texas_gop"].append(mkt)
        elif "win the 2025" in q and "champions league" in q:
            groups["ucl_winner"].append(mkt)
        elif "win the 2026 colombian presidential election" in q and "1st round" not in q:
            groups["colombia_final"].append(mkt)
        elif "1st round of the 2026 colombian" in q:
            groups["colombia_r1"].append(mkt)
        elif "win the 2025" in q and "premier league" in q:
            groups["epl_winner"].append(mkt)
        elif "nba eastern conference" in q:
            groups["nba_east"].append(mkt)
        elif "nba western conference" in q:
            groups["nba_west"].append(mkt)
        elif "nba rookie of the year" in q:
            groups["nba_roty"].append(mkt)
        elif "2025-26 serie a" in q or "2025–26 serie a" in q:
            groups["serie_a"].append(mkt)
        elif "next james bond" in q:
            groups["james_bond"].append(mkt)

    # Completeness heuristic: group is likely complete if:
    # - Has a leader > 0.20
    # - Sum of prices > 0.80
    # - At least 3 markets
    TAKER_FEE = 0.01  # 1% per side

    arbs = []
    for name, group in sorted(groups.items()):
        if len(group) < 2:
            continue

        mkts = []
        total_ask = 0.0
        total_bid = 0.0

        for m in sorted(group, key=lambda x: -(x.best_bid or x.last_price or 0)):
            bid = m.best_bid or 0
            ask = m.best_ask or m.last_price or 0.001
            mkts.append((m.market_id, m.question[:55], bid, ask))
            total_ask += ask
            total_bid += bid

        # Is this group likely complete?
        has_leader = any(m.best_bid and m.best_bid > 0.20 for m in group)
        is_complete = has_leader and total_ask > 0.80 and len(group) >= 3

        # Profit from buying all outcomes at ask price
        gross_profit = 1.0 - total_ask
        # Taker fee on entry only (maker exit at resolution = 0 fee)
        net_profit = gross_profit - total_ask * TAKER_FEE
        roi = (net_profit / total_ask * 100) if total_ask > 0 else 0

        arbs.append(ArbOpportunity(
            group_name=name,
            markets=mkts,
            total_ask=total_ask,
            total_bid=total_bid,
            guaranteed_payout=1.0,
            gross_profit_buy_all=gross_profit,
            net_profit_after_fees=net_profit,
            roi_pct=roi,
            capital_required=total_ask * 100,  # Per $100 payout
            is_complete=is_complete,
        ))

    return arbs


# ===================================================================
# STRATEGY 3: LLM BIAS CALIBRATION
# ===================================================================

@dataclass
class BiasStats:
    market_id: str
    question: str
    avg_model_prob: float
    avg_market_price: float
    avg_bias: float  # model - market (negative = bearish)
    bias_stdev: float
    observations: int
    # Signals where model deviates from its OWN bias pattern
    anomaly_signals: int
    anomaly_direction: str  # "more_bullish" or "more_bearish"


def analyze_llm_bias(signals: list[Signal]) -> tuple[float, list[BiasStats]]:
    """Analyze the LLM's systematic bias and find deviation signals.

    The model is consistently bearish. Instead of using its absolute predictions,
    we calibrate: when the model is LESS bearish than usual on a specific market,
    that's a genuine bullish signal (the model has extra reason to believe).
    """
    # Group by market
    by_market: dict[str, list[Signal]] = defaultdict(list)
    for sig in signals:
        if sig.best_bid and sig.best_bid > 0:
            by_market[sig.market_id].append(sig)

    # Global bias: average(model_prob - market_price) across all signals
    global_biases = []
    for sigs in by_market.values():
        for s in sigs:
            global_biases.append(s.model_prob - s.best_bid)

    if not global_biases:
        return 0.0, []

    global_avg_bias = sum(global_biases) / len(global_biases)

    stats = []
    for mid, sigs in by_market.items():
        if len(sigs) < 3:
            continue

        biases = [s.model_prob - s.best_bid for s in sigs]
        avg_bias = sum(biases) / len(biases)
        avg_model = sum(s.model_prob for s in sigs) / len(sigs)
        avg_market = sum(s.best_bid for s in sigs) / len(sigs)

        # Variance
        bias_var = sum((b - avg_bias) ** 2 for b in biases) / len(biases)
        bias_std = bias_var ** 0.5

        # Anomaly: signals where bias deviates from this market's average
        anomalies = 0
        direction = "neutral"
        for s in sigs:
            b = s.model_prob - s.best_bid
            if b > avg_bias + bias_std:
                anomalies += 1
                direction = "more_bullish"  # Model unusually bullish
            elif b < avg_bias - bias_std:
                anomalies += 1
                direction = "more_bearish"  # Model unusually bearish

        stats.append(BiasStats(
            market_id=mid,
            question=sigs[0].question[:55],
            avg_model_prob=avg_model,
            avg_market_price=avg_market,
            avg_bias=avg_bias,
            bias_stdev=bias_std,
            observations=len(sigs),
            anomaly_signals=anomalies,
            anomaly_direction=direction,
        ))

    return global_avg_bias, stats


# ===================================================================
# STRATEGY 4: INVENTORY-NEUTRAL SPREAD (YES+NO simultaneous)
# ===================================================================

def simulate_neutral_spread(markets: dict) -> list[dict]:
    """For each market, calculate the profit from simultaneously placing:
    - A YES limit buy at best_bid + improvement
    - A NO limit buy at (1 - best_ask) + improvement

    Since YES + NO = $1.00, if both fill, you lock in a profit equal to
    the spread minus your improvements, with ZERO directional exposure.

    Maker fee: 0% on both sides.
    """
    results = []
    for mid, mkt in markets.items():
        if not (mkt.best_bid and mkt.best_bid > 0 and mkt.best_ask and mkt.best_ask > 0):
            continue

        spread = mkt.best_ask - mkt.best_bid
        if spread < 0.015:
            continue

        # YES side: bid at best_bid + 0.005
        yes_bid = mkt.best_bid + 0.005
        # NO side: the NO best_bid is (1 - best_ask), we improve by 0.005
        no_bid = (1.0 - mkt.best_ask) + 0.005

        # Total cost if both fill: yes_bid + no_bid
        total_cost = yes_bid + no_bid
        # Guaranteed payout: $1.00 (one of YES/NO will be worth $1)
        guaranteed_payout = 1.0

        profit_per_pair = guaranteed_payout - total_cost
        # Maker fee = 0 for limit orders
        # Only pay fee on resolution payout? No - resolution is automatic

        if profit_per_pair <= 0:
            continue

        # Scale: how many pairs can we place?
        # Limited by depth on the thinner side
        size_usd = min(500.0, mkt.liquidity * 0.01)  # 1% of market depth
        num_pairs = size_usd / total_cost if total_cost > 0 else 0
        total_profit = num_pairs * profit_per_pair

        roi_pct = (profit_per_pair / total_cost) * 100

        results.append({
            "market_id": mid,
            "question": mkt.question[:55],
            "spread": spread,
            "yes_bid": yes_bid,
            "no_bid": no_bid,
            "total_cost": total_cost,
            "profit_per_pair": profit_per_pair,
            "roi_pct": roi_pct,
            "size_usd": size_usd,
            "total_profit": total_profit,
            "liquidity": mkt.liquidity,
        })

    results.sort(key=lambda x: -x["total_profit"])
    return results


# ===================================================================
# REPORTING
# ===================================================================

def main():
    base = Path(__file__).parent.parent
    print("Loading data...")
    signals = load_signals(str(base / "backtest_data.csv"))
    markets = load_markets(str(base / "backtest_markets.csv"))
    print(f"  {len(signals)} signals, {len(markets)} markets\n")

    # ---------------------------------------------------------------
    # 1. MARKET MAKING
    # ---------------------------------------------------------------
    print("=" * 100)
    print("STRATEGY 1: MARKET MAKING SIMULATION")
    print("  Earn the bid-ask spread. No prediction needed. 0% maker fee.")
    print("=" * 100)

    mm_results = simulate_market_making(signals, markets)
    mm_total = sum(r.total_pnl for r in mm_results)
    mm_spread_total = sum(r.spreads_earned for r in mm_results)

    print(f"\n{'Market':<65} {'Quotes':>6} {'Spread$':>8} {'InvP&L':>8} {'Total':>8} {'MaxInv':>8}")
    print("-" * 100)
    for r in sorted(mm_results, key=lambda x: -x.total_pnl):
        print(
            f"{r.question:<65} "
            f"{r.num_quotes:>6} "
            f"{r.spreads_earned:>+8.2f} "
            f"{r.inventory_pnl:>+8.2f} "
            f"{r.total_pnl:>+8.2f} "
            f"${r.max_inventory:>7.0f}"
        )
    print(f"\n  Total spread earned: ${mm_spread_total:+.2f}")
    print(f"  Total w/ inventory MTM: ${mm_total:+.2f}")
    print(f"  Note: Conservative fill model (15-min candles). Real fills would be more frequent.")

    # ---------------------------------------------------------------
    # 2. CROSS-OUTCOME ARBITRAGE
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STRATEGY 2: CROSS-OUTCOME ARBITRAGE")
    print("  Buy all outcomes in a group. One MUST win. Profit = $1 - cost.")
    print("  0% maker fee if placed as limit orders.")
    print("=" * 100)

    arbs = find_cross_outcome_arbs(markets)
    total_arb_profit = 0

    for arb in sorted(arbs, key=lambda a: -a.net_profit_after_fees):
        completeness = "COMPLETE" if arb.is_complete else "INCOMPLETE"
        actionable = arb.net_profit_after_fees > 0 and arb.is_complete

        print(f"\n  {arb.group_name} ({len(arb.markets)} markets) [{completeness}]")
        for mid, q, bid, ask in arb.markets:
            print(f"    {q:<58} bid={bid:.3f}  ask={ask:.3f}")
        print(f"    Cost (all at ask): ${arb.total_ask:.3f}")
        print(f"    Guaranteed payout: $1.000")
        print(f"    Gross profit:      ${arb.gross_profit_buy_all:+.3f}")
        print(f"    Net (after 1% fee): ${arb.net_profit_after_fees:+.3f}")
        print(f"    ROI:               {arb.roi_pct:+.1f}%")
        if actionable:
            print(f"    ** ACTIONABLE: invest ${arb.capital_required:.0f} for ${arb.net_profit_after_fees * 100:.0f} guaranteed profit **")
            total_arb_profit += arb.net_profit_after_fees * 100  # Per $100 unit

    print(f"\n  Total actionable arb profit (per $100 unit): ${total_arb_profit:.2f}")

    # ---------------------------------------------------------------
    # 3. LLM BIAS CALIBRATION
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STRATEGY 3: LLM BIAS CALIBRATION")
    print("  The model is consistently bearish. Use DEVIATION from its bias")
    print("  as the signal, not the absolute prediction.")
    print("=" * 100)

    global_bias, bias_stats = analyze_llm_bias(signals)
    print(f"\n  Global model bias: {global_bias:+.4f} (negative = consistently bearish)")
    print(f"  The model underestimates prices by {abs(global_bias)*100:.1f} cents on average")
    print()

    print(f"  {'Market':<58} {'AvgModel':>8} {'AvgMkt':>7} {'Bias':>7} {'StdDev':>7} {'Anomalies':>9}")
    print("  " + "-" * 98)
    for s in sorted(bias_stats, key=lambda x: -abs(x.avg_bias)):
        print(
            f"  {s.question:<58} "
            f"{s.avg_model_prob:>8.3f} "
            f"{s.avg_market_price:>7.3f} "
            f"{s.avg_bias:>+7.3f} "
            f"{s.bias_stdev:>7.4f} "
            f"{s.anomaly_signals:>5}/{s.observations}"
        )

    # Calibrated signals: correct for known bias
    print(f"\n  Calibration approach:")
    print(f"    calibrated_prob = model_prob - global_bias ({global_bias:+.4f})")
    print(f"    calibrated_edge = calibrated_prob - market_price")
    print(f"    This removes the systematic component; remaining edge is informational")

    # Simulate calibrated trades
    calibrated_trades = 0
    calibrated_pnl = 0.0
    for mid, sigs in defaultdict(list, {s.market_id: [] for s in signals}).items():
        pass  # Would need price series for proper backtest

    for s in bias_stats:
        calibrated_prob = s.avg_model_prob - global_bias
        calibrated_edge = calibrated_prob - s.avg_market_price
        if abs(calibrated_edge) > 0.03:
            calibrated_trades += 1
            print(f"    {s.question[:50]}: raw_edge={s.avg_bias:+.3f} -> calibrated_edge={calibrated_edge:+.3f}")

    # ---------------------------------------------------------------
    # 4. INVENTORY-NEUTRAL SPREAD
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STRATEGY 4: INVENTORY-NEUTRAL SPREAD (YES + NO)")
    print("  Buy YES at bid+improvement AND NO at (1-ask)+improvement simultaneously.")
    print("  Total cost < $1.00 = guaranteed profit with ZERO directional risk.")
    print("  0% maker fee on both sides.")
    print("=" * 100)

    spreads = simulate_neutral_spread(markets)
    total_neutral_profit = sum(s["total_profit"] for s in spreads)

    if spreads:
        print(f"\n  {'Market':<58} {'Spread':>6} {'Cost':>7} {'Profit':>7} {'ROI%':>6} {'TotalP':>8}")
        print("  " + "-" * 95)
        for s in spreads[:15]:
            print(
                f"  {s['question']:<58} "
                f"{s['spread']:>6.3f} "
                f"${s['total_cost']:>5.3f} "
                f"${s['profit_per_pair']:>5.3f} "
                f"{s['roi_pct']:>5.1f}% "
                f"${s['total_profit']:>7.2f}"
            )
        print(f"\n  Total profit across all markets: ${total_neutral_profit:.2f}")
        print(f"  Note: requires BOTH sides to fill. Profit is locked until resolution.")
    else:
        print(f"\n  No profitable neutral spread opportunities found at current spreads.")

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print("STRATEGY COMPARISON — STRUCTURAL (non-predictive) EDGES")
    print("=" * 100)
    print(f"\n  {'Strategy':<40} {'Est. P&L':>10} {'Risk':>15} {'Requires':>20}")
    print("  " + "-" * 90)
    print(f"  {'1. Market Making':<40} {'$' + f'{mm_total:.2f}':>10} {'Inventory':>15} {'Live mode':>20}")
    print(f"  {'2. Cross-Outcome Arb':<40} {'$' + f'{total_arb_profit:.2f}':>10} {'Capital lockup':>15} {'Live mode':>20}")
    print(f"  {'3. LLM Bias Calibration':<40} {'TBD':>10} {'Directional':>15} {'More data':>20}")
    print(f"  {'4. Inventory-Neutral Spread':<40} {'$' + f'{total_neutral_profit:.2f}':>10} {'Fill risk':>15} {'Live mode':>20}")

    total = mm_total + total_arb_profit + total_neutral_profit
    print(f"\n  Combined estimated edge: ${total:.2f}")
    print(f"\n  Key insight: NONE of these require predicting outcomes correctly.")
    print(f"  They exploit market STRUCTURE: spreads, fee asymmetry, and pricing constraints.")
    print("=" * 100)


if __name__ == "__main__":
    main()
