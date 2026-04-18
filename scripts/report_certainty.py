#!/usr/bin/env python3
"""Summarize certainty sniper fill quality and realized performance."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
FILL_LOG = ROOT / "data" / "certainty_fill_journal.jsonl"
STATE_FILE = ROOT / "data" / "certainty_sniper_state.json"


def load_fill_attempts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[warn] skipped malformed journal line {line_no}", file=sys.stderr)
    return rows


def load_trades(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("[warn] certainty state file is malformed", file=sys.stderr)
        return []
    return state.get("trades", [])


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def price_bucket(price: float | None) -> str:
    if price is None or price <= 0:
        return "unknown"
    start = math.floor(price * 20) / 20.0
    end = start + 0.05
    return f"{start:.2f}-{end:.2f}"


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def usd(x: float) -> str:
    return f"${x:.2f}"


def build_joined_rows(fills: list[dict], trades: list[dict]) -> list[dict]:
    trades_by_order = {
        t.get("order_id"): t
        for t in trades
        if t.get("order_id")
    }
    rows = []
    for fill in fills:
        order_id = fill.get("order_id") or fill.get("raw", {}).get("orderID") or fill.get("raw", {}).get("id")
        trade = trades_by_order.get(order_id)
        rows.append({
            "coin": fill.get("coin"),
            "order_id": order_id,
            "requested_price": to_float(fill.get("requested_price")),
            "fill_price": to_float(fill.get("fill_price")),
            "requested_notional": to_float(fill.get("requested_notional")),
            "filled_notional": to_float(fill.get("filled_notional")),
            "requested_tokens": to_float(fill.get("requested_tokens")),
            "filled_tokens": to_float(fill.get("filled_tokens")),
            "ask_usd_vol": to_float(fill.get("ask_usd_vol")),
            "status": trade.get("status") if trade else None,
            "pnl": to_float(trade.get("pnl")) if trade else None,
            "entry_price": to_float(trade.get("entry_price")) if trade else None,
            "move_pct": to_float(trade.get("move_pct")) if trade else None,
        })
    return rows


def to_float(value) -> float:
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def summarize(rows: list[dict]) -> dict:
    by_coin = defaultdict(list)
    by_bucket = defaultdict(list)
    for row in rows:
        by_coin[row["coin"]].append(row)
        by_bucket[price_bucket(row["requested_price"])].append(row)
    return {"by_coin": by_coin, "by_bucket": by_bucket}


def print_overview(fills: list[dict], trades: list[dict], rows: list[dict]) -> None:
    fill_count = len(fills)
    trade_count = len(trades)
    matched = sum(1 for row in rows if row["status"])
    full_fills = sum(1 for row in rows if row["filled_tokens"] and row["requested_tokens"] and row["filled_tokens"] >= row["requested_tokens"] * 0.999)
    any_fills = sum(1 for row in rows if row["filled_notional"] > 0)
    settled = [t for t in trades if t.get("status") in ("won", "lost")]
    wins = sum(1 for t in settled if t.get("status") == "won")
    pnl = sum(to_float(t.get("pnl")) for t in settled)

    print("=== CERTAINTY REPORT ===")
    print(f"Fill attempts:      {fill_count}")
    print(f"Any fill:           {any_fills} ({pct(safe_div(any_fills, fill_count)) if fill_count else '--'})")
    print(f"Full fill:          {full_fills} ({pct(safe_div(full_fills, fill_count)) if fill_count else '--'})")
    print(f"State trades:       {trade_count}")
    print(f"Matched by order:   {matched}")
    print(f"Settled trades:     {len(settled)} ({wins}W/{len(settled) - wins}L)")
    print(f"Realized P&L:       {usd(pnl)}")
    print()


def print_group_table(title: str, groups: dict[str, list[dict]]) -> None:
    print(title)
    print("key          attempts  any_fill  full_fill  avg_slip  avg_notional  settled_wr  avg_pnl")
    for key in sorted(groups):
        rows = groups[key]
        attempts = len(rows)
        any_fill = sum(1 for row in rows if row["filled_notional"] > 0)
        full_fill = sum(
            1
            for row in rows
            if row["filled_tokens"] and row["requested_tokens"]
            and row["filled_tokens"] >= row["requested_tokens"] * 0.999
        )
        slippages = [
            row["fill_price"] - row["requested_price"]
            for row in rows
            if row["fill_price"] > 0 and row["requested_price"] > 0 and row["filled_notional"] > 0
        ]
        settled = [row for row in rows if row["status"] in ("won", "lost")]
        wins = sum(1 for row in settled if row["status"] == "won")
        avg_pnl = avg([row["pnl"] for row in settled if row["pnl"] is not None])
        avg_notional = avg([row["filled_notional"] for row in rows if row["filled_notional"] > 0])
        settled_wr = pct(safe_div(wins, len(settled))) if settled else "--"
        avg_slip = f"{avg(slippages):+.4f}" if slippages else "--"
        print(
            f"{key:<12} {attempts:>8}  {pct(safe_div(any_fill, attempts)):>8}  "
            f"{pct(safe_div(full_fill, attempts)):>9}  {avg_slip:>8}  "
            f"{usd(avg_notional):>12}  {settled_wr:>9}  {usd(avg_pnl):>7}"
        )
    print()


def print_threshold_hints(rows: list[dict]) -> None:
    if not rows:
        return
    print("Threshold hints")
    poor_fill = [r for r in rows if r["requested_notional"] > 0 and safe_div(r["filled_notional"], r["requested_notional"]) < 0.5]
    pricey = [r for r in rows if r["requested_price"] >= 0.90 and r["status"] in ("won", "lost")]
    if poor_fill:
        by_coin = defaultdict(list)
        for row in poor_fill:
            by_coin[row["coin"]].append(row)
        for coin in sorted(by_coin):
            avg_ask = avg([r["ask_usd_vol"] for r in by_coin[coin] if r["ask_usd_vol"] > 0])
            print(f"{coin}: low-fill attempts suggest raising book minimum toward {usd(avg_ask)} or cutting `BOOK_TAKE_PCT`")
    if pricey:
        wr = safe_div(sum(1 for r in pricey if r["status"] == "won"), len(pricey))
        avg_pnl = avg([r["pnl"] for r in pricey if r["pnl"] is not None])
        print(f"requested price >= 0.90: WR {pct(wr)}, avg pnl {usd(avg_pnl)}")
    print()


def main() -> None:
    fills = load_fill_attempts(FILL_LOG)
    trades = load_trades(STATE_FILE)
    if not fills and not trades:
        print("No certainty data yet.")
        print(f"Expected files: {FILL_LOG} and/or {STATE_FILE}")
        return

    rows = build_joined_rows(fills, trades)
    summary = summarize(rows)

    print_overview(fills, trades, rows)
    if rows:
        print_group_table("By coin", summary["by_coin"])
        print_group_table("By requested price bucket", summary["by_bucket"])
        print_threshold_hints(rows)


if __name__ == "__main__":
    main()
