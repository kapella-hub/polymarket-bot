#!/usr/bin/env python3
"""
Daily Compounder -- Near-Expiry Edge Trading

TRACK 1 -- Crypto Threshold (<24h):
  Parse "Will BTC close above $X?" markets. Compare live Binance price.
  Gap requirements: <2h->2%, <6h->5%, <12h->8%, <24h->12%.
  Near-certain outcome = trade without LLM.

TRACK 2 -- LLM Near-Expiry (1-72h, up to 10/cycle):
  Claude + web search. Min edge 8c, min confidence 65%.

Kelly sizing: fraction = edge/odds, capped 20% bankroll. $3-$30/trade.
Scan interval: 30 min default.
"""

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import httpx
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

import structlog
logger = structlog.get_logger()

LOG_FILE = Path(__file__).parent / "daily_compounder_output.log"
STATE_FILE = Path(__file__).parent / "data" / "daily_compounder_state.json"

ASSET_MAP = {
    "BTC": "BTCUSDT", "BITCOIN": "BTCUSDT",
    "ETH": "ETHUSDT", "ETHEREUM": "ETHUSDT",
    "SOL": "SOLUSDT", "SOLANA": "SOLUSDT",
    "XRP": "XRPUSDT",
    "BNB": "BNBUSDT",
}

CRYPTO_REGEX = re.compile(
    r"(?i)(?:will\s+)?(BTC|ETH|SOL|XRP|BNB|bitcoin|ethereum|solana)\s+"
    r"(?:close\s+|be\s+|end\s+|finish\s+|trade\s+)?(?:at\s+)?"
    r"(above|below|over|under|higher\s+than|lower\s+than|exceed)\s+\$?([\d,]+(?:\.\d+)?)",
    re.IGNORECASE,
)


def create_clob():
    creds = ApiCreds(
        api_key=os.getenv("PM_POLYMARKET_API_KEY"),
        api_secret=os.getenv("PM_POLYMARKET_API_SECRET"),
        api_passphrase=os.getenv("PM_POLYMARKET_API_PASSPHRASE"),
    )
    clob = ClobClient(
        "https://clob.polymarket.com",
        key=os.getenv("PM_POLYMARKET_WALLET_PRIVATE_KEY"),
        chain_id=137, signature_type=0,
    )
    clob.set_api_creds(creds)
    return clob


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"bankroll": 100.0, "trades": [], "total_invested": 0.0, "total_returned": 0.0}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


async def get_binance_prices() -> dict:
    symbols = list(set(ASSET_MAP.values()))
    prices = {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Compact JSON (no spaces) required by Binance symbols param
            resp = await client.get(
                "https://api.binance.com/api/v3/ticker/price",
                params={"symbols": json.dumps(symbols, separators=(",", ":"))},
            )
            data = resp.json()
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "symbol" in item:
                        prices[item["symbol"]] = float(item["price"])
            elif isinstance(data, dict) and "symbol" in data:
                prices[data["symbol"]] = float(data["price"])
            if not prices:
                # Fallback: fetch each symbol individually
                for sym in symbols:
                    try:
                        r = await client.get(
                            "https://api.binance.com/api/v3/ticker/price",
                            params={"symbol": sym},
                        )
                        d = r.json()
                        if isinstance(d, dict) and "price" in d:
                            prices[sym] = float(d["price"])
                    except Exception:
                        pass
    except Exception as e:
        logger.warning("binance_error", error=str(e))
    return prices


def parse_crypto_market(question: str) -> dict | None:
    m = CRYPTO_REGEX.search(question)
    if not m:
        return None
    raw = m.group(1).upper()
    symbol = ASSET_MAP.get(raw)
    if not symbol:
        return None
    direction = m.group(2).lower()
    above = direction in ("above", "over", "exceed", "higher than")
    threshold = float(m.group(3).replace(",", ""))
    return {"symbol": symbol, "above": above, "threshold": threshold}


def crypto_edge(parsed: dict, prices: dict, hours_left: float, yes_price: float) -> dict | None:
    current = prices.get(parsed["symbol"])
    if not current:
        return None
    threshold = parsed["threshold"]
    gap_pct = abs(current - threshold) / threshold
    # Required gap shrinks as expiry approaches
    if hours_left < 2:
        min_gap = 0.02
    elif hours_left < 6:
        min_gap = 0.05
    elif hours_left < 12:
        min_gap = 0.08
    else:
        min_gap = 0.12
    if gap_pct < min_gap:
        return None
    currently_above = current > threshold
    above = parsed["above"]
    yes_wins = (above and currently_above) or (not above and not currently_above)
    side = "buy_yes" if yes_wins else "buy_no"
    market_price = yes_price if yes_wins else (1.0 - yes_price)
    if market_price >= 0.92:
        return None
    confidence = min(0.95, 0.60 + gap_pct * 2.5 - hours_left * 0.005)
    edge = confidence - market_price
    if edge < 0.08:
        return None
    return {
        "side": side, "edge": edge, "confidence": confidence,
        "current_price": current, "threshold": threshold, "gap_pct": gap_pct,
    }


async def fetch_near_expiry_markets() -> list:
    now = datetime.now(timezone.utc)
    candidates = []
    async with httpx.AsyncClient(timeout=20) as client:
        all_markets = []
        for offset in [0, 100, 200, 300]:
            try:
                resp = await client.get("https://gamma-api.polymarket.com/markets", params={
                    "closed": "false", "active": "true", "limit": 100, "offset": offset,
                })
                data = resp.json()
                if isinstance(data, list):
                    all_markets.extend(data)
                if len(data) < 100:
                    break
                await asyncio.sleep(0.3)
            except Exception as e:
                logger.warning("fetch_error", offset=offset, error=str(e))
        for m in all_markets:
            try:
                end_str = m.get("endDate")
                if not end_str:
                    continue
                end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                hours_left = (end_date - now).total_seconds() / 3600
                if hours_left < 1 or hours_left > 96:
                    continue
                prices = m.get("outcomePrices", [])
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if not prices or len(prices) < 2:
                    continue
                yes_price = float(prices[0])
                if yes_price < 0.05 or yes_price > 0.95:
                    continue
                volume = float(m.get("volumeNum", 0))
                min_vol = 2000 if hours_left < 24 else 3000
                if volume < min_vol:
                    continue
                best_bid = float(m["bestBid"]) if m.get("bestBid") else None
                best_ask = float(m["bestAsk"]) if m.get("bestAsk") else None
                if best_bid is None or best_ask is None:
                    continue
                if best_ask - best_bid > 0.12:
                    continue
                tokens = m.get("clobTokenIds", [])
                if isinstance(tokens, str):
                    tokens = json.loads(tokens)
                if len(tokens) < 2:
                    continue
                candidates.append({
                    "id": m.get("conditionId", ""),
                    "question": m.get("question", ""),
                    "category": m.get("category", ""),
                    "yes_price": yes_price,
                    "no_price": float(prices[1]),
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "volume": volume,
                    "hours_left": hours_left,
                    "end_date": end_str[:16],
                    "yes_token_id": str(tokens[0]),
                    "no_token_id": str(tokens[1]),
                    "description": m.get("description", "")[:200],
                })
            except Exception:
                continue
    candidates.sort(key=lambda c: c["hours_left"])
    return candidates


async def claude_analyze_near_expiry(market: dict) -> dict | None:
    prompt = (
        "You are a superforecaster analyzing a prediction market expiring SOON. "
        "Use web search to find the latest relevant data.\n\n"
        "Market: " + market["question"] + "\n"
        "Category: " + market["category"] + "\n"
        "Current YES price: $" + "%.3f" % market["yes_price"] +
        " (%.0f%% per market)\n" % (market["yes_price"] * 100) +
        "Time left: %.1f hours\n" % market["hours_left"] +
        "Volume: $%s\n" % "{:,.0f}".format(market["volume"])
    )
    if market["description"]:
        prompt += "Description: " + market["description"] + "\n"
    prompt += (
        "\nThis resolves soon. Search for the LATEST news. "
        "If the outcome is nearly certain, indicate that.\n\n"
        "Output ONLY valid JSON:\n"
        "{\"probability\": 0.XX, \"confidence\": 0.XX, "
        "\"reasoning\": \"2-3 sentences\", \"near_certain\": true/false}"
    )
    run_args = [
        "claude", "-p", prompt,
        "--output-format", "text",
        "--model", os.getenv("PM_LLM_MODEL", "haiku"),
        "--allowedTools", "WebSearch,WebFetch",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *run_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=90)
        if proc.returncode != 0:
            return None
        output = stdout.decode("utf-8", errors="replace").strip()
        pat = re.compile(r"\{[^{}]*\"probability\"[^{}]*\}", re.DOTALL)
        for candidate in [output, pat.search(output)]:
            if candidate is None:
                continue
            text = candidate if isinstance(candidate, str) else candidate.group()
            try:
                data = json.loads(text)
                if "probability" in data:
                    prob = float(data["probability"])
                    if 0.0 <= prob <= 1.0:
                        return data
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
        return None
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        logger.error("claude_error", error=str(e))
        return None


def kelly_size(bankroll: float, edge: float, price: float, boost: float = 1.0) -> float:
    if price <= 0 or price >= 1:
        return 0.0
    odds = (1.0 - price) / price
    kelly_f = max(0.0, min((edge / odds) * boost, 0.20))
    return max(3.0, min(bankroll * kelly_f, 30.0))


async def execute_trade(clob, market: dict, side: str, size_usd: float,
                        track: str, state: dict) -> bool:
    token_id = market["yes_token_id"] if side == "buy_yes" else market["no_token_id"]
    price = market["best_ask"] if side == "buy_yes" else (1.0 - market["best_bid"])
    if price <= 0 or price >= 0.95:
        return False
    tokens = size_usd / price
    try:
        order_args = OrderArgs(token_id=token_id, price=price, size=round(tokens, 2), side=BUY)
        signed = clob.create_order(order_args)
        result = clob.post_order(signed, OrderType.GTC)
        order_id = result.get("orderID", "")
        if not order_id:
            logger.warning("no_order_id", resp=str(result)[:100])
            return False
        state["trades"].append({
            "market_id": market["id"],
            "question": market["question"],
            "side": side,
            "token_id": token_id,
            "price": price,
            "size_usd": round(size_usd, 2),
            "tokens": round(tokens, 2),
            "order_id": order_id,
            "end_date": market["end_date"],
            "hours_left_at_entry": round(market["hours_left"], 1),
            "track": track,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "status": "open",
        })
        state["bankroll"] -= size_usd
        state["total_invested"] = state.get("total_invested", 0) + size_usd
        save_state(state)
        logger.info(
            "DAILY_TRADE",
            track=track,
            question=market["question"][:52],
            side=side,
            price="$%.3f" % price,
            size="$%.2f" % size_usd,
            hours_left="%.1fh" % market["hours_left"],
            bankroll="$%.2f" % state["bankroll"],
            order_id=order_id[:16],
        )
        return True
    except Exception as e:
        logger.error("trade_error", error=str(e))
        return False


async def run_cycle(clob, state: dict, max_trades: int = 5):
    logger.info("daily_cycle_start", bankroll="$%.2f" % state["bankroll"])
    prices, markets = await asyncio.gather(
        get_binance_prices(),
        fetch_near_expiry_markets(),
    )
    logger.info(
        "cycle_data",
        markets=len(markets),
        btc="$%s" % "{:,.0f}".format(prices.get("BTCUSDT", 0)),
        eth="$%s" % "{:,.0f}".format(prices.get("ETHUSDT", 0)),
    )
    if not markets:
        return
    trades_placed = 0
    llm_calls = 0
    max_llm_per_cycle = 10
    open_ids = {t["market_id"] for t in state["trades"] if t["status"] == "open"}
    for market in markets:
        if trades_placed >= max_trades or state["bankroll"] < 5:
            break
        if market["id"] in open_ids:
            continue
        # TRACK 1: Crypto threshold -- deterministic
        parsed = parse_crypto_market(market["question"])
        if parsed and market["hours_left"] < 24 and prices:
            signal = crypto_edge(parsed, prices, market["hours_left"], market["yes_price"])
            if signal:
                side = signal["side"]
                price_val = (market["best_ask"] if side == "buy_yes"
                             else (1.0 - market["best_bid"]))
                size = kelly_size(state["bankroll"], signal["edge"], price_val)
                logger.info(
                    "crypto_signal",
                    q=market["question"][:52],
                    current="$%s" % "{:,.0f}".format(signal["current_price"]),
                    threshold="$%s" % "{:,.0f}".format(signal["threshold"]),
                    gap="%.1f%%" % (signal["gap_pct"] * 100),
                    hours="%.1fh" % market["hours_left"],
                    edge="%+.3f" % signal["edge"],
                    side=side,
                    size="$%.2f" % size,
                )
                if size >= 3 and size <= state["bankroll"] - 3:
                    if await execute_trade(clob, market, side, size, "crypto", state):
                        trades_placed += 1
            continue  # Skip LLM for crypto markets -- math is more reliable
        # TRACK 2: LLM near-expiry
        if llm_calls >= max_llm_per_cycle:
            continue
        llm_calls += 1
        logger.info(
            "llm_analyzing",
            q=market["question"][:52],
            hours="%.1fh" % market["hours_left"],
            vol="$%s" % "{:,.0f}".format(market["volume"]),
        )
        analysis = await claude_analyze_near_expiry(market)
        if not analysis:
            continue
        prob = float(analysis.get("probability", 0.5))
        confidence = float(analysis.get("confidence", 0.5))
        near_certain = bool(analysis.get("near_certain", False))
        reasoning = analysis.get("reasoning", "")
        edge_yes = prob - market["yes_price"]
        edge_no = (1 - prob) - market["no_price"]
        logger.info(
            "llm_signal",
            q=market["question"][:45],
            claude="%.2f" % prob,
            market_p="%.2f" % market["yes_price"],
            edge_yes="%+.3f" % edge_yes,
            edge_no="%+.3f" % edge_no,
            conf="%.2f" % confidence,
            hours="%.1fh" % market["hours_left"],
            certain=near_certain,
            why=reasoning[:60],
        )
        if confidence < 0.60:
            continue
        if edge_yes > 0.06:
            side, edge = "buy_yes", edge_yes
        elif edge_no > 0.06:
            side, edge = "buy_no", edge_no
        else:
            continue
        price_val = (market["best_ask"] if side == "buy_yes"
                     else (1.0 - market["best_bid"]))
        boost = 1.5 if near_certain else 1.0
        size = kelly_size(state["bankroll"], edge, price_val, boost=boost)
        if size >= 3 and size <= state["bankroll"] - 3:
            if await execute_trade(clob, market, side, size, "llm", state):
                trades_placed += 1
    logger.info(
        "daily_cycle_done",
        placed=trades_placed,
        llm_calls=llm_calls,
        bankroll="$%.2f" % state["bankroll"],
    )


async def main():
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    initial_bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else None
    scan_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 1800  # 30 min
    clob = create_clob()
    state = load_state()
    if initial_bankroll is not None:
        state["bankroll"] = initial_bankroll
        save_state(state)
    print("[%s] Daily Compounder starting" % datetime.now(timezone.utc).isoformat())
    print("  Duration: %dh | Bankroll: $%.2f" % (hours, state["bankroll"]))
    print("  Scan: every %d min | Max 5 trades/cycle" % (scan_interval // 60))
    print("  Track 1: Crypto thresholds (<24h, gap-based confidence)")
    print("  Track 2: LLM near-expiry (1-72h, >=8c edge, >=65%% conf)")
    print("  Sizing: Kelly fraction $3-30, 20%% bankroll cap")
    print()
    start = time.time()
    cycle = 0
    while time.time() - start < hours * 3600:
        cycle += 1
        try:
            await run_cycle(clob, state)
        except Exception as e:
            logger.error("cycle_error", cycle=cycle, error=str(e))
        open_trades = [t for t in state["trades"] if t["status"] == "open"]
        logger.info(
            "daily_status",
            cycle=cycle,
            bankroll="$%.2f" % state["bankroll"],
            open=len(open_trades),
            total=len(state["trades"]),
        )
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": cycle, "bankroll": state["bankroll"],
                "open_trades": len(open_trades), "total_trades": len(state["trades"]),
            }) + "\n")
        remaining = (start + hours * 3600) - time.time()
        if remaining > scan_interval:
            logger.info("daily_sleeping", next="%dmin" % (scan_interval // 60))
            await asyncio.sleep(scan_interval)
        elif remaining > 60:
            await asyncio.sleep(remaining)
    closed = [t for t in state["trades"] if t.get("status") != "open"]
    print("\n" + "=" * 60)
    print("DAILY COMPOUNDER RESULTS")
    print("=" * 60)
    print("  Final bankroll: $%.2f" % state["bankroll"])
    print("  Total trades:   %d (%d closed)" % (len(state["trades"]), len(closed)))
    print("=" * 60)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "final": True, "bankroll": state["bankroll"],
            "total_trades": len(state["trades"]),
        }) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
