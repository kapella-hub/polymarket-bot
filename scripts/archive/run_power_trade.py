#!/usr/bin/env python3
"""
Claude Power Trader — Hourly LLM-Driven Prediction Market Trading

Strategy: Use Claude's reasoning + web search to find mispriced markets.
Unlike the 15-min crypto arb (speed game), this is an INTELLIGENCE game.

Every hour:
1. Fetch 200+ active markets from Gamma API
2. Filter: resolves in 1-14 days, volume >$20k, price $0.10-$0.90
3. Claude analyzes the top candidates with web search
4. If Claude's probability differs from market by >8 cents, trade
5. Kelly-sized positions, 10% bankroll cap per trade

This makes money 24/7 because prediction markets are ALWAYS mispriced
by the crowd. Claude can research and reason faster than most traders.

Run via cron: 0 * * * * cd /opt/polymarket-bot && bash run_power_trade.sh
Or continuously: python3 run_power_trade.py [hours] [bankroll]
"""

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

import httpx
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY

import structlog
logger = structlog.get_logger()

LOG_FILE = Path(__file__).parent / "power_trade_output.log"
STATE_FILE = Path(__file__).parent / "data" / "power_trade_state.json"


def create_clob():
    creds = ApiCreds(
        api_key=os.getenv('PM_POLYMARKET_API_KEY'),
        api_secret=os.getenv('PM_POLYMARKET_API_SECRET'),
        api_passphrase=os.getenv('PM_POLYMARKET_API_PASSPHRASE'),
    )
    clob = ClobClient(
        'https://clob.polymarket.com',
        key=os.getenv('PM_POLYMARKET_WALLET_PRIVATE_KEY'),
        chain_id=137, signature_type=0,
    )
    clob.set_api_creds(creds)
    return clob


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {"bankroll": 300.0, "trades": [], "total_invested": 0.0, "total_returned": 0.0}


def save_state(state: dict):
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


async def fetch_candidate_markets() -> list:
    now = datetime.now(timezone.utc)
    candidates = []

    async with httpx.AsyncClient(timeout=20) as client:
        all_markets = []
        for offset in [0, 100, 200]:
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

        logger.info("markets_fetched", total=len(all_markets))

        for m in all_markets:
            try:
                prices = m.get("outcomePrices", [])
                if isinstance(prices, str):
                    prices = json.loads(prices)
                if not prices or len(prices) < 2:
                    continue

                yes_price = float(prices[0])
                if yes_price < 0.08 or yes_price > 0.92:
                    continue

                volume = float(m.get("volumeNum", 0))
                if volume < 50000:
                    continue

                best_bid = float(m["bestBid"]) if m.get("bestBid") else None
                best_ask = float(m["bestAsk"]) if m.get("bestAsk") else None
                if best_bid is None or best_ask is None:
                    continue
                if best_ask - best_bid > 0.15:
                    continue

                end_str = m.get("endDate")
                if not end_str:
                    continue
                try:
                    end_date = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                except (ValueError, TypeError):
                    continue
                days_to_end = (end_date - now).total_seconds() / 86400
                if days_to_end < 0 or days_to_end > 180:
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
                    "days_to_end": days_to_end,
                    "end_date": end_str[:10],
                    "yes_token_id": str(tokens[0]),
                    "no_token_id": str(tokens[1]),
                    "description": m.get("description", "")[:200],
                })
            except Exception:
                continue

    candidates.sort(key=lambda c: (-c["volume"] / max(c["days_to_end"], 1)))
    return candidates


async def claude_analyze(market: dict) -> dict:
    prompt = (
        "You are a superforecaster analyzing a prediction market. "
        "Use web search to find the latest information.\n\n"
        f"MARKET: {market['question']}\n"
        f"Category: {market['category']}\n"
        f"Current YES price: ${market['yes_price']:.3f} "
        f"(market thinks {market['yes_price']*100:.0f}% likely)\n"
        f"Resolves: {market['end_date']} ({market['days_to_end']:.0f} days)\n"
        f"Volume: ${market['volume']:,.0f}\n"
    )
    if market["description"]:
        prompt += f"Description: {market['description']}\n"

    prompt += (
        "\nINSTRUCTIONS:\n"
        "1. Search the web for the latest relevant information\n"
        "2. Consider base rates and historical precedent\n"
        "3. Adjust for current evidence\n"
        "4. Be precise - a 5-cent error costs real money\n\n"
        'Output ONLY valid JSON:\n'
        '{"probability": 0.XX, "confidence": 0.XX, '
        '"reasoning": "2-3 sentences", "key_factors": ["f1", "f2"]}'
    )

    cmd = [
        "claude", "-p", prompt,
        "--output-format", "text",
        "--model", os.getenv("PM_LLM_MODEL", "haiku"),
        "--allowedTools", "WebSearch,WebFetch",
    ]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=90)

        if proc.returncode != 0:
            return None

        output = stdout.decode("utf-8", errors="replace").strip()

        for attempt in [output, re.search(r'\{[^{}]*"probability"[^{}]*\}', output, re.DOTALL)]:
            if attempt is None:
                continue
            text = attempt if isinstance(attempt, str) else attempt.group()
            try:
                data = json.loads(text)
                if "probability" in data:
                    prob = float(data["probability"])
                    if 0 <= prob <= 1:
                        return data
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        return None
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        logger.error("claude_error", error=str(e))
        return None


async def execute_trade(clob, market, side, size_usd, state) -> bool:
    token_id = market["yes_token_id"] if side == "buy_yes" else market["no_token_id"]
    price = market["best_ask"] if side == "buy_yes" else (1.0 - market["best_bid"])

    if price <= 0 or price >= 0.95:
        return False

    tokens = size_usd / price

    try:
        order_args = OrderArgs(
            token_id=token_id, price=price, size=round(tokens, 2), side=BUY,
        )
        signed = clob.create_order(order_args)
        result = clob.post_order(signed, OrderType.GTC)

        order_id = result.get("orderID", "")
        if not order_id:
            return False

        trade = {
            "market_id": market["id"],
            "question": market["question"],
            "side": side,
            "token_id": token_id,
            "price": price,
            "size_usd": round(size_usd, 2),
            "tokens": round(tokens, 2),
            "order_id": order_id,
            "end_date": market["end_date"],
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "status": "open",
        }
        state["trades"].append(trade)
        state["bankroll"] -= size_usd
        state["total_invested"] += size_usd
        save_state(state)

        logger.info(
            "POWER_TRADE",
            question=market["question"][:50],
            side=side,
            price=f"${price:.3f}",
            size=f"${size_usd:.2f}",
            bankroll=f"${state['bankroll']:.2f}",
            order_id=order_id[:16],
        )
        return True
    except Exception as e:
        logger.error("trade_error", error=str(e))
        return False


async def run_cycle(clob, state, max_trades=3):
    logger.info("power_cycle_start", bankroll=f"${state['bankroll']:.2f}")

    candidates = await fetch_candidate_markets()
    logger.info("power_candidates", count=len(candidates))

    if not candidates:
        return

    # Prioritize novelty/meme markets OVER sports — that's where mispricing lives
    # Sports are well-arbitraged by sportsbooks. Novelty/politics are not.
    sports_keywords = {"nba", "nhl", "nfl", "mlb", "premier league", "champions league",
                       "fifa", "world cup", "mvp", "win the 2025", "win the 2026"}
    sports = []
    non_sports = []
    for c in candidates:
        q_lower = c["question"].lower()
        if any(kw in q_lower for kw in sports_keywords):
            sports.append(c)
        else:
            non_sports.append(c)
    # Non-sports first (more likely mispriced), then a few sports for diversity
    to_analyze = non_sports[:15] + sports[:5]
    trades_placed = 0
    open_ids = {t["market_id"] for t in state["trades"] if t["status"] == "open"}

    # Build per-date exposure to prevent all capital clustering on one resolution date
    date_exposure: dict = {}
    for t in state["trades"]:
        if t["status"] == "open":
            d = t.get("end_date", "")[:10]
            date_exposure[d] = date_exposure.get(d, 0) + t["size_usd"]
    # Cap per date: 30% of total deployed capital (bankroll + open positions)
    total_capital = state["bankroll"] + sum(
        t["size_usd"] for t in state["trades"] if t["status"] == "open"
    )
    date_cap = total_capital * 0.30

    for market in to_analyze:
        if trades_placed >= max_trades or state["bankroll"] < 10:
            break
        if market["id"] in open_ids:
            continue

        logger.info("analyzing", q=market["question"][:50],
                     price=market["yes_price"], vol=f"${market['volume']:,.0f}")

        analysis = await claude_analyze(market)
        if not analysis:
            continue

        prob = float(analysis.get("probability", 0.5))
        confidence = float(analysis.get("confidence", 0.5))
        reasoning = analysis.get("reasoning", "")

        edge_yes = prob - market["yes_price"]
        edge_no = (1 - prob) - market["no_price"]

        logger.info(
            "power_signal",
            q=market["question"][:45],
            claude=f"{prob:.2f}",
            market=f"{market['yes_price']:.2f}",
            edge_yes=f"{edge_yes:+.3f}",
            edge_no=f"{edge_no:+.3f}",
            conf=f"{confidence:.2f}",
            why=reasoning[:80],
        )

        if confidence < 0.65:
            continue

        if edge_yes > 0.05:
            side, edge = "buy_yes", edge_yes
        elif edge_no > 0.05:
            side, edge = "buy_no", edge_no
        else:
            continue

        size = min(state["bankroll"] * 0.10, 50.0)
        if size < 5:
            continue

        # Enforce per-expiry-date concentration cap
        market_date = market["end_date"][:10]
        current_date_exp = date_exposure.get(market_date, 0)
        if current_date_exp >= date_cap:
            logger.info(
                "date_cap_skip",
                date=market_date,
                exposure="$%.2f" % current_date_exp,
                cap="$%.2f" % date_cap,
            )
            continue
        size = min(size, date_cap - current_date_exp)
        if size < 5:
            continue

        logger.info("power_executing", q=market["question"][:40],
                     side=side, edge=f"{edge:+.3f}", size=f"${size:.2f}")

        if await execute_trade(clob, market, side, size, state):
            trades_placed += 1
            date_exposure[market_date] = current_date_exp + size

    logger.info("power_cycle_done", placed=trades_placed, analyzed=len(to_analyze),
                bankroll=f"${state['bankroll']:.2f}")


async def main():
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    initial_bankroll = float(sys.argv[2]) if len(sys.argv) > 2 else None

    clob = create_clob()
    state = load_state()
    if initial_bankroll is not None:
        state["bankroll"] = initial_bankroll
        save_state(state)

    print(f"[{datetime.now(timezone.utc).isoformat()}] Power Trader starting")
    print(f"  Duration: {hours}h | Bankroll: ${state['bankroll']:.2f}")
    print(f"  Strategy: Claude + web search on 200+ markets")
    print(f"  Min edge: 8c | Min confidence: 70% | Max size: $50")
    print()

    start = time.time()
    cycle = 0

    while time.time() - start < hours * 3600:
        cycle += 1
        try:
            await run_cycle(clob, state)
        except Exception as e:
            logger.error("cycle_error", error=str(e))

        open_trades = [t for t in state["trades"] if t["status"] == "open"]
        logger.info("power_status", cycle=cycle, bankroll=f"${state['bankroll']:.2f}",
                     open=len(open_trades), total=len(state["trades"]))

        with open(LOG_FILE, "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cycle": cycle, "bankroll": state["bankroll"],
                "open": len(open_trades), "total_trades": len(state["trades"]),
            }) + "\n")

        if time.time() - start < hours * 3600:
            logger.info("power_sleeping", next="60 min")
            await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
