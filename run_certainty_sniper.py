#!/usr/bin/env python3
"""
Certainty Sniper — Late-Period Triple-Confirmed Compounder

Strategy: In minutes 12-14 of each 15-min period, when a coin has moved
>0.7% AND 2+ coins confirm the direction AND Binance order book still shows
pressure, buy the winning-side token at current market price (~$0.82-0.90).
Token resolves $1 in <3 minutes. Half-Kelly compounding.

Gates:
  1. Target coin moved >=0.7% from period open
  2. >=2 coins moved >=0.5% in same direction (macro signal)
  3. Binance top-10 order book: bid depth > ask depth (UP) or vice versa
"""

import asyncio
import functools
import json
import os
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

import httpx
import structlog

try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
    from py_clob_client.order_builder.constants import BUY
except ImportError:
    ClobClient = Any
    ApiCreds = None
    OrderArgs = None
    OrderType = None
    BUY = None
from src.crypto_arb.ws_feeds import BinanceWSFeed

logger = structlog.get_logger()

LOG_FILE   = Path(__file__).parent / "certainty_sniper_output.log"
STATE_FILE = Path(__file__).parent / "data" / "certainty_sniper_state.json"
FILL_LOG_FILE = Path(__file__).parent / "data" / "certainty_fill_journal.jsonl"

COINS         = ["btc", "eth", "sol", "xrp", "doge", "bnb"]
COIN_TO_ASSET = {"btc": "BTC", "eth": "ETH", "sol": "SOL",
                 "xrp": "XRP", "doge": "DOGE", "bnb": "BNB"}
CERTAINTY_ACTIVE_COINS = ("btc", "eth", "sol", "bnb")
BOOK_MIN_BY_COIN = {
    "btc": 20.0,
    "eth": 16.0,
    "sol": 14.0,
    "xrp": 10.0,
    "doge": 10.0,
    "bnb": 10.0,
}

# Timing
CERTAINTY_START  = 630   # 10.5 min into period — earlier entries are cheaper and still highly informative
CERTAINTY_END    = 810   # 13.5 min into period — stop before the market becomes near-resolved and untradeable
CHECK_INTERVAL   = 10    # Late-period books can change quickly; poll often enough to catch temporary liquidity
MAX_OPEN_TRADES  = 2     # Avoid over-concentrating when multiple coins fire together

# Signal thresholds
MIN_MOVE_PCT       = 0.0055  # Base move gate; effective threshold tightens/loosens with elapsed time
CONFIRM_MOVE_PCT   = 0.0035  # Base confirm threshold; breadth is scored, not just hard-filtered
MIN_CONFIRMING     = 2       # Includes the target coin
MAX_ENTRY_PRICE    = 0.88   # Hard cap; above this the payout buffer is too thin to justify the trade
MIN_SIGNAL_SCORE   = 1.75   # Minimum composite score to trade

# Sizing
BET_MIN              = 8.0
BET_MAX              = 40.0
MAX_BET_PCT_BANKROLL = 0.15   # Hard ceiling — guards against over-optimistic win_rate
BOOK_TAKE_PCT        = 0.35   # Only lean on a fraction of visible asks to reduce FAK misses
MAX_TRADES_PER_COIN_PER_DAY = 8

# Risk controls
CIRCUIT_LOSSES    = 2     # Consecutive losses before pause
CIRCUIT_SKIP      = 4     # Periods to skip after circuit triggers
DAILY_LOSS_PCT    = 0.12  # 12% daily loss → halt for 24h
MIN_BANKROLL      = 30.0  # Stop trading below this
PRICE_HISTORY_SEC = 1800  # Keep enough Binance ticks to resolve prior periods accurately


def kelly_size(bankroll: float, win_rate: float = 0.90, avg_entry: float = 0.85) -> float:
    """Half-Kelly with hard 15% bankroll cap. Protects against mis-estimated win_rate."""
    b = (1.0 - avg_entry) / avg_entry
    kelly = (win_rate * b - (1.0 - win_rate)) / b
    size = bankroll * (kelly / 2.0)
    pct_cap = bankroll * MAX_BET_PCT_BANKROLL
    size = min(size, pct_cap)
    return min(BET_MAX, max(BET_MIN, round(size, 2)))


def gate1_move(cur: float, start: float) -> tuple[bool, float]:
    """Gate 1: coin moved >=MIN_MOVE_PCT from period open. Returns (passed, move_fraction)."""
    if not cur or not start:
        return False, 0.0
    move = (cur - start) / start
    return abs(move) >= MIN_MOVE_PCT, move


def gate2_confirm(all_moves: dict[str, float], target_direction: str) -> bool:
    """Gate 2: >=MIN_CONFIRMING coins moved >=CONFIRM_MOVE_PCT in same direction."""
    # all_moves includes the target coin (which passed Gate 1 >= MIN_MOVE_PCT > CONFIRM_MOVE_PCT),
    # so effective requirement is >= (MIN_CONFIRMING - 1) OTHER coins confirming.
    if target_direction == "up":
        confirming = sum(1 for m in all_moves.values() if m >= CONFIRM_MOVE_PCT)
    else:
        confirming = sum(1 for m in all_moves.values() if m <= -CONFIRM_MOVE_PCT)
    return confirming >= MIN_CONFIRMING


def gate2_confirm_count(all_moves: dict[str, float], target_direction: str) -> int:
    """Return the number of confirming coins, including the target coin."""
    if target_direction == "up":
        return sum(1 for m in all_moves.values() if m >= CONFIRM_MOVE_PCT)
    return sum(1 for m in all_moves.values() if m <= -CONFIRM_MOVE_PCT)


def compute_all_moves(ws_feed: BinanceWSFeed, period_prices: dict[str, float]) -> dict[str, float]:
    """Compute move fraction for each coin from period open. Returns {coin: move_fraction}."""
    moves = {}
    for coin, asset in COIN_TO_ASSET.items():
        cur   = ws_feed.get(asset)
        start = period_prices.get(asset)
        if cur and start:
            moves[coin] = (cur - start) / start
    return moves


def price_at_or_before(history: deque, target_ts: float) -> float | None:
    """Return the most recent observed price at or before the target timestamp."""
    best = None
    for ts, price in history:
        if ts <= target_ts:
            best = price
        else:
            break
    return best


def certainty_move_threshold(elapsed: int) -> float:
    """Relax the move threshold later in the period when certainty is cheaper to infer."""
    progress = min(max((elapsed - CERTAINTY_START) / max(CERTAINTY_END - CERTAINTY_START, 1), 0.0), 1.0)
    return MIN_MOVE_PCT - 0.0015 * progress


def certainty_confirm_threshold(elapsed: int) -> float:
    """Relax breadth slightly later in the period to keep the strategy active in quieter tapes."""
    progress = min(max((elapsed - CERTAINTY_START) / max(CERTAINTY_END - CERTAINTY_START, 1), 0.0), 1.0)
    return CONFIRM_MOVE_PCT - 0.0010 * progress


def orderbook_imbalance(bid_vol: float, ask_vol: float, direction: str) -> float:
    """Signed [0, 1] pressure score for the target direction."""
    total = bid_vol + ask_vol
    if total <= 0:
        return 0.0
    raw = (bid_vol - ask_vol) / total
    return raw if direction == "up" else -raw


def certainty_signal_score(
    move_abs: float,
    move_threshold: float,
    confirm_count: int,
    imbalance: float,
    entry_price: float,
    ask_usd_vol: float,
    elapsed: int,
) -> float:
    """Composite score balancing directional certainty, breadth, execution, and timeliness."""
    if move_threshold <= 0 or entry_price <= 0:
        return 0.0
    move_score = move_abs / move_threshold
    breadth_score = max(0.0, confirm_count - 1) * 0.35
    imbalance_score = max(0.0, imbalance) * 2.0
    price_score = max(0.0, (MAX_ENTRY_PRICE - entry_price) / 0.10)
    liquidity_score = min(ask_usd_vol / 40.0, 1.0) * 0.35
    time_bonus = min(max((elapsed - CERTAINTY_START) / 120.0, 0.0), 1.0) * 0.15
    return move_score + breadth_score + imbalance_score + price_score + liquidity_score + time_bonus


def create_clob() -> ClobClient:
    if ApiCreds is None:
        raise RuntimeError("py_clob_client is not installed")
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
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            logger.warning("state_file_corrupt", path=str(STATE_FILE))
    return {
        "bankroll": 0.0,
        "trades": [],
        "total_invested": 0.0,
        "total_returned": 0.0,
        "consecutive_losses": 0,
        "skip_signals": 0,
        "daily_start_bankroll": 0.0,
        "daily_start_date": "",
    }


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=2))


def append_fill_journal(entry: dict) -> None:
    FILL_LOG_FILE.parent.mkdir(exist_ok=True)
    with FILL_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")


def coin_book_min_usd(coin: str) -> float:
    return BOOK_MIN_BY_COIN.get(coin, min(BOOK_MIN_BY_COIN.values()))


def count_coin_trades_today(trades: list[dict], coin: str, today: str) -> int:
    count = 0
    for trade in trades:
        if trade.get("coin") != coin:
            continue
        placed_at = trade.get("placed_at")
        if not placed_at:
            continue
        try:
            trade_day = datetime.fromtimestamp(float(placed_at), tz=timezone.utc).date().isoformat()
        except (TypeError, ValueError, OSError):
            continue
        if trade_day == today:
            count += 1
    return count


def certainty_coin_is_tradeable(coin: str) -> bool:
    return coin in CERTAINTY_ACTIVE_COINS


def target_size_usd(bankroll: float, ask_usd_vol: float, coin: str) -> float:
    """Size by bankroll, then clip to a fraction of visible ask liquidity."""
    base_size = min(kelly_size(bankroll), bankroll - 5)
    book_limited = ask_usd_vol * BOOK_TAKE_PCT
    coin_min = coin_book_min_usd(coin)
    if ask_usd_vol < coin_min:
        return 0.0
    if book_limited < BET_MIN:
        return 0.0
    return min(base_size, BET_MAX, round(book_limited, 2))


def _coerce_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_fill_metrics(result: dict, requested_tokens: float, limit_price: float) -> tuple[float, float]:
    """Best-effort parse of filled tokens / average price from exchange response."""
    token_candidates = (
        result.get("filledSize"),
        result.get("filled_size"),
        result.get("sizeMatched"),
        result.get("size_matched"),
        result.get("matchedSize"),
        result.get("matched_size"),
        result.get("takerFillSize"),
        result.get("makerFillSize"),
    )
    filled_tokens = next((v for v in (_coerce_float(x) for x in token_candidates) if v and v > 0), None)
    if filled_tokens is None:
        status = str(result.get("status", "")).lower()
        if status in {"filled", "matched"}:
            filled_tokens = requested_tokens
        else:
            filled_tokens = 0.0

    price_candidates = (
        result.get("avgPrice"),
        result.get("avg_price"),
        result.get("price"),
        result.get("matchedPrice"),
        result.get("matched_price"),
    )
    avg_price = next((v for v in (_coerce_float(x) for x in price_candidates) if v and v > 0), None)
    if avg_price is None:
        spent_candidates = (
            result.get("filledValue"),
            result.get("filled_value"),
            result.get("takingAmount"),
            result.get("taking_amount"),
            result.get("matchedAmount"),
            result.get("matched_amount"),
        )
        spent = next((v for v in (_coerce_float(x) for x in spent_candidates) if v and v > 0), None)
        if spent is not None and filled_tokens > 0:
            avg_price = spent / filled_tokens

    if avg_price is None or avg_price <= 0:
        avg_price = limit_price
    return round(filled_tokens, 4), round(avg_price, 4)


async def gate3_orderbook(client: httpx.AsyncClient, coin: str) -> tuple[float, float]:
    """Return Binance top-10 bid/ask depth totals for pressure scoring."""
    symbol = COIN_TO_ASSET[coin] + "USDT"
    try:
        resp = await client.get(
            f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit=10",
            timeout=3,
        )
        resp.raise_for_status()
        data = resp.json()
        bid_vol = sum(float(qty) for _, qty in data.get("bids", []))
        ask_vol = sum(float(qty) for _, qty in data.get("asks", []))
        return bid_vol, ask_vol
    except Exception as e:
        logger.debug("orderbook_error", coin=coin, error=str(e))
        return 0.0, 0.0


async def get_clob_ask(client: httpx.AsyncClient, token_id: str) -> tuple[float, float]:
    """Fetch best ask price and USD volume resting at that price from CLOB.
    Returns (best_ask, usd_volume). Returns (0.0, 0.0) on failure."""
    try:
        resp = await client.get(
            f"https://clob.polymarket.com/book?token_id={token_id}",
            timeout=3,
        )
        data = resp.json()
        asks = sorted(data.get("asks", []), key=lambda x: float(x["price"]))
        if not asks:
            return 0.0, 0.0
        best_price = float(asks[0]["price"])
        vol_tokens = sum(float(a["size"]) for a in asks
                         if float(a["price"]) <= best_price + 0.01)
        return best_price, vol_tokens * best_price
    except Exception:
        return 0.0, 0.0


async def fetch_market_price(client: httpx.AsyncClient, coin: str, period_ts: int) -> dict | None:
    """Fetch current market prices for a coin's period — called fresh at minute 12."""
    slug = f"{coin}-updown-15m-{period_ts}"
    try:
        resp = await client.get(
            f"https://gamma-api.polymarket.com/markets?slug={slug}",
            timeout=8,
        )
        data = resp.json()
        if not data:
            return None
        m = data[0]
        tokens   = json.loads(m["clobTokenIds"])  if isinstance(m.get("clobTokenIds"), str) else m.get("clobTokenIds", [])
        outcomes = json.loads(m["outcomes"])       if isinstance(m.get("outcomes"), str)      else m.get("outcomes", [])
        prices   = json.loads(m["outcomePrices"])  if isinstance(m.get("outcomePrices"), str) else m.get("outcomePrices", [])
        if len(tokens) < 2 or len(outcomes) < 2:
            return None
        up_idx   = next((i for i, o in enumerate(outcomes) if o.lower() == "up"),   0)
        down_idx = next((i for i, o in enumerate(outcomes) if o.lower() == "down"), 1)
        return {
            "coin":          coin,
            "condition_id":  m.get("conditionId", ""),
            "up_token_id":   str(tokens[up_idx]),
            "down_token_id": str(tokens[down_idx]),
            "up_price":      float(prices[up_idx]),
            "down_price":    float(prices[down_idx]),
        }
    except Exception as e:
        logger.debug("fetch_market_price_error", coin=coin, error=str(e))
        return None


async def place_order(loop, clob: ClobClient, token_id: str,
                      entry_price: float, tokens: float) -> dict:
    """Place a FAK buy order and return the raw exchange response plus parsed fill metrics."""
    if OrderArgs is None or OrderType is None or BUY is None:
        raise RuntimeError("py_clob_client is not installed")
    try:
        order_args = OrderArgs(
            token_id=token_id,
            price=round(entry_price, 4),
            size=round(tokens, 2),
            side=BUY,
        )
        signed = await loop.run_in_executor(None, clob.create_order, order_args)
        result = await loop.run_in_executor(
            None, functools.partial(clob.post_order, signed, OrderType.FAK))
        order_id = result.get("orderID", result.get("id", ""))
        filled_tokens, avg_price = extract_fill_metrics(result, tokens, entry_price)
        return {
            "order_id": order_id,
            "filled_tokens": filled_tokens,
            "avg_price": avg_price,
            "raw": result,
        }
    except Exception as e:
        logger.error("place_order_error", error=str(e)[:120])
        return {
            "order_id": "",
            "filled_tokens": 0.0,
            "avg_price": entry_price,
            "raw": {"error": str(e)[:240]},
        }


async def main():
    duration      = int(sys.argv[1])   if len(sys.argv) > 1 else 604800  # 7 days
    bankroll_init = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    clob = create_clob()
    http = httpx.AsyncClient(timeout=10)
    loop = asyncio.get_running_loop()

    state = load_state()
    if bankroll_init > 0 and state.get("bankroll", 0) <= 0 and not state.get("trades"):
        state["bankroll"] = bankroll_init
        state["daily_start_bankroll"] = bankroll_init
        state["daily_start_date"] = datetime.now(timezone.utc).date().isoformat()
        save_state(state)

    bankroll = state.get("bankroll", 0.0)
    if bankroll <= 0 and bankroll_init > 0:
        bankroll = bankroll_init
        state["bankroll"] = bankroll
        state["daily_start_bankroll"] = bankroll_init
        state["daily_start_date"] = datetime.now(timezone.utc).date().isoformat()
        save_state(state)
    consecutive_losses = state.get("consecutive_losses", 0)
    skip_signals = state.get("skip_signals", 0)

    # Start Binance WebSocket
    ws_feed = BinanceWSFeed()
    ws_task = asyncio.create_task(ws_feed.run())

    for _ in range(100):
        if ws_feed.get("BTC"):
            break
        await asyncio.sleep(0.1)

    if not ws_feed.get("BTC"):
        logger.error("no_btc_price_timeout")
        return

    logger.info("certainty_sniper_ready",
                btc=ws_feed.get("BTC"), bankroll=round(bankroll, 2))

    current_period     = 0
    hist_start: dict[int, dict[str, float]] = {}
    open_trades: list[dict] = [t for t in state.get("trades", [])
                                if t.get("status") in ("placed", "pending_settlement")]
    rolling_prices: dict[str, deque] = {asset: deque() for asset in COIN_TO_ASSET.values()}
    fired_this_period: set[str] = set()
    last_skip_period = 0
    last_check_ts = 0
    last_status   = 0
    start_time    = time.time()

    try:
        while time.time() - start_time < duration:
            now_ts    = int(time.time())
            period_ts = (now_ts // 900) * 900
            elapsed   = now_ts - period_ts
            now_f     = time.time()

            # Maintain a rolling Binance price tape so settlement uses the price at cutoff.
            for asset in COIN_TO_ASSET.values():
                px = ws_feed.get(asset)
                if not px:
                    continue
                history = rolling_prices[asset]
                history.append((now_f, px))
                cutoff_ts = now_f - PRICE_HISTORY_SEC
                while history and history[0][0] < cutoff_ts:
                    history.popleft()

            # ── NEW PERIOD ────────────────────────────────────
            if period_ts != current_period:
                fired_this_period.clear()
                last_check_ts = 0

                if skip_signals > 0 and period_ts != last_skip_period:
                    skip_signals -= 1
                    state["skip_signals"] = skip_signals
                    last_skip_period = period_ts
                    logger.info("CIRCUIT_BREAKER_TICK", skip_remaining=skip_signals)

                # Record period-start prices
                hist_start[period_ts] = {
                    asset: ws_feed.get(asset)
                    for asset in COIN_TO_ASSET.values()
                    if ws_feed.get(asset)
                }
                # Prune old history
                cutoff = period_ts - 4 * 900
                for old_ts in [k for k in list(hist_start) if k < cutoff]:
                    del hist_start[old_ts]

                current_period = period_ts

                # Reset daily loss tracking at start of new UTC day
                today = datetime.now(timezone.utc).date().isoformat()
                if state.get("daily_start_date") != today:
                    state["daily_start_bankroll"] = bankroll
                    state["daily_start_date"] = today
                    save_state(state)

            # ── STATUS HEARTBEAT (every 60s) ──────────────────
            if now_ts - last_status >= 60:
                wins   = sum(1 for t in state["trades"] if t.get("status") == "won")
                losses = sum(1 for t in state["trades"] if t.get("status") == "lost")
                total  = wins + losses
                wr     = ("%.0f%%" % (wins / total * 100)) if total else "--"
                pnl    = sum(t.get("pnl", 0) for t in state["trades"])
                logger.info("certainty_status",
                            bankroll=round(bankroll, 2),
                            btc=ws_feed.get("BTC"),
                            open=sum(1 for t in open_trades if t["status"] == "placed"),
                            trades="%dW/%dL" % (wins, losses),
                            win_rate=wr,
                            pnl=round(pnl, 2),
                            period_elapsed="%ds" % elapsed,
                            circuit="skip%d" % skip_signals if skip_signals else "ok")
                last_status = now_ts

            # ── RESOLVE TRADES (period_end + 120s grace) ──────
            for trade in list(open_trades):
                if trade["status"] != "placed":
                    continue
                if now_ts <= trade["period_end"] + 120:
                    continue

                coin  = trade["coin"]
                asset = COIN_TO_ASSET[coin]
                t_per = trade["period"]

                start_p = hist_start.get(t_per, {}).get(asset)
                end_p   = price_at_or_before(rolling_prices.get(asset, deque()), trade["period_end"])

                if not start_p or end_p is None:
                    trade["status"] = "pending_settlement"
                    continue

                actual_move = ((end_p - start_p) / start_p) if end_p else 0
                won = (
                    (trade["side"] == "buy_up"   and actual_move > 0) or
                    (trade["side"] == "buy_down"  and actual_move < 0)
                )

                if won:
                    payout = trade["tokens"] * 0.98  # 2% Polymarket fee
                    trade["pnl"]    = round(payout - trade["size_usd"], 2)
                    trade["status"] = "won"
                    bankroll += payout
                    state["total_returned"] = state.get("total_returned", 0) + payout
                    consecutive_losses = 0
                else:
                    trade["pnl"]    = -trade["size_usd"]
                    trade["status"] = "lost"
                    consecutive_losses += 1
                    if consecutive_losses >= CIRCUIT_LOSSES:
                        skip_signals = CIRCUIT_SKIP
                        state["skip_signals"] = skip_signals
                        logger.warning("CIRCUIT_BREAKER",
                                       consecutive_losses=consecutive_losses,
                                       skipping_next=CIRCUIT_SKIP)
                        consecutive_losses = 0

                state["bankroll"]            = bankroll
                state["consecutive_losses"]  = consecutive_losses
                save_state(state)

                wins_total   = sum(1 for t in state["trades"] if t.get("status") == "won")
                losses_total = sum(1 for t in state["trades"] if t.get("status") == "lost")
                total_t      = wins_total + losses_total
                wr           = ("%.0f%%" % (wins_total / total_t * 100)) if total_t else "--"

                logger.info("CERTAINTY_RESOLVED",
                            coin=coin.upper(),
                            side=trade["side"],
                            won=won,
                            pnl=trade["pnl"],
                            bankroll=round(bankroll, 2),
                            win_rate=wr,
                            trades="%dW/%dL" % (wins_total, losses_total))

            # ── EXPIRE STUCK PENDING_SETTLEMENT TRADES ────────
            for trade in list(open_trades):
                if trade["status"] != "pending_settlement":
                    continue
                if now_ts <= trade["period_end"] + 300:
                    continue
                trade["status"] = "expired"
                trade["pnl"]    = 0
                bankroll += trade["size_usd"]
                state["bankroll"] = bankroll
                save_state(state)
                logger.warning("certainty_expired",
                               coin=trade["coin"].upper(),
                               size_usd=trade["size_usd"],
                               reason="no_start_price_after_300s")

            # ── DAILY LOSS LIMIT (only blocks new trades) ─────
            # Exclude deployed capital — a placed bet is not a loss yet
            open_deployed = sum(t["size_usd"] for t in open_trades if t["status"] == "placed")
            daily_start = state.get("daily_start_bankroll", bankroll)
            if daily_start > 0 and (daily_start - (bankroll + open_deployed)) / daily_start >= DAILY_LOSS_PCT:
                logger.warning("daily_loss_limit_hit",
                               start=round(daily_start, 2), now=round(bankroll, 2))
                await asyncio.sleep(300)
                continue

            # ── MIN BANKROLL ──────────────────────────────────
            if bankroll < MIN_BANKROLL:
                logger.warning("bankroll_too_low", bankroll=round(bankroll, 2))
                await asyncio.sleep(60)
                continue

            # ── CERTAINTY WINDOW: minutes 12–14 ──────────────
            in_window = CERTAINTY_START <= elapsed <= CERTAINTY_END
            check_due = (now_ts - last_check_ts) >= CHECK_INTERVAL

            if in_window and check_due and skip_signals == 0:
                last_check_ts = now_ts
                period_prices = hist_start.get(current_period, {})
                all_moves     = compute_all_moves(ws_feed, period_prices)
                active_positions = sum(1 for t in open_trades if t["status"] == "placed")
                today = datetime.now(timezone.utc).date().isoformat()
                move_gate = certainty_move_threshold(elapsed)
                confirm_gate = certainty_confirm_threshold(elapsed)
                candidates: list[dict] = []

                for coin in COINS:
                    if not certainty_coin_is_tradeable(coin):
                        continue
                    if coin in fired_this_period:
                        continue
                    if active_positions >= MAX_OPEN_TRADES:
                        break
                    if count_coin_trades_today(state["trades"], coin, today) >= MAX_TRADES_PER_COIN_PER_DAY:
                        logger.info("certainty_skip_daily_coin_cap", coin=coin, cap=MAX_TRADES_PER_COIN_PER_DAY)
                        fired_this_period.add(coin)
                        continue

                    move = all_moves.get(coin, 0.0)
                    passed_g1, move_fraction = gate1_move(
                        cur=ws_feed.get(COIN_TO_ASSET[coin]),
                        start=period_prices.get(COIN_TO_ASSET[coin]),
                    )
                    if not passed_g1 or abs(move_fraction) < move_gate:
                        continue

                    direction = "up" if move > 0 else "down"
                    adjusted_moves = {
                        c: m for c, m in all_moves.items()
                        if (m >= confirm_gate if direction == "up" else m <= -confirm_gate)
                    }
                    confirm_count = len(adjusted_moves)
                    if confirm_count < MIN_CONFIRMING:
                        continue

                    # Gate 3 — order book contributes to score rather than acting as a single hard cliff.
                    try:
                        bid_vol, ask_vol = await asyncio.wait_for(
                            gate3_orderbook(http, coin), timeout=3)
                    except asyncio.TimeoutError:
                        bid_vol, ask_vol = 0.0, 0.0
                    imbalance = orderbook_imbalance(bid_vol, ask_vol, direction)
                    if imbalance <= -0.10:
                        logger.debug("certainty_skip_gate3", coin=coin, move="%.3f%%" % (move * 100))
                        continue

                    market = await fetch_market_price(http, coin, current_period)
                    if not market:
                        logger.warning("certainty_no_market", coin=coin)
                        continue

                    side      = "buy_up" if direction == "up" else "buy_down"
                    token_id  = market["up_token_id"] if direction == "up" else market["down_token_id"]
                    clob_ask, ask_usd_vol = await get_clob_ask(http, token_id)
                    min_book_usd = coin_book_min_usd(coin)
                    if clob_ask <= 0 or ask_usd_vol < min_book_usd:
                        logger.info("certainty_skip_thin_book",
                                    coin=coin, ask_vol_usd=round(ask_usd_vol, 1),
                                    min_book_usd=min_book_usd)
                        continue

                    entry_price = round(clob_ask, 4)

                    if entry_price > MAX_ENTRY_PRICE:
                        logger.info("certainty_skip",
                                    reason="entry_too_high",
                                    coin=coin, price=entry_price)
                        continue

                    size_usd = target_size_usd(bankroll, ask_usd_vol, coin)
                    if size_usd < BET_MIN:
                        logger.info("certainty_skip_small_size",
                                    coin=coin,
                                    ask_vol_usd=round(ask_usd_vol, 1),
                                    target_size=round(size_usd, 2))
                        continue

                    # Cap to a fraction of visible depth; FAK should not assume we own the whole level.
                    available_tokens = (ask_usd_vol * BOOK_TAKE_PCT) / entry_price
                    tokens = min(size_usd / entry_price, available_tokens)
                    actual_size = round(tokens * entry_price, 2)
                    if actual_size < BET_MIN:
                        logger.info("certainty_skip_fractional_depth",
                                    coin=coin,
                                    ask_vol_usd=round(ask_usd_vol, 1),
                                    take_pct=BOOK_TAKE_PCT)
                        continue

                    score = certainty_signal_score(
                        move_abs=abs(move_fraction),
                        move_threshold=move_gate,
                        confirm_count=confirm_count,
                        imbalance=imbalance,
                        entry_price=entry_price,
                        ask_usd_vol=ask_usd_vol,
                        elapsed=elapsed,
                    )
                    if score < MIN_SIGNAL_SCORE:
                        logger.debug(
                            "certainty_skip_low_score",
                            coin=coin,
                            score=round(score, 3),
                            move=round(move_fraction * 100, 3),
                            confirm_count=confirm_count,
                            imbalance=round(imbalance, 3),
                            entry_price=entry_price,
                        )
                        continue

                    candidates.append({
                        "coin": coin,
                        "direction": direction,
                        "side": side,
                        "move": move,
                        "entry_price": entry_price,
                        "size_usd": size_usd,
                        "ask_usd_vol": ask_usd_vol,
                        "tokens": tokens,
                        "actual_size": actual_size,
                        "market": market,
                        "token_id": token_id,
                        "confirm_count": confirm_count,
                        "score": score,
                        "imbalance": imbalance,
                    })

                candidates.sort(key=lambda c: (c["score"], c["ask_usd_vol"], -c["entry_price"]), reverse=True)

                for candidate in candidates[: max(MAX_OPEN_TRADES - active_positions, 0)]:
                    coin = candidate["coin"]
                    side = candidate["side"]
                    direction = candidate["direction"]
                    move = candidate["move"]
                    entry_price = candidate["entry_price"]
                    size_usd = candidate["size_usd"]
                    ask_usd_vol = candidate["ask_usd_vol"]
                    tokens = candidate["tokens"]
                    actual_size = candidate["actual_size"]
                    market = candidate["market"]
                    token_id = candidate["token_id"]
                    confirm_count = candidate["confirm_count"]
                    score = candidate["score"]

                    logger.info("certainty_signal",
                                coin=coin,
                                direction=direction,
                                move="%.3f%%" % (move * 100),
                                entry_price=entry_price,
                                size_usd=actual_size,
                                ask_vol=round(ask_usd_vol, 1),
                                confirm_count=confirm_count,
                                score=round(score, 3),
                                elapsed=elapsed)

                    order = await place_order(loop, clob, token_id, entry_price, tokens)
                    filled_tokens = order["filled_tokens"]
                    fill_price = order["avg_price"]
                    actual_size = round(filled_tokens * fill_price, 2)
                    order_id = order["order_id"]

                    append_fill_journal({
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "coin": coin,
                        "side": side,
                        "period": current_period,
                        "token_id": token_id,
                        "requested_tokens": round(tokens, 4),
                        "requested_price": entry_price,
                        "requested_notional": round(tokens * entry_price, 2),
                        "filled_tokens": filled_tokens,
                        "fill_price": fill_price,
                        "filled_notional": actual_size,
                        "ask_usd_vol": round(ask_usd_vol, 2),
                        "raw": order["raw"],
                    })

                    if not order_id or filled_tokens <= 0 or actual_size < 1.0:
                        logger.info("certainty_unfilled",
                                    coin=coin,
                                    requested_tokens=round(tokens, 2),
                                    order_id=(order_id[:16] if order_id else ""),
                                    filled_tokens=filled_tokens)
                        fired_this_period.add(coin)
                        continue

                    bankroll -= actual_size
                    state["bankroll"]        = bankroll
                    state["total_invested"]  = state.get("total_invested", 0) + actual_size

                    trade = {
                        "period":       current_period,
                        "coin":         coin,
                        "side":         side,
                        "token_id":     token_id,
                        "condition_id": market["condition_id"],
                        "entry_price":  fill_price,
                        "size_usd":     actual_size,
                        "requested_entry_price": entry_price,
                        "requested_tokens": round(tokens, 4),
                        "tokens":       filled_tokens,
                        "order_id":     order_id,
                        "placed_at":    time.time(),
                        "period_end":   current_period + 900,
                        "move_pct":     round(move * 100, 3),
                        "confirm_count": confirm_count,
                        "signal_score": round(score, 3),
                        "btc_at_entry": ws_feed.get("BTC"),
                        "status":       "placed",
                        "pnl":          0,
                    }
                    open_trades.append(trade)
                    state["trades"].append(trade)
                    fired_this_period.add(coin)
                    active_positions += 1
                    save_state(state)

                    logger.info("CERTAINTY_TRADE",
                                coin=coin.upper(), side=side,
                                move="%.3f%%" % (move * 100),
                                requested_price=entry_price,
                                fill_price=fill_price,
                                size_usd=actual_size,
                                tokens=filled_tokens,
                                elapsed=elapsed,
                                bankroll=round(bankroll, 2),
                                order_id=order_id[:16])

            open_trades = [t for t in open_trades
                           if t["status"] not in ("won", "lost", "cancelled", "expired")]

            await asyncio.sleep(1)

    finally:
        ws_task.cancel()
        await http.aclose()
        save_state(state)
        logger.info("certainty_sniper_stopped", bankroll=round(bankroll, 2))


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        handlers=[logging.StreamHandler()],
        format="%(message)s",
        level=logging.INFO,
    )
    logging.getLogger("websockets").setLevel(logging.WARNING)
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
    )
    asyncio.run(main())
