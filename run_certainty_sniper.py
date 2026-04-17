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
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / '.env')

import httpx
import structlog

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import ApiCreds, OrderArgs, OrderType
from py_clob_client.order_builder.constants import BUY
from src.crypto_arb.ws_feeds import BinanceWSFeed

logger = structlog.get_logger()

LOG_FILE   = Path(__file__).parent / "certainty_sniper_output.log"
STATE_FILE = Path(__file__).parent / "data" / "certainty_sniper_state.json"

COINS         = ["btc", "eth", "sol", "xrp", "doge", "bnb"]
COIN_TO_ASSET = {"btc": "BTC", "eth": "ETH", "sol": "SOL",
                 "xrp": "XRP", "doge": "DOGE", "bnb": "BNB"}
# Only trade coins with enough CLOB liquidity to support late-period entries
LIQUID_COINS  = frozenset(["btc"])

# Timing
CERTAINTY_START  = 720   # 12 min into period — start checking
CERTAINTY_END    = 780   # 13 min into period — stop (2 min remaining, avoid liquidity desert)
CHECK_INTERVAL   = 30    # Check every 30s in window
CANCEL_AFTER     = 5     # Cancel unfilled order quickly — need fill before period ends
MIN_BOOK_VOLUME  = 20.0  # Min USD resting at best ask on CLOB before placing

# Signal thresholds
MIN_MOVE_PCT       = 0.007  # Gate 1: target coin moved >=0.7%
CONFIRM_MOVE_PCT   = 0.005  # Gate 2: confirming coins moved >=0.5%
MIN_CONFIRMING     = 2      # Gate 2: need >=2 confirming coins
MAX_ENTRY_PRICE    = 0.92   # Skip if market already priced certainty in

# Sizing
BET_MIN              = 8.0
BET_MAX              = 40.0
MAX_BET_PCT_BANKROLL = 0.15   # Hard ceiling — guards against over-optimistic win_rate

# Risk controls
CIRCUIT_LOSSES    = 2     # Consecutive losses before pause
CIRCUIT_SKIP      = 4     # Periods to skip after circuit triggers
DAILY_LOSS_PCT    = 0.12  # 12% daily loss → halt for 24h
MIN_BANKROLL      = 30.0  # Stop trading below this


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


def compute_all_moves(ws_feed: BinanceWSFeed, period_prices: dict[str, float]) -> dict[str, float]:
    """Compute move fraction for each coin from period open. Returns {coin: move_fraction}."""
    moves = {}
    for coin, asset in COIN_TO_ASSET.items():
        cur   = ws_feed.get(asset)
        start = period_prices.get(asset)
        if cur and start:
            moves[coin] = (cur - start) / start
    return moves


def create_clob() -> ClobClient:
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


async def gate3_orderbook(client: httpx.AsyncClient, coin: str, direction: str) -> bool:
    """Gate 3: Binance top-10 order book confirms continued pressure in direction.
    UP move: bid volume > ask volume. DOWN move: ask volume > bid volume."""
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
        if direction == "up":
            return bid_vol > ask_vol
        else:
            return ask_vol > bid_vol
    except Exception as e:
        logger.debug("orderbook_error", coin=coin, error=str(e))
        return False  # fail safe: skip if can't check


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
                      entry_price: float, tokens: float) -> str:
    """Place a FAK buy order. Fills immediately or auto-cancels. Returns order_id or ''."""
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
        return result.get("orderID", "")
    except Exception as e:
        logger.error("place_order_error", error=str(e)[:120])
        return ""


async def cancel_order(loop, clob: ClobClient, order_id: str) -> bool:
    try:
        await loop.run_in_executor(None, functools.partial(clob.cancel, order_id))
        return True
    except Exception as e:
        logger.debug("cancel_error", order_id=order_id[:16], error=str(e))
        return False


async def main():
    duration      = int(sys.argv[1])   if len(sys.argv) > 1 else 604800  # 7 days
    bankroll_init = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

    clob = create_clob()
    http = httpx.AsyncClient(timeout=10)
    loop = asyncio.get_running_loop()

    state = load_state()
    if bankroll_init > 0:
        state["bankroll"] = bankroll_init
        state["daily_start_bankroll"] = bankroll_init
        state["daily_start_date"] = datetime.now(timezone.utc).date().isoformat()
        save_state(state)

    bankroll = state["bankroll"]
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

            # ── CANCEL UNFILLED ORDERS ────────────────────────
            now_f = time.time()
            for trade in list(open_trades):
                if trade["status"] != "placed":
                    continue
                if now_f < trade.get("cancel_at", float("inf")):
                    continue
                cancelled = await cancel_order(loop, clob, trade["order_id"])
                if cancelled:
                    bankroll += trade["size_usd"]
                    state["bankroll"] = bankroll
                    trade["status"] = "cancelled"
                    trade["pnl"]    = 0
                    save_state(state)
                    logger.info("CERTAINTY_CANCELLED",
                                coin=trade["coin"].upper(),
                                order_id=trade["order_id"][:16])
                else:
                    trade["cancel_at"] = now_f + 15

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
                end_p   = ws_feed.get(asset)

                if not start_p:
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

                for coin in COINS:
                    if coin not in LIQUID_COINS:
                        continue
                    if coin in fired_this_period:
                        continue

                    move = all_moves.get(coin, 0.0)
                    passed_g1, _ = gate1_move(
                        cur=ws_feed.get(COIN_TO_ASSET[coin]),
                        start=period_prices.get(COIN_TO_ASSET[coin]),
                    )
                    if not passed_g1:
                        continue

                    direction = "up" if move > 0 else "down"
                    if not gate2_confirm(all_moves, direction):
                        continue

                    # Gate 3 — order book (async, with timeout guard)
                    try:
                        ob_ok = await asyncio.wait_for(
                            gate3_orderbook(http, coin, direction), timeout=3)
                    except asyncio.TimeoutError:
                        ob_ok = False
                    if not ob_ok:
                        logger.debug("certainty_skip_gate3", coin=coin, move="%.3f%%" % (move * 100))
                        continue

                    # All 3 gates passed — fetch fresh market price
                    market = await fetch_market_price(http, coin, current_period)
                    if not market:
                        logger.warning("certainty_no_market", coin=coin)
                        continue

                    side      = "buy_up" if direction == "up" else "buy_down"
                    token_id  = market["up_token_id"] if direction == "up" else market["down_token_id"]
                    raw_price = market["up_price"] if direction == "up" else market["down_price"]
                    entry_price = round(min(raw_price + 0.01, 0.99), 4)

                    # Gate 4: CLOB book must have enough liquidity to fill
                    clob_ask, ask_usd_vol = await get_clob_ask(http, token_id)
                    if ask_usd_vol < MIN_BOOK_VOLUME:
                        logger.info("certainty_skip_thin_book",
                                    coin=coin, ask_vol_usd=round(ask_usd_vol, 1))
                        fired_this_period.add(coin)
                        continue

                    if entry_price > MAX_ENTRY_PRICE:
                        logger.info("certainty_skip",
                                    reason="entry_too_high",
                                    coin=coin, price=entry_price)
                        fired_this_period.add(coin)
                        continue

                    size_usd = min(kelly_size(bankroll), bankroll - 5)
                    if size_usd < BET_MIN:
                        continue

                    # Cap tokens to available CLOB volume so FAK fills fully
                    available_tokens = ask_usd_vol / entry_price
                    tokens = min(size_usd / entry_price, available_tokens)
                    actual_size = round(tokens * entry_price, 2)

                    logger.info("certainty_signal",
                                coin=coin, direction=direction,
                                move="%.3f%%" % (move * 100),
                                entry_price=entry_price,
                                size_usd=actual_size,
                                ask_vol=round(ask_usd_vol, 1),
                                elapsed=elapsed)

                    order_id = await place_order(loop, clob, token_id, entry_price, tokens)
                    if not order_id:
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
                        "entry_price":  entry_price,
                        "size_usd":     actual_size,
                        "tokens":       round(tokens, 4),
                        "order_id":     order_id,
                        "placed_at":    time.time(),
                        "cancel_at":    time.time() + CANCEL_AFTER,
                        "period_end":   current_period + 900,
                        "move_pct":     round(move * 100, 3),
                        "btc_at_entry": ws_feed.get("BTC"),
                        "status":       "placed",
                        "pnl":          0,
                    }
                    open_trades.append(trade)
                    state["trades"].append(trade)
                    fired_this_period.add(coin)
                    save_state(state)

                    logger.info("CERTAINTY_TRADE",
                                coin=coin.upper(), side=side,
                                move="%.3f%%" % (move * 100),
                                entry_price=entry_price,
                                size_usd=round(size_usd, 2),
                                tokens=round(tokens, 2),
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
