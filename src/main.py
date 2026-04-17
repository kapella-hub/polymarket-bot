"""FastAPI application with lifespan-managed background services."""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import structlog
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.config import settings, ExecutionMode
from src.db.database import engine, async_session
from src.db.models import Base
from src.db.repositories import MarketRepository, PositionRepository, SignalRepository
from src.alpha.cross_market import CrossMarketAlpha
from src.alpha.llm_signal import LLMAlpha
from src.alpha.orderbook import OrderBookAlpha
from src.alpha.smart_money import SmartMoneyAlpha, SmartMoneyTracker
from src.ensemble.engine import EnsembleEngine
from src.exchange.polymarket import PolymarketAdapter
from src.execution.executor import ExecutionEngine
from src.llm.batch import BatchScheduler
from src.markets.discovery import GammaAPIClient
from src.markets.filters import MarketFilter
from src.nexus.client import NexusClient
from src.risk.controller import RiskController

logger = structlog.get_logger()

# Global state
_exchange: PolymarketAdapter | None = None
_gamma: GammaAPIClient | None = None
_filter: MarketFilter | None = None
_batch_scheduler: BatchScheduler | None = None
_cross_market: CrossMarketAlpha | None = None
_ensemble: EnsembleEngine | None = None
_executor: ExecutionEngine | None = None
_nexus: NexusClient | None = None
_shutdown_event = asyncio.Event()


async def _discovery_loop() -> None:
    """Background loop: discover and persist active markets."""
    while not _shutdown_event.is_set():
        try:
            markets = await _gamma.fetch_active_markets()
            filtered = _filter.apply(markets)

            async with async_session() as session:
                repo = MarketRepository(session)
                for m in filtered:
                    yes = m.outcome_yes
                    no = m.outcome_no
                    if not yes or not no:
                        continue
                    await repo.upsert({
                        "id": m.id,
                        "question": m.question,
                        "category": m.category,
                        "end_date": m.end_date,
                        "status": "active",
                        "clob_token_id_yes": yes.clob_token_id,
                        "clob_token_id_no": no.clob_token_id,
                        "outcome_yes": yes.name,
                        "outcome_no": no.name,
                        "volume": m.volume,
                        "liquidity": m.liquidity,
                        "best_bid": m.best_bid,
                        "best_ask": m.best_ask,
                        "last_price": m.yes_price,
                        "description": m.description,
                        "resolution_source": m.resolution_source,
                        "tags": m.tags,
                    })
                await session.commit()

            logger.info(
                "discovery_complete",
                total_fetched=len(markets),
                passed_filter=len(filtered),
            )
        except Exception as e:
            logger.error("discovery_error", error=str(e))

        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=settings.discovery_interval_seconds,
            )
            break
        except asyncio.TimeoutError:
            pass


async def _llm_batch_loop() -> None:
    """Background loop: run LLM batch evaluations on schedule."""
    # Wait one cycle before first batch to let discovery populate markets
    try:
        await asyncio.wait_for(
            _shutdown_event.wait(),
            timeout=settings.discovery_interval_seconds + 10,
        )
        return
    except asyncio.TimeoutError:
        pass

    while not _shutdown_event.is_set():
        try:
            count = await _batch_scheduler.run_batch()
            logger.info("llm_batch_cycle_done", evaluated=count)
        except Exception as e:
            logger.error("llm_batch_error", error=str(e))

        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=settings.llm_batch_interval_seconds,
            )
            break
        except asyncio.TimeoutError:
            pass


async def _strategy_loop() -> None:
    """Background loop: run ensemble engine on each active market."""
    # Wait for discovery + first LLM batch
    try:
        await asyncio.wait_for(
            _shutdown_event.wait(),
            timeout=settings.llm_batch_interval_seconds + 30,
        )
        return
    except asyncio.TimeoutError:
        pass

    while not _shutdown_event.is_set():
        try:
            async with async_session() as session:
                market_repo = MarketRepository(session)
                signal_repo = SignalRepository(session)

                markets = await market_repo.get_active()

                pos_repo = PositionRepository(session)
                positions = await pos_repo.get_all()

                # Update cross-market groups for arbitrage detection
                _cross_market.update_groups(markets)

                # Liquidity filter: only run ensemble on tradeable markets
                tradeable = [
                    m for m in markets
                    if m.best_bid is not None and m.best_bid > 0
                ]
                skipped = len(markets) - len(tradeable)
                if skipped:
                    logger.debug(
                        "strategy_skipped_illiquid", skipped=skipped,
                        tradeable=len(tradeable),
                    )

                decisions = []

                for market in tradeable:
                    signal = await signal_repo.get_latest(market.id)

                    # Build context for alphas
                    ctx = {"signal": signal}

                    # Optionally fetch order book
                    if _exchange and _exchange._connected:
                        try:
                            book = await _exchange.get_order_book(
                                market.clob_token_id_yes
                            )
                            ctx["order_book"] = book
                        except Exception:
                            pass

                    decision = await _ensemble.evaluate(market, ctx)
                    if decision:
                        decisions.append(decision)

                if decisions:
                    logger.info(
                        "strategy_cycle",
                        opportunities=len(decisions),
                        top_edge=f"{max(d.edge for d in decisions):+.3f}",
                    )
                    for decision in decisions:
                        await _executor.process(decision)

                # Update portfolio value for drawdown tracking
                if positions:
                    portfolio_val = sum(
                        abs(p.size * p.avg_entry_price) for p in positions
                    ) + settings.bankroll_usd
                    _executor._risk.update_portfolio_value(portfolio_val)

        except Exception as e:
            logger.error("strategy_loop_error", error=str(e))

        try:
            await asyncio.wait_for(
                _shutdown_event.wait(),
                timeout=settings.strategy_loop_interval_seconds,
            )
            break
        except asyncio.TimeoutError:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    global _exchange, _gamma, _filter, _batch_scheduler, _cross_market, _ensemble, _executor, _nexus, _shutdown_event
    _shutdown_event = asyncio.Event()  # Fresh event per startup — survives uvicorn --reload

    logger.info(
        "bot_starting",
        mode=settings.execution_mode.value,
        bot_name=settings.bot_name,
    )

    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Initialize components
    _gamma = GammaAPIClient()
    _blacklist = [c.strip() for c in settings.category_blacklist.split(",") if c.strip()]
    _filter = MarketFilter(category_blacklist=_blacklist)
    _exchange = PolymarketAdapter()

    if settings.polymarket_wallet_private_key:
        try:
            await _exchange.connect()
        except Exception as e:
            logger.warning("exchange_connect_failed", error=str(e))

    # Initialize NexusStack integration
    _nexus = NexusClient()

    # Initialize LLM batch scheduler, ensemble, and execution
    _batch_scheduler = BatchScheduler(nexus=_nexus)
    _smart_money = SmartMoneyTracker()
    _cross_market = CrossMarketAlpha()
    _ensemble = EnsembleEngine(alphas=[
        LLMAlpha(),
        OrderBookAlpha(),
        SmartMoneyAlpha(tracker=_smart_money),
        _cross_market,
    ])
    _executor = ExecutionEngine(
        exchange=_exchange,
        risk=RiskController(),
        nexus=_nexus,
    )

    # Start background tasks
    tasks = [
        asyncio.create_task(_discovery_loop()),
        asyncio.create_task(_llm_batch_loop()),
        asyncio.create_task(_strategy_loop()),
    ]

    logger.info("bot_started", mode=settings.execution_mode.value)
    await _nexus.event_bot_started(settings.execution_mode.value)

    yield

    # Shutdown
    logger.info("bot_shutting_down")
    _shutdown_event.set()
    for t in tasks:
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
    await _nexus.event_bot_stopped()
    if _gamma:
        await _gamma.close()
    await _nexus.close()
    await engine.dispose()
    logger.info("bot_stopped")


app = FastAPI(
    title="Polymarket Trading Bot",
    version="0.1.0",
    lifespan=lifespan,
)


# --- Health & Status ---


@app.get("/health")
async def health():
    kill_switch_active = os.path.exists(settings.kill_switch_file)
    return {
        "status": "halted" if kill_switch_active else "running",
        "mode": settings.execution_mode.value,
        "kill_switch": kill_switch_active,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/status")
async def status():
    async with async_session() as session:
        market_repo = MarketRepository(session)
        signal_repo = SignalRepository(session)
        position_repo = PositionRepository(session)

        markets = await market_repo.get_active()
        signals = await signal_repo.get_fresh_signals()
        positions = await position_repo.get_all()

    return {
        "mode": settings.execution_mode.value,
        "kill_switch": os.path.exists(settings.kill_switch_file),
        "active_markets": len(markets),
        "fresh_signals": len(signals),
        "open_positions": len(positions),
        "top_markets": [
            {
                "id": m.id,
                "question": m.question[:100],
                "volume": m.volume,
                "best_bid": m.best_bid,
                "best_ask": m.best_ask,
            }
            for m in markets[:10]
        ],
        "positions": [
            {
                "market_id": p.market_id,
                "outcome": p.outcome,
                "size": p.size,
                "avg_entry": p.avg_entry_price,
                "unrealized_pnl": p.unrealized_pnl,
            }
            for p in positions
        ],
    }


@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# --- Dashboard ---

@app.get("/dashboard")
async def dashboard():
    import pathlib
    html_path = pathlib.Path(__file__).parent.parent / "dashboard" / "index.html"
    if html_path.exists():
        return Response(content=html_path.read_text(), media_type="text/html")
    return Response(content="<h1>Dashboard not found</h1>", media_type="text/html")


@app.get("/api/crypto-arb")
async def crypto_arb_status():
    """Read the latest crypto arb paper trading state from the log file."""
    import json as jsonmod
    import pathlib
    # Priority: live v4 > WS engine > REST engine
    live_log = pathlib.Path(__file__).parent.parent / "live_arb_output.log"
    ws_log = pathlib.Path(__file__).parent.parent / "fast_arb_ws_output.log"
    rest_log = pathlib.Path(__file__).parent.parent / "crypto_arb_output.log"
    log_path = live_log if live_log.exists() and live_log.stat().st_size > 0 else (
        ws_log if ws_log.exists() and ws_log.stat().st_size > 0 else rest_log
    )
    result = {
        "bankroll": None, "total_pnl": "$0.00", "total_pnl_num": 0,
        "win_rate": "--", "roi": "--", "total_trades": 0,
        "open": 0, "won": 0, "lost": 0, "trades": [], "prices": {},
        "markets_scanned": 0, "signals": 0, "skip_repriced": 0,
        "ticks": 0, "period_remaining": "", "version": "legacy",
    }
    if not log_path.exists():
        return result
    try:
        lines = log_path.read_text().strip().split("\n")
        # Parse structlog key=value lines (handles single-quoted values with spaces)
        import re as _re

        def _parse_structlog(line: str) -> dict:
            parts = {}
            for m in _re.finditer(r"(\w+)='([^']*)'|(\w+)=(\S+)", line):
                if m.group(1):
                    parts[m.group(1)] = m.group(2)
                elif m.group(3):
                    parts[m.group(3)] = m.group(4)
            return parts

        trades = []
        latest_cycle = {}
        for line in lines:
            if "fast_trade" in line:
                parts = _parse_structlog(line)
                trades.append({
                    "question": parts.get("question", parts.get("coin", "")),
                    "side": parts.get("side", ""),
                    "entry_price": float(parts.get("entry", "0").replace("$", "")),
                    "size_usd": float(parts.get("size", "0").replace("$", "")),
                    "asset": parts.get("coin", "?"),
                    "binance_at_entry": 0,
                    "edge": parts.get("move", ""),
                    "confidence": parts.get("conf", ""),
                    "status": "open",
                    "pnl": 0,
                })
            elif "fast_resolved" in line:
                parts = _parse_structlog(line)
                coin = parts.get("coin", "")
                status = parts.get("status", "")
                pnl = float(parts.get("pnl", "0").replace("$", "").replace("+", ""))
                # Match to latest open trade for this coin
                for t in reversed(trades):
                    if t.get("asset") == coin and t["status"] == "open":
                        t["status"] = status
                        t["pnl"] = pnl
                        break
            elif "LIVE_FILL" in line:
                parts = _parse_structlog(line)
                trades.append({
                    "question": f"{parts.get('coin', '?')} 15-min {parts.get('side', '')}",
                    "side": parts.get("side", ""),
                    "entry_price": float(parts.get("clob_ask", "0").replace("$", "")),
                    "size_usd": float(parts.get("size", "0").replace("$", "")),
                    "asset": parts.get("coin", "?"),
                    "binance_at_entry": 0,
                    "edge": parts.get("move", ""),
                    "confidence": parts.get("conf", ""),
                    "status": "open",
                    "pnl": 0,
                })
            elif "LIVE_RESOLVED" in line:
                parts = _parse_structlog(line)
                coin = parts.get("coin", "")
                pnl_val = float(parts.get("pnl", "0").replace("$", "").replace("+", ""))
                status = "won" if pnl_val > 0 else "lost"
                for t in reversed(trades):
                    if t.get("asset") == coin and t["status"] == "open":
                        t["status"] = status
                        t["pnl"] = pnl_val
                        break
            elif "v4_cycle" in line:
                latest_cycle = _parse_structlog(line)
            elif "v4_new_period" in line:
                latest_cycle.update(_parse_structlog(line))
            elif "fast_cycle" in line:
                latest_cycle = _parse_structlog(line)
            elif "paper_trade" in line:
                parts = _parse_structlog(line)
                market_q = parts.get("market", "")
                # Extract asset from question
                asset = "?"
                for a in ["Bitcoin", "Ethereum", "Solana", "XRP"]:
                    if a.lower() in market_q.lower():
                        asset = {"bitcoin":"BTC","ethereum":"ETH","solana":"SOL","xrp":"XRP"}[a.lower()]
                        break
                trades.append({
                    "question": market_q,
                    "side": parts.get("side", ""),
                    "entry_price": float(parts.get("entry", "0").replace("$", "")),
                    "size_usd": float(parts.get("size", "0").replace("$", "")),
                    "asset": asset,
                    "binance_at_entry": float(parts.get("binance", "0").replace("$", "").replace(",", "")),
                    "edge": parts.get("edge", ""),
                    "confidence": parts.get("confidence", ""),
                    "status": "open",
                    "pnl": 0,
                })
            elif "crypto_arb_cycle" in line:
                latest_cycle = _parse_structlog(line)
            elif "paper_resolved" in line:
                parts = _parse_structlog(line)
                for t in trades:
                    if parts.get("market", "") in t.get("question", ""):
                        t["status"] = parts.get("status", "open")
                        t["pnl"] = float(parts.get("pnl", "0").replace("$", "").replace("+", ""))
                        break

        # Extract prices from latest cycle
        prices = {}
        price_str = latest_cycle.get("prices", "")
        for chunk in price_str.split("|"):
            chunk = chunk.strip()
            if "=" in chunk:
                sym, _, val = chunk.partition("=")
                try:
                    prices[sym.strip()] = float(val.strip().replace("$", "").replace(",", ""))
                except ValueError:
                    pass

        bankroll = latest_cycle.get("bankroll", "")
        open_pos = int(latest_cycle.get("open_positions", 0))
        markets_scanned = int(latest_cycle.get("markets", 0))

        # Compute totals
        won = sum(1 for t in trades if t["status"] == "won")
        lost = sum(1 for t in trades if t["status"] == "lost")
        closed = won + lost
        total_pnl = sum(t["pnl"] for t in trades if t["status"] != "open")
        total_invested = sum(t["size_usd"] for t in trades)

        # Detect v4 engine
        is_v4 = "v4_cycle" in str(latest_cycle) or any(
            k in latest_cycle for k in ("signals", "ticks", "period_remaining")
        )

        result.update({
            "bankroll": bankroll,
            "total_pnl": f"${total_pnl:+.2f}",
            "total_pnl_num": total_pnl,
            "win_rate": f"{won/closed*100:.0f}%" if closed > 0 else "--",
            "roi": f"{(total_pnl/total_invested*100):+.1f}%" if total_invested > 0 else "--",
            "total_trades": len(trades),
            "open": open_pos,
            "won": won,
            "lost": lost,
            "trades": trades,
            "prices": prices,
            "markets_scanned": markets_scanned,
            "signals": int(latest_cycle.get("signals", 0)),
            "skip_repriced": int(latest_cycle.get("skip_repriced", 0)),
            "ticks": int(latest_cycle.get("ticks", 0)),
            "period_remaining": latest_cycle.get("period_remaining", ""),
            "version": "v4" if is_v4 else "legacy",
        })
    except Exception as e:
        logger.warning("crypto_arb_status_parse_error", error=str(e))

    return result


@app.get("/api/all-status")
async def all_status():
    """Unified status endpoint for dashboard — reads all bot logs and state files."""
    import json as jsonmod
    import pathlib
    import re as _re

    base = pathlib.Path(__file__).parent.parent
    result = {
        "sniper": {"bankroll": 0, "pnl": 0, "trades": 0, "wins": 0, "losses": 0,
                   "win_rate": "--", "open": 0, "btc": 0, "period_elapsed": "",
                   "circuit": "ok", "recent_trades": [], "recent_resolved": []},
        "certainty_sniper": {"bankroll": 0, "pnl": 0, "wins": 0, "losses": 0,
                             "win_rate": "--", "open": 0, "btc": 0, "period_elapsed": "",
                             "circuit": "ok", "recent_trades": [], "recent_resolved": []},
        "power_trader": {"bankroll": 0, "positions": [], "total_invested": 0},
        "wallet_usdc": 0,
        "ai_monitor": {"status": "none", "summary": "", "issues": "", "timestamp": ""},
    }

    def _parse(line):
        parts = {}
        for m in _re.finditer(r"(\w+)='([^']*)'|(\w+)=(\S+)", line):
            if m.group(1):
                parts[m.group(1)] = m.group(2)
            elif m.group(3):
                parts[m.group(3)] = m.group(4)
        return parts

    # --- Momentum Sniper bot ---
    slog = base / "sniper_output.log"
    if slog.exists():
        try:
            lines = slog.read_text().strip().split("\n")
            trades = []
            resolved = []
            latest = {}
            skip_remaining = 0
            cancelled_ids: set = set()
            for line in lines[-1000:]:
                if "sniper_status" in line:
                    latest = _parse(line)
                elif "SNIPER_TRADE" in line:
                    trades.append(_parse(line))
                elif "SNIPER_CANCELLED" in line:
                    p = _parse(line)
                    cancelled_ids.add(p.get("order_id", ""))
                elif "SNIPER_RESOLVED" in line:
                    resolved.append(_parse(line))
                elif "CIRCUIT_BREAKER" in line and "TICK" not in line:
                    p = _parse(line)
                    skip_remaining = int(p.get("skipping_next", 4))
                elif "CIRCUIT_BREAKER_TICK" in line:
                    p = _parse(line)
                    skip_remaining = int(p.get("skip_remaining", 0))
            for t in trades:
                t["status"] = "cancelled" if t.get("order_id", "") in cancelled_ids else "open"

            bankroll = float(latest.get("bankroll", "0").replace("$", ""))
            pnl = float(latest.get("pnl", "0").replace("$", "").replace("+", ""))
            btc = float(latest.get("btc", "0"))
            tr = latest.get("trades", "0W/0L")
            wins = int(tr.split("W")[0]) if "W" in tr else 0
            losses = int(tr.split("/")[1].replace("L", "")) if "/" in tr else 0

            result["sniper"] = {
                "bankroll": bankroll,
                "pnl": pnl,
                "trades": wins + losses,
                "wins": wins,
                "losses": losses,
                "win_rate": latest.get("win_rate", "--"),
                "open": int(latest.get("open", "0")),
                "btc": btc,
                "period_elapsed": latest.get("period_elapsed", ""),
                "circuit": latest.get("circuit", "ok"),
                "skip_remaining": skip_remaining,
                "recent_trades": trades[-10:],
                "recent_resolved": resolved[-10:],
            }
        except Exception:
            pass

    # --- Certainty Sniper bot ---
    cslog = base / "certainty_sniper_output.log"
    if cslog.exists():
        try:
            lines = cslog.read_text().strip().split("\n")
            cs_trades = []
            cs_resolved = []
            cs_latest = {}
            cs_cancelled_ids: set = set()
            for line in lines[-1000:]:
                if "certainty_status" in line:
                    cs_latest = _parse(line)
                elif "CERTAINTY_TRADE" in line:
                    cs_trades.append(_parse(line))
                elif "CERTAINTY_CANCELLED" in line:
                    p = _parse(line)
                    cs_cancelled_ids.add(p.get("order_id", ""))
                elif "CERTAINTY_RESOLVED" in line:
                    cs_resolved.append(_parse(line))
            for t in cs_trades:
                t["status"] = "cancelled" if t.get("order_id", "") in cs_cancelled_ids else "open"

            cs_bankroll = float(cs_latest.get("bankroll", "0").replace("$", ""))
            cs_pnl = float(cs_latest.get("pnl", "0").replace("$", "").replace("+", ""))
            cs_tr = cs_latest.get("trades", "0W/0L")
            cs_wins = int(cs_tr.split("W")[0]) if "W" in cs_tr else 0
            cs_losses = int(cs_tr.split("/")[1].replace("L", "")) if "/" in cs_tr else 0
            result["certainty_sniper"] = {
                "bankroll": cs_bankroll,
                "pnl": cs_pnl,
                "wins": cs_wins,
                "losses": cs_losses,
                "win_rate": cs_latest.get("win_rate", "--"),
                "open": int(cs_latest.get("open", "0")),
                "btc": float(cs_latest.get("btc", "0")),
                "period_elapsed": cs_latest.get("period_elapsed", ""),
                "circuit": cs_latest.get("circuit", "ok"),
                "recent_trades": cs_trades[-10:],
                "recent_resolved": cs_resolved[-10:],
            }
        except Exception:
            pass

    # --- Power trader (state file + live log) ---
    pt_state = base / "data" / "power_trade_state.json"
    if pt_state.exists():
        try:
            state = jsonmod.loads(pt_state.read_text())
            positions = []
            for t in state.get("trades", []):
                if t.get("status") == "open":
                    positions.append({
                        "question": t.get("question", "")[:50],
                        "side": t.get("side", ""),
                        "price": t.get("price", 0),
                        "size": t.get("size_usd", 0),
                        "tokens": t.get("tokens", 0),
                        "entry_time": t.get("entry_time", ""),
                    })
            result["power_trader"] = {
                "bankroll": state.get("bankroll", 0),
                "positions": positions,
                "total_invested": state.get("total_invested", 0),
            }
        except Exception:
            pass

    # Augment power trader with live log data
    ptlog = base / "power_trade_output.log"
    if ptlog.exists():
        try:
            lines = ptlog.read_text().strip().split("\n")
            pt_latest = {}
            pt_signals = []
            pt_trades = []
            pt_sleeping = ""
            for line in lines[-500:]:  # only last 500 lines for performance
                if "power_status" in line:
                    pt_latest = _parse(line)
                elif "power_signal" in line:
                    pt_signals.append(_parse(line))
                elif "power_executing" in line:
                    pt_trades.append(_parse(line))
                elif "power_sleeping" in line:
                    p = _parse(line)
                    pt_sleeping = p.get("next", "")
                elif "trade_error" in line:
                    pass  # errors counted separately

            if pt_latest:
                result["power_trader"].update({
                    "bankroll": float(pt_latest.get("bankroll", result["power_trader"]["bankroll"]).replace("$", "")),
                    "cycle": int(pt_latest.get("cycle", 0)),
                    "total_trades": int(pt_latest.get("total", 0)),
                    "open_trades": int(pt_latest.get("open", 0)),
                    "sleeping_until": pt_sleeping,
                    "recent_signals": pt_signals[-5:],
                    "recent_executions": [{**t, "size": float(t.get("size", "0").replace("$", ""))} for t in pt_trades[-5:]],
                })
        except Exception:
            pass

    # --- Wallet USDC (cached from last redemption/update) ---
    wallet_state = base / "data" / "wallet_state.json"
    if wallet_state.exists():
        try:
            ws = jsonmod.loads(wallet_state.read_text())
            result["wallet_usdc"] = float(ws.get("usdc", 0))
        except Exception:
            pass

    # --- Daily compounder ---
    dc_state = base / "data" / "daily_compounder_state.json"
    if dc_state.exists():
        try:
            state = jsonmod.loads(dc_state.read_text())
            dc_trades = [t for t in state.get("trades", []) if t.get("status") == "open"]
            dc_closed = [t for t in state.get("trades", []) if t.get("status") in ("won", "lost")]
            dc_wins = sum(1 for t in dc_closed if t.get("status") == "won")
            dc_pnl = sum(t.get("pnl", 0) for t in dc_closed if isinstance(t.get("pnl"), (int, float)))
            result["daily_compounder"] = {
                "bankroll": state.get("bankroll", 0),
                "open_trades": len(dc_trades),
                "closed_trades": len(dc_closed),
                "wins": dc_wins,
                "pnl": dc_pnl,
                "total_invested": state.get("total_invested", 0),
            }
        except Exception:
            pass

    return result


@app.get("/api/ai-monitor")
async def ai_monitor_status():
    """Read the latest AI monitor verdict."""
    import pathlib
    log_path = pathlib.Path(__file__).parent.parent / "data" / "ai_monitor.log"
    result = {"status": "none", "summary": "", "issues": "", "timestamp": ""}
    if not log_path.exists():
        return result
    try:
        text = log_path.read_text().strip()
        # Parse the last verdict block (between ── separators)
        blocks = text.split("──────────────────────────────────────────")
        if len(blocks) >= 2:
            last_block = blocks[-1].strip()
            for line in last_block.split("\n"):
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    result["timestamp"] = line.strip("[]")
                elif line.startswith("STATUS:"):
                    result["status"] = line.split(":", 1)[1].strip()
                elif line.startswith("SUMMARY:"):
                    result["summary"] = line.split(":", 1)[1].strip()
                elif line.startswith("ISSUES:"):
                    result["issues"] = line.split(":", 1)[1].strip()
    except Exception as e:
        logger.warning("ai_monitor_parse_error", error=str(e))
    return result


# --- Admin ---


@app.post("/admin/kill-switch")
async def activate_kill_switch():
    with open(settings.kill_switch_file, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat())
    logger.warning("kill_switch_activated")
    return {"status": "kill_switch_active"}


@app.delete("/admin/kill-switch")
async def deactivate_kill_switch():
    if os.path.exists(settings.kill_switch_file):
        os.remove(settings.kill_switch_file)
    logger.info("kill_switch_deactivated")
    return {"status": "kill_switch_cleared"}
