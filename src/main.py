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
                decisions = []

                for market in markets:
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
    global _exchange, _gamma, _filter, _batch_scheduler, _ensemble, _executor, _nexus

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
    _filter = MarketFilter()
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
    _ensemble = EnsembleEngine(alphas=[
        LLMAlpha(),
        OrderBookAlpha(),
        SmartMoneyAlpha(tracker=_smart_money),
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
