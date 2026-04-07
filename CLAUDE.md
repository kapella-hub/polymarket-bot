# Polymarket Trading Bot

## Project Overview
Category-agnostic prediction market trading bot for Polymarket. Uses Claude CLI as core probability analyst with alpha/ensemble architecture for trade decisions.

## Architecture
Pipeline with signal store. LLM batch analysis (every 10-15 min) writes probability estimates to PostgreSQL. Strategy loop (every 30-60s) reads cached signals, runs ensemble, executes trades. Single async process.

## Tech Stack
- Python 3.12+ / FastAPI / async
- PostgreSQL via SQLAlchemy async (asyncpg driver)
- py-clob-client for Polymarket CLOB API
- httpx for Gamma API calls
- Pydantic for models and config
- structlog for structured logging
- prometheus-client for metrics

## Conventions
- All config via environment variables with PM_ prefix (see src/config.py)
- Async everywhere (async def, await, asyncpg)
- Type hints on all function signatures
- Structured logging via structlog (no print statements)
- Tests in tests/ mirroring src/ structure

## Key Patterns
- Exchange adapter: abstract base -> concrete adapter (src/exchange/)
- Alpha sources: compute(market, context) -> AlphaOutput (src/alpha/)
- Intent state machine: CREATED -> ARMED -> EXECUTED/EXPIRED/CANCELLED (src/execution/)
- Risk controls: kill switch, position limits, drawdown (src/risk/)

## Running
```bash
# Local development
cp .env.example .env  # Fill in API keys
pip install -r requirements.txt
uvicorn src.main:app --reload

# Docker
docker compose up -d
```
