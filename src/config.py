"""Application configuration via environment variables."""

from enum import Enum
from pydantic_settings import BaseSettings


class ExecutionMode(str, Enum):
    SHADOW = "shadow"
    PAPER = "paper"
    LIVE = "live"


class Settings(BaseSettings):
    """All configuration loaded from environment variables."""

    # --- Execution ---
    execution_mode: ExecutionMode = ExecutionMode.SHADOW
    bot_name: str = "polymarket-bot"

    # --- Polymarket API ---
    polymarket_host: str = "https://clob.polymarket.com"
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""
    polymarket_chain_id: int = 137  # Polygon mainnet
    polymarket_wallet_private_key: str = ""

    # --- Gamma API (market discovery) ---
    gamma_api_host: str = "https://gamma-api.polymarket.com"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/polymarket_bot"

    # --- FastAPI ---
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Market Discovery ---
    discovery_interval_seconds: int = 300  # 5 minutes
    min_volume_usd: float = 50_000.0
    min_liquidity_usd: float = 10_000.0
    min_spread: float = 0.02  # 2 cents minimum spread for MM opportunities
    max_end_date_days: int = 180  # 6 months
    min_end_date_hours: int = 1  # At least 1 hour to resolution

    # --- LLM Batch Analysis ---
    llm_batch_interval_seconds: int = 900  # 15 minutes
    llm_batch_size: int = 20  # Markets per batch
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 2
    llm_signal_ttl_seconds: int = 1800  # 30 min before signal is stale
    llm_model: str = "sonnet"  # Claude model: "sonnet", "haiku", "opus"

    # --- Strategy Loop ---
    strategy_loop_interval_seconds: int = 30

    # --- Risk Management ---
    kill_switch_file: str = "KILL_SWITCH"
    max_position_per_market_usd: float = 500.0
    max_portfolio_exposure_pct: float = 0.80  # 80% of bankroll
    max_portfolio_drawdown_pct: float = 0.15  # 15% drawdown halt
    max_daily_loss_usd: float = 200.0
    max_daily_loss_pct: float = 0.10  # 10% daily loss
    bankroll_usd: float = 10000.0  # Actual account bankroll — used for Kelly sizing and risk caps
    max_correlated_exposure_usd: float = 1000.0  # Max across related markets
    kelly_fraction: float = 0.25  # Quarter-Kelly for safety
    min_edge_threshold: float = 0.05  # 5 cents minimum edge to trade

    # --- Bayesian Combo Strategy Filter ---
    strategy_filter_enabled: bool = True
    min_edge_confidence_product: float = 0.05  # edge * confidence must exceed this
    min_conviction_prob: float = 0.15  # prob must be < this or > (1 - this) for conviction
    category_blacklist: str = "Sports,Pop Culture"  # Comma-separated, weak LLM categories
    volume_edge_base: float = 0.05  # Base edge threshold for volume scaling
    volume_edge_scale: float = 0.05  # Additional edge per $2M volume
    min_bayesian_score: int = 2  # Minimum combined signal score to trade

    # --- Logging ---
    log_level: str = "INFO"
    log_format: str = "json"

    model_config = {
        "env_prefix": "PM_",
        "env_file": ".env",
        "case_sensitive": False,
    }


settings = Settings()
