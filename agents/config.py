"""Shared configuration for the commodities trading system.

Single source of truth for all parameters, paths, and commodity definitions.
"""

from pathlib import Path
from dataclasses import dataclass, field


# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent           # repo root
COMMODITIES_DIR = ROOT_DIR                        # agents/ is at repo root level
LOGS_DIR = ROOT_DIR / "logs"


# ── Commodity definitions ──────────────────────────────────────────────
@dataclass
class CommodityConfig:
    name: str
    ticker: str
    price_col: str
    dir_name: str  # directory name under ROOT_DIR
    horizon: int = 63
    confidence_threshold: float = 0.75
    model_version: str = "v1"

    @property
    def project_dir(self) -> Path:
        return ROOT_DIR / self.dir_name

    @property
    def models_dir(self) -> Path:
        return self.project_dir / "models"

    @property
    def data_dir(self) -> Path:
        return self.project_dir / "data"

    @property
    def metadata_path(self) -> Path:
        return self.models_dir / "production_metadata.json"


COMMODITIES = {
    "coffee": CommodityConfig(
        name="Coffee", ticker="KC=F", price_col="coffee_close",
        dir_name="coffee", confidence_threshold=0.75, model_version="v3",
    ),
    "cocoa": CommodityConfig(
        name="Cocoa", ticker="CC=F", price_col="cocoa_close",
        dir_name="chocolate", confidence_threshold=0.80, model_version="v5",
    ),
    "sugar": CommodityConfig(
        name="Sugar", ticker="SB=F", price_col="sugar_close",
        dir_name="sugar", confidence_threshold=0.75, model_version="v3",
    ),
    "natgas": CommodityConfig(
        name="Natural Gas", ticker="NG=F", price_col="natgas_close",
        dir_name="natgas", confidence_threshold=0.95, model_version="v3",
    ),
    "soybeans": CommodityConfig(
        name="Soybeans", ticker="ZS=F", price_col="soybeans_close",
        dir_name="soybeans", confidence_threshold=0.80, model_version="v3",
    ),
    "wheat": CommodityConfig(
        name="Wheat", ticker="ZW=F", price_col="wheat_close",
        dir_name="wheat", confidence_threshold=0.75, model_version="v5",
    ),
    "copper": CommodityConfig(
        name="Copper", ticker="HG=F", price_col="copper_close",
        dir_name="copper", confidence_threshold=0.75, model_version="v5",
        horizon=63,
    ),
}


# ── Correlation groups for risk management ─────────────────────────────
CORRELATION_GROUPS = {
    "brazil_soft": ["coffee", "sugar"],
    "grains": ["soybeans", "wheat"],
    "energy": ["natgas"],
    "industrial": ["copper"],
    "tropical": ["cocoa"],
}


# ── Position sizing ────────────────────────────────────────────────────
@dataclass
class SizingConfig:
    """Position sizing parameters."""
    kelly_fraction: float = 0.5
    base_sizes: dict = field(default_factory=lambda: {
        "75-80": 0.08,
        "80-85": 0.12,
        "85-90": 0.18,
        "90+": 0.22,
    })
    correlated_cut: float = 0.70       # 30% cut when correlated signals fire
    portfolio_overload_cut: float = 0.80  # 20% cut when 4+ signals fire
    max_portfolio_exposure: float = 0.50  # Never exceed 50% total exposure
    equity_size: float = 0.04           # 4% per equity trade


SIZING = SizingConfig()


# ── Trading defaults ───────────────────────────────────────────────────
@dataclass
class TradeDefaults:
    stop_loss_pct: float = 0.10
    take_profit_multiplier: float = 1.0
    max_hold_days: int = 63
    allow_short: bool = True


TRADE_DEFAULTS = TradeDefaults()


# ── Data pipeline ──────────────────────────────────────────────────────
DATA_STALENESS_WARN_DAYS = 3   # Warn if data older than this
DATA_STALENESS_FAIL_DAYS = 7   # Fail if data older than this

FETCH_SCRIPTS = [
    "fetch_data.py",
    "fetch_cot.py",
    "fetch_weather.py",
    "fetch_enso.py",
]


# ── Model training ─────────────────────────────────────────────────────
TRAINING_DEFAULTS = {
    "n_splits": 5,
    "test_size": 63,
    "min_train_size": 504,
    "optuna_trials": 200,
    "early_stopping_rounds": 30,
    "random_state": 42,
}
