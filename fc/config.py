from pydantic import BaseModel, Field
from typing import List
import yaml
from pathlib import Path

class EloConfig(BaseModel):
    start: float = 1500.0
    k: float = 20.0
    hfa: float = 60.0
    mov_factor: float = 0.0     # margin-of-victory scaling (0 = off)
    season_regression: float = 0.0
    time_decay_days: float = 0.0
    adaptive_k: bool = True
    home_away_split: bool = True

class FeaturesConfig(BaseModel):
    rolling_n: int = 5
    ewm_alpha: float = 0.4
    max_goals: int = 8
    add_features: bool = True
    safe_fill: bool = True
    include_advanced: bool = True
    # nuovi flag opzionali usati in train
    use_advanced_stats: bool = True
    use_h2h: bool = True
    use_xg_proxy: bool = True
    use_xg_real: bool = True
    use_shots_real: bool = True
    h2h_matches: int = 5

class CalibrationConfig(BaseModel):
    enabled: bool = True
    method: str = "isotonic"

class GBMConfig(BaseModel):
    enabled: bool = True
    blend_weight: float = 0.7   # 0..1, quanto pesare GBM rispetto al Poisson

class DrawMetaConfig(BaseModel):
    enabled: bool = True
    blend_weight: float = 0.4

class DrawBoosterConfig(BaseModel):
    enabled: bool = True
    elo_abs_diff_max: float = 35.0
    goals_ewm_sum_max: float = 2.4
    market_draw_min: float = 0.34
    weight: float = 0.15
    max_boost: float = 0.06
    # Se true, promuove X quando è molto vicino al top1 (near-tie)
    promote_near_tie: bool = True
    tie_margin: float = 0.02
    # Gate: evita booster/near-tie quando c'è un forte favorito di mercato
    skip_booster_if_favorite: bool = True
    skip_near_tie_if_favorite: bool = True
    favorite_prob_min: float = 0.60

class MarketGuardrailsConfig(BaseModel):
    enabled: bool = True
    # Limiti per scostamento dalle probabilità implicite del mercato
    max_abs_diff_home: float = 0.18
    max_abs_diff_draw: float = 0.14
    max_abs_diff_away: float = 0.18
    # Clipping dolce: blend verso mercato se oltre soglia
    blend_weight: float = 0.5

class ModelConfig(BaseModel):
    alpha: float = 1.0      # L2 reg PoissonRegressor
    use_dixon_coles: bool = True
    dc_rho: float = 0.05
    use_ensemble: bool = True
    ensemble_weight: float = 0.7
    robust_sanitization: bool = True
    adaptive_clipping: bool = True
    calibration: CalibrationConfig = CalibrationConfig()
    gbm: GBMConfig = GBMConfig()
    market_blend_weight: float = 0.25
    draw_weight: float = 1.9        # peso per rinforzare la classe pareggio nelle fasi discriminative
    draw_booster: DrawBoosterConfig = DrawBoosterConfig()
    market_guardrails: MarketGuardrailsConfig = MarketGuardrailsConfig()
    draw_meta: DrawMetaConfig = DrawMetaConfig()

class BacktestConfig(BaseModel):
    n_splits: int = 6
    gap: int = 0       # optional: gap between train and test
    tune: bool = True
    tune_trials: int = 30

class DataConfig(BaseModel):
    paths: List[str] = Field(default_factory=list)      # historical CSV
    delimiter: str = ","
    use_market: bool = False
    xg_path: str | None = None
    shots_path: str | None = None

class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    elo: EloConfig = EloConfig()
    features: FeaturesConfig = FeaturesConfig()
    model: ModelConfig = ModelConfig()
    backtest: BacktestConfig = BacktestConfig()
    artifacts_dir: str = "artifacts/models"

def load_config(path: str | Path) -> AppConfig:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
