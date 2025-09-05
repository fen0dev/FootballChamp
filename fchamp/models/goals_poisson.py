from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from joblib import dump, load
from fchamp.features.dixon_coles import dc_score_matrix
from scipy.stats import poisson
import logging

logger = logging.getLogger(__name__)

@dataclass
class GoalsPoissonModel:
    alpha: float = 1.0
    use_dixon_coles: bool = True
    dc_rho: float = 0.05
    feature_cols: List[str] = None
    use_ensemble: bool = False           # Se True, usa ensemble con GBM
    ensemble_weight: float = 0.7         # Peso del modello Poisson (0.3 per GBM)
    robust_sanitization: bool = True     # Sanitizzazione più robusta
    adaptive_clipping: bool = False      # Clipping adattivo basato su dati

    def __post_init__(self):
        self.model_home = PoissonRegressor(alpha=self.alpha, max_iter=2000)
        self.model_away = PoissonRegressor(alpha=self.alpha, max_iter=2000)
        self.feature_cols = []
        
        # 🚀 ENSEMBLE COMPONENTS (opzionali)
        if self.use_ensemble:
            # Usa regressori Poisson per predire i gol attesi (lambdas)
            self.gbm_home = HistGradientBoostingRegressor(
                loss="poisson", early_stopping=True, random_state=42
            )
            self.gbm_away = HistGradientBoostingRegressor(
                loss="poisson", early_stopping=True, random_state=42
            )
            logger.info("🚀 Initialized ensemble mode with GBM Poisson Regressors")

    def _sanitize(self, X: pd.DataFrame) -> pd.DataFrame:
        """🚀 ENHANCED sanitization con opzioni avanzate"""
        Xs = X[self.feature_cols].copy()
        Xs = Xs.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        
        if self.robust_sanitization:
            # 🚀 SANITIZZAZIONE ROBUSTA: usa defaults sensati invece di ffill
            safe_defaults = {
                "elo_home_pre": 1500.0, "elo_away_pre": 1500.0, "elo_diff": 0.0,
                "home_gf_roll": 1.3, "home_ga_roll": 1.3, "away_gf_roll": 1.1, "away_ga_roll": 1.1,
                "home_gf_ewm": 1.3, "away_gf_ewm": 1.1,
                "book_logit_diff": 0.0
            }
            
            for col in Xs.columns:
                if col in safe_defaults:
                    Xs[col] = Xs[col].fillna(safe_defaults[col])
                else:
                    Xs[col] = Xs[col].fillna(0.0)
        else:
            # Comportamento originale
            Xs = Xs.ffill().fillna(0.0)
        
        Xs = Xs.astype(float)
        
        # 🚀 ADAPTIVE CLIPPING (opzionale)
        if self.adaptive_clipping:
            # Clipping basato su percentili dei dati invece di valori fissi
            for col in Xs.columns:
                q01, q99 = Xs[col].quantile([0.01, 0.99])
                Xs[col] = Xs[col].clip(lower=q01, upper=q99)
        else:
            # Clipping originale
            Xs = Xs.clip(lower=-10.0, upper=10.0)

        return Xs

    def market_probs(self, lh: float, la: float, max_goals: int) -> dict:
        """🚀 ENHANCED market probabilities con più mercati"""
        M = self.score_matrix(lh, la, max_goals)

        p1 = float(np.tril(M, - 1).sum())
        px = float(np.trace(M))
        p2 = float(np.triu(M, 1).sum())

        p_1x = p1 + px
        p_12 = 1.0 - px
        p_x2 = px + p2

        p_over_1_5 = float(sum(M[i, j] for i in range(M.shape[0]) for j in range(M.shape[1]) if i + j >= 2))
        p_over_2_5 = float(sum(M[i, j] for i in range(M.shape[0]) for j in range(M.shape[1]) if i + j >= 3))
        
        # 🚀 MERCATI AGGIUNTIVI
        p_over_0_5 = 1.0 - float(M[0, 0])  # Almeno un gol
        p_over_3_5 = float(sum(M[i, j] for i in range(M.shape[0]) for j in range(M.shape[1]) if i + j >= 4))
        p_under_2_5 = 1.0 - p_over_2_5

        p_btts_yes = float(M[1:, 1:].sum())
        p_btts_no = 1.0 - p_btts_yes

        p_home_scores = 1.0 - float(M[0, :].sum())
        p_away_scores = 1.0 - float(M[:, 0].sum())
        
        # 🚀 EXACT SCORE PROBS (top scores)
        exact_scores = {}
        for i in range(min(4, M.shape[0])):
            for j in range(min(4, M.shape[1])):
                exact_scores[f"{i}-{j}"] = float(M[i, j])

        return {
            "p1": p1, "px": px, "p2": p2,
            "p_1x": p_1x, "p_12": p_12, "p_x2": p_x2,
            "p_over_0_5": p_over_0_5, "p_over_1_5": p_over_1_5, 
            "p_over_2_5": p_over_2_5, "p_over_3_5": p_over_3_5,
            "p_under_2_5": p_under_2_5,
            "p_btts_yes": p_btts_yes, "p_btts_no": p_btts_no,
            "p_home_scores": p_home_scores, "p_away_scores": p_away_scores,
            "exact_scores": exact_scores,
            "lambda_home": lh, "lambda_away": la
        }

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        """🚀 ENHANCED fit con ensemble opzionale"""
        self.feature_cols = list(X.columns)
        Xs = self._sanitize(X)

        # 🎯 NEW: Identificazione automatica feature avanzate
        advanced_features = [col for col in X.columns if any(
            keyword in col for keyword in [
                'shots', 'corners', 'discipline', 'xg', 
                'h2h', 'ref_', 'shot_eff', 'shot_acc'
            ]
        )]
        
        if advanced_features:
            logger.info(f"🎯 Using {len(advanced_features)} advanced features in model")
            
            # Crea ensemble con peso maggiore per feature avanzate
            if self.use_ensemble:
                # Split features per importanza
                core_features = [col for col in X.columns if col not in advanced_features]
                
                # Train modelli separati e poi combina
                self.core_model_home = PoissonRegressor(alpha=self.alpha, max_iter=2000)
                self.core_model_away = PoissonRegressor(alpha=self.alpha, max_iter=2000)
                self.core_model_home.fit(X[core_features], y_home)
                self.core_model_away.fit(X[core_features], y_away)
                
                self.advanced_model_home = GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1
                )
                self.advanced_model_away = GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1
                )
                self.advanced_model_home.fit(X[advanced_features], y_home)
                self.advanced_model_away.fit(X[advanced_features], y_away)
        
        # Fit modelli Poisson (sempre)
        self.model_home.fit(Xs, y_home)
        self.model_away.fit(Xs, y_away)
        
        # 🚀 FIT ENSEMBLE (opzionale)
        if self.use_ensemble:
            logger.info("🚀 Training ensemble models")
            
            # Converti targets per GBM (che si aspetta interi non-negativi)
            y_home_int = y_home.astype(int).clip(lower=0, upper=10)
            y_away_int = y_away.astype(int).clip(lower=0, upper=10)
            
            try:
                self.gbm_home.fit(Xs, y_home_int)
                self.gbm_away.fit(Xs, y_away_int)
                logger.info("🚀 Ensemble models trained successfully")
            except Exception as e:
                logger.warning(f"Ensemble training failed: {e}, using Poisson only")
                self.use_ensemble = False

    def predict_lambdas(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """🚀 ENHANCED prediction con ensemble opzionale"""
        Xs = self._sanitize(X)

        # Predizioni Poisson (sempre) con clamp minimo più alto per evitare degenerazioni
        lh_poisson = np.clip(self.model_home.predict(Xs), 0.2, 10.0)
        la_poisson = np.clip(self.model_away.predict(Xs), 0.2, 10.0)
        
        # 🚀 ENSEMBLE PREDICTIONS (opzionale)
        if self.use_ensemble and hasattr(self, 'gbm_home'):
            try:
                # GBM predictions (già clippate internamente)
                lh_gbm = np.clip(self.gbm_home.predict(Xs), 0.2, 10.0)
                la_gbm = np.clip(self.gbm_away.predict(Xs), 0.2, 10.0)
                
                # 🚀 WEIGHTED ENSEMBLE
                w = self.ensemble_weight
                lh = w * lh_poisson + (1 - w) * lh_gbm
                la = w * la_poisson + (1 - w) * la_gbm
                
                logger.debug(f"🚀 Used ensemble: Poisson weight={w:.2f}")
                return lh, la
                
            except Exception as e:
                logger.warning(f"Ensemble prediction failed: {e}, using Poisson only")
        
        # Fallback a Poisson puro
        return lh_poisson, la_poisson

    def score_matrix(self, lh: float, la: float, max_goals: int) -> np.ndarray:
        """Score matrix con miglioramenti numerici"""
        if self.use_dixon_coles:
            M = dc_score_matrix(lh, la, rho=self.dc_rho, max_goals=max_goals)
        else:
            gh = np.arange(0, max_goals + 1)
            ga = np.arange(0, max_goals + 1)
            M = np.outer(poisson.pmf(gh, lh), poisson.pmf(ga, la))
        
        # 🚀 NUMERICAL STABILITY
        M = np.maximum(M, 1e-10)  # Evita probabilità zero
        M = M / M.sum()  # Renormalizza
        
        return M

    def outcome_probs(self, lh: float, la: float, max_goals: int) -> Tuple[float, float, float]:
        M = self.score_matrix(lh, la, max_goals)

        p_home = np.tril(M, -1).sum()
        p_draw = np.trace(M)
        p_away = np.triu(M, 1).sum()

        return float(p_home), float(p_draw), float(p_away)

    def get_model_info(self) -> dict:
        """🚀 NEW: Info complete sul modello"""
        info = {
            "alpha": self.alpha,
            "use_dixon_coles": self.use_dixon_coles,
            "dc_rho": self.dc_rho,
            "feature_count": len(self.feature_cols) if self.feature_cols else 0,
            "features": self.feature_cols,
            "use_ensemble": self.use_ensemble,
            "ensemble_weight": self.ensemble_weight if self.use_ensemble else None,
            "robust_sanitization": self.robust_sanitization,
            "adaptive_clipping": self.adaptive_clipping
        }
        
        return info

    def save(self, path: str):
        """🚀 ENHANCED save con parametri avanzati"""
        data = {
            "alpha": self.alpha,
            "use_dixon_coles": self.use_dixon_coles,
            "dc_rho": self.dc_rho,
            "feature_cols": self.feature_cols,
            "model_home": self.model_home,
            "model_away": self.model_away,
            "use_ensemble": self.use_ensemble,
            "ensemble_weight": self.ensemble_weight,
            "robust_sanitization": self.robust_sanitization,
            "adaptive_clipping": self.adaptive_clipping
        }
        
        # Salva anche ensemble se presente
        if self.use_ensemble and hasattr(self, 'gbm_home'):
            data["gbm_home"] = self.gbm_home
            data["gbm_away"] = self.gbm_away
        
        dump(data, path)

    @classmethod
    def load(cls, path: str) -> "GoalsPoissonModel":
        """🚀 ENHANCED load con backward compatibility"""
        d = load(path)

        # Parametri base (backward compatible)
        obj = cls(
            alpha=d["alpha"], 
            use_dixon_coles=d["use_dixon_coles"], 
            dc_rho=d["dc_rho"]
        )
        
        obj.use_ensemble = d.get("use_ensemble", False)
        obj.ensemble_weight = d.get("ensemble_weight", 0.7)
        obj.robust_sanitization = d.get("robust_sanitization", True)
        obj.adaptive_clipping = d.get("adaptive_clipping", True)
        
        # Modelli base
        obj.feature_cols = d["feature_cols"]
        obj.model_home = d["model_home"]
        obj.model_away = d["model_away"]
        
        # 🚀 ENSEMBLE MODELS (se presenti)
        if obj.use_ensemble and "gbm_home" in d:
            obj.gbm_home = d["gbm_home"]
            obj.gbm_away = d["gbm_away"]
        
        return obj