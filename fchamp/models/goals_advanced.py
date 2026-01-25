import numpy as np
import pandas as pd
from dataclasses import dataclass
from joblib import dump, load
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, logsumexp
import logging
from fchamp.models.goals_poisson import GoalsPoissonModel

logger = logging.getLogger(__name__)

def _safe_clip_probs(M: np.ndarray) -> np.ndarray:
    M = np.maximum(M, 1e-12)
    s = float(M.sum())

    return M / s if s > 0 else np.full_like(M, 1.0 / M.size)

def _bivar_logpmf(x: int, y: int, lam1: float, lam2: float, lam3: float) -> float:
    # X = U + W, Y = V + W, U~Pois(lam1), v~Pois(lam2), W~Pois(lam3)
    # P(X=x, Y=y) = sum_{k=0..min(x,y)} e^{-(lam1+lam2+lam3)} lam1^{x-k} lam2^{y-k} lam3^k /((x-k)!(y-k)!k!)
    m = min(x, y)

    if lam1 < 0 or lam2 < 0 or lam3 < 0:
        return -np.inf

    base = -(lam1 + lam2 + lam3)
    terms = []

    for k in range(m + 1):
        a = x - k
        b = y - k

        terms.append(
            base
            + a * np.log(lam1 + 1e-12) - gammaln(a + 1)
            + b * np.log(lam2 + 1e-12) - gammaln(b + 1)
            + k * np.log(lam3 + 1e-12) - gammaln(k + 1)
        )

    return float(logsumexp(np.array(terms, dtype=float)))

def _nb_logpmf(y: int, mu: float, theta: float) -> float:
    # NB2: Var = mu + mu^2/theta, theta > 0
    # pmf(y) = Γ(y+θ)/(Γ(θ) y!) * (θ/(θ+μ))^θ * (μ/(θ+μ))^y

    if mu <= 0 or theta <= 0:
        return -np.inf

    return float(
        gammaln(y + theta) - gammaln(theta) - gammaln(y + 1)
        + theta * np.log(theta / (theta + mu))
        + y * np.log(mu / (theta + mu))
    )

@dataclass
class GoalsBivariatePoissonModel(GoalsPoissonModel):
    # sigma in (0, max_sigma): lam3 = sigma * min(mu_h, mu_a)
    sigma: float = 0.10
    max_sigma: float = 0.30

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        super().fit(X, y_home, y_away)

        # estimated sigma on train (1D) given model's mu_h, mu_a 
        mu_h, mu_a = self.predict_lambdas(X)
        yh = y_home.astype(int).clip(lower=0, upper=10).values
        ya = y_away.astype(int).clip(lower=0, upper=10).values

        def nll(sig: float) -> float:
            sig = float(np.clip(sig, 0.0, self.max_sigma))
            ll = 0.0
            
            for i in range(len(yh)):
                mh = float(mu_h[i]); ma = float(mu_a[i])
                lam3 = sig * min(mh, ma)
                lam1 = max(mh - lam3, 1e-12)
                lam2 = max(ma - lam3, 1e-12)
                ll += _bivar_logpmf(int(yh[i]), int(ya[i]), lam1, lam2, lam3)
            
            return -ll

        try:
            res = minimize_scalar(nll, bounds=(0.0, self.max_sigma), method="bounded")
            if res.success:
                self.sigma = float(res.x)
                logger.info(f"🚀 Fitted bivariate sigma={self.sigma:.4f}")
        except Exception as e:
            logger.warning(f"[-] Bivariate sigma fit failed: {str(e)}. Using sigma={self.sigma:.3f}")

        return self

    def score_matrix(self, lh: float, la: float, max_goals: int) -> np.ndarray:
        # construct joint bivariate matrix
        sig = float(np.clip(self.sigma, 0.0, self.max_sigma))
        lam3 = sig * min(float(lh), float(la))
        lam1 = max(float(lh) - lam3, 1e-12)
        lam2 = max(float(la) - lam3, 1e-12)

        M = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                M[i, j] = np.exp(_bivar_logpmf(i, j, lam1, lam2, lam3))

        return _safe_clip_probs(M)

    def save(self, path: str):
        data = {
            "kind": "bivariate_poisson",
            "base": {
                "alpha": self.alpha,
                "use_dixon_coles": self.use_dixon_coles,
                "dc_rho": self.dc_rho,
                "feature_cols": self.feature_cols,
                "model_home": self.model_home,
                "model_away": self.model_away,
                "use_ensemble": self.use_ensemble,
                "ensemble_weight": self.ensemble_weight,
                "robust_sanitization": self.robust_sanitization,
                "adaptive_clipping": self.adaptive_clipping,
            },
            "sigma": self.sigma,
            "max_sigma": self.max_sigma,
        }

        if self.use_ensemble and hasattr(self, "gbm_home"):
            data["base"]["gbm_home"] = self.gbm_home
            data["base"]["gbm_away"] = self.gbm_away

        dump(data, path)

    @classmethod
    def load(cls, path: str) -> "GoalsBivariatePoissonModel":
        d = load(path)
        b = d["base"]
        obj = cls(
            alpha=b["alpha"],
            use_dixon_coles=b["use_dixon_coles"],
            dc_rho=b["dc_rho"],
            sigma=d.get("sigma", 0.10),
            max_sigma=d.get("max_sigma", 0.30),
        )
        obj.use_ensemble = b.get("use_ensemble", False)
        obj.ensemble_weight = b.get("ensemble_weight", 0.7)
        obj.robust_sanitization = b.get("robust_sanitization", True)
        obj.adaptive_clipping = b.get("adaptive_clipping", True)

        obj.feature_cols = b["feature_cols"]
        obj.model_home = b["model_home"]
        obj.model_away = b["model_away"]

        if obj.use_ensemble and "gbm_home" in b:
            obj.gbm_home = b["gbm_home"]
            obj.gbm_away = b["gbm_away"]
        return obj

@dataclass
class GoalsNegBinModel(GoalsPoissonModel):
    theta_home: float = 6.0
    theta_away: float = 6.0

    def fit(self, X: pd.DataFrame, y_home: pd.Series, y_away: pd.Series):
        super().fit(X, y_home, y_away)

        mu_h, mu_a = self.predict_lambdas(X)
        yh = y_home.astype(int).clip(lower=0, upper=10).values
        ya = y_away.astype(int).clip(lower=0, upper=10).values

        def fit_theta(y, mu):
            def nll(log_theta):
                theta = float(np.exp(log_theta))
                ll = 0.0

                for i in range(len(y)):
                    ll += _nb_logpmf(int(y[i]), float(mu[i]), theta)

                return -ll

            res = minimize_scalar(nll, bounds=(-2.0, 6.0), method="bounded")  # theta in [e^-2, e^6]
            return float(np.exp(res.x)) if res.success else 6.0

        try:
            self.theta_home = fit_theta(yh, mu_h)
            self.theta_away = fit_theta(ya, mu_a)
            logger.info(f"🚀 Fitted NB thetas: home={self.theta_home: .3f}, away={self.theta_away: .3f}")
        except Exception as e:
            logger.warning(f"[-] NB theta fit failed: {str(e)}. Using defaults.")

        return self


    def score_matrix(self, lh: float, la: float, max_goals: int) -> np.ndarray:
        gh = np.arange(0, max_goals + 1)
        ga = np.arange(0, max_goals + 1)

        ph = np.array([np.exp(_nb_logpmf(int(k), float(lh), float(self.theta_home))) for k in gh], dtype=float)
        pa = np.array([np.exp(_nb_logpmf(int(k), float(la), float(self.theta_away))) for k in ga], dtype=float)

        M = np.outer(ph, pa)
        return _safe_clip_probs(M)

    def save(self, path: str):
        data = {
            "kind": "negbin",
            "base": {
                "alpha": self.alpha,
                "use_dixon_coles": self.use_dixon_coles,
                "dc_rho": self.dc_rho,
                "feature_cols": self.feature_cols,
                "model_home": self.model_home,
                "model_away": self.model_away,
                "use_ensemble": self.use_ensemble,
                "ensemble_weight": self.ensemble_weight,
                "robust_sanitization": self.robust_sanitization,
                "adaptive_clipping": self.adaptive_clipping,
            },
            "theta_home": self.theta_home,
            "theta_away": self.theta_away,
        }

        if self.use_ensemble and hasattr(self, "gbm_home"):
            data["base"]["gbm_home"] = self.gbm_home
            data["base"]["gbm_away"] = self.gbm_away

        dump(data, path)

    @classmethod
    def load(cls, path: str) -> "GoalsNegBinModel":
        d = load(path)
        b = d["base"]
        obj = cls(
            alpha=b["alpha"],
            use_dixon_coles=b["use_dixon_coles"],
            dc_rho=b["dc_rho"],
            theta_home=d.get("theta_home", 6.0),
            theta_away=d.get("theta_away", 6.0),
        )
        obj.use_ensemble = b.get("use_ensemble", False)
        obj.ensemble_weight = b.get("ensemble_weight", 0.7)
        obj.robust_sanitization = b.get("robust_sanitization", True)
        obj.adaptive_clipping = b.get("adaptive_clipping", True)

        obj.feature_cols = b["feature_cols"]
        obj.model_home = b["model_home"]
        obj.model_away = b["model_away"]

        if obj.use_ensemble and "gbm_home" in b:
            obj.gbm_home = b["gbm_home"]
            obj.gbm_away = b["gbm_away"]

        return obj
