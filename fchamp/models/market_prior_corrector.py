import numpy as np
from dataclasses import dataclass
from joblib import dump, load
from scipy.optimize import minimize
from scipy.special import logsumexp

def _softmax_logits(L: np.ndarray) -> np.ndarray:
    Z = L - logsumexp(L, axis=1, keepdims=True)
    P = np.exp(Z)

    result = np.clip(P, 1e-12, 1.0) / np.clip(P.sum(axis=1, keepdims=True), 1e-12, np.inf)

    return result

@dataclass
class MarketPriorCorrector:
    l2: float = 1.0
    standardize: bool = False
    W: np.ndarray | None = None     # shape (d,3)
    z_mean: np.ndarray | None = None
    z_std:np.ndarray | None = None

    def _prep_Z(self, Z: np.ndarray, fit: bool = False) -> np.ndarray:
        Z = np.asarray(Z, dtype=float)
        if not self.standardize:
            return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

        if fit or self.z_mean is None or self.z_std is None:
            self.z_mean = np.nanmean(Z, axis=0, keepdims=True)
            self.z_std = np.nanstd(Z, axis=0, keepdims=True)
            self.z_std = np.where(self.z_std < 1e-9, 1.0, self.z_std)

        Zs = (Z - self.z_mean) / self.z_std
        return np.nan_to_num(Zs, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, Z: np.ndarray, mk: np.ndarray, y: np.ndarray):
        """
            logit(P) = log(mk) + Z @ W
            Z: (n,d), mk:(n,3), y in {0,1,2}
        """
        Z = self._prep_Z(Z, fit=True)
        n, d = Z.shape
        mk = np.clip(mk, 1e-12, 1.0)
        mk = mk / mk.sum(axis=1, keepdims=True)
        L0 = np.log(mk)

        y = y.astype(int)
        Y = np.zeros((n, 3), dtype=float)
        Y[np.arange(n), y] = 1.0

        def obj(wflat: np.ndarray) -> float:
            W = wflat.reshape(d, 3)
            L = L0 + Z @ W

            logP = L - logsumexp(L, axis=1, keepdims=True)
            nll = -float((Y * logP).sum())
            reg = 0.5 * float(self.l2) * float((W * W).sum())

            to_return = nll + reg

            return to_return

        w0 = np.zeros((d, 3), dtype=float).ravel()
        res = minimize(obj, w0, method="L-BFGS-B")
        self.W = res.x.reshape(d, 3)

        return self

    def predict_proba(self, Z: np.ndarray, mk: np.ndarray) -> np.ndarray:
        Z = self._prep_Z(Z, fit=False)
        mk = np.clip(mk, 1e-12, 1.0)
        mk = mk / mk.sum(axis=1, keepdims=True)
        L0 = np.log(mk)

        W = self.W
        L = L0 + Z @ W

        return _softmax_logits(L)

    def save(self, path: str):
        dump({
            "l2": self.l2,
            "W": self.W,
            "standardize": self.standardize,
            "z_mean": self.z_mean,
            "z_std": self.z_std
        }, path)

    @classmethod
    def load(cls, path: str) -> "MarketPriorCorrector":
        d = load(path)
        obj = cls(l2=float(d["l2"]), standardize=bool(d.get("standardize", False)))
        obj.W = d["W"]
        obj.z_mean = d.get("z_mean", None)
        obj.z_std = d.get("z_std", None)
        
        return obj