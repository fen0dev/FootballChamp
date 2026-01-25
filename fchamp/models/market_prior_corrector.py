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
    W: np.ndarray | None = None     # shape (d,3)

    def fit(self, Z: np.ndarray, mk: np.ndarray, y: np.ndarray):
        """
            logit(P) = log(mk) + Z @ W
            Z: (n,d), mk:(n,3), y in {0,1,2}
        """

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
        mk = np.clip(mk, 1e-12, 1.0)
        mk = mk / mk.sum(axis=1, keepdims=True)
        L0 = np.log(mk)

        W = self.W
        L = L0 + Z @ W

        return _softmax_logits(L)

    def save(self, path: str):
        dump({ "l2": self.l2, "W": self.W }, path)

    @classmethod
    def load(cls, path: str) -> "MarketPriorCorrector":
        d = load(path)
        obj = cls(l2=float(d["l2"]))
        obj.W = d["W"]
        
        return obj