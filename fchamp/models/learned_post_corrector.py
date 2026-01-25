import numpy as np
from dataclasses import dataclass
from joblib import load, dump
from scipy.optimize import minimize
from scipy.special import logsumexp

def _softmax(L: np.ndarray) -> np.ndarray:
    Z = L - logsumexp(L, axis=1, keepdims=True)
    P = np.exp(Z)

    res = np.clip(P, 1e-12, 1.0) / np.clip(P.sum(axis=1, keepdims=True), 1e-12, np.inf)

    return res

@dataclass
class LearnedPostCorrector:
    l2: float = 1.0
    W: np.ndarray | None = None     # (d, 3)

    def fit(self, F: np.ndarray, y: np.ndarray):
        """
            Multinomial LR: logit(P) = F@W
            F: (n,d) includes already log(P_base), log(mk), context feature
        """

        n, d = F.shape
        y = y.astype(int)
        Y = np.zeros((n, 3), dtype=float)
        Y[np.arange(n), y] = 1.0

        def obj(wflat):
            W = wflat.reshape(d, 3)
            L = F @ W
            logP = L - logsumexp(L, axis=1, keepdims=True)
            nll = -float((Y * logP).sum())
            reg = 0.5 * float(self.l2) * float((W * W).sum())
            res = nll + reg
            return res

        w0 = np.zeros((d, 3), dtype=float).ravel()
        res = minimize(obj, w0, method="L-BFGS-B")
        self.W = res.x.reshape(d, 3)

        return self


    def predict_proba(self, F: np.ndarray) -> np.ndarray:
        return _softmax(F @ self.W)

    def save(self, path: str):
        dump({ "l2": self.l2, "W": self.W }, path)

    @classmethod
    def load(cls, path: str) -> "LearnedPostCorrector":
        d = load(path)
        obj = cls(l2=float(d["l2"]))
        obj.W = d["W"]

        return obj

        