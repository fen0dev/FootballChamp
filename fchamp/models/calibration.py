import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from joblib import dump, load

class OneVsRestIsotonic:
    def __init__(self):
        self.models = []

    def fit(self, P: np.ndarray, y: np.ndarray):
        # P shape (n,3), y in {0,1,2}
        self.models = []
        
        for k in range(P.shape[1]):
            ir = IsotonicRegression(out_of_bounds="clip")
            z = (y == k).astype(int)
            ir.fit(P[:, k], z)
        
            self.models.append(ir)
        
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        Q = np.column_stack([m.predict(P[:, k]) for k, m in enumerate(self.models)])
        Q = np.clip(Q, 1e-9, 1.0)
        Q = Q / Q.sum(axis=1, keepdims=True)
        
        return Q

    def save(self, path: str):
        dump({"models": self.models}, path)

    @classmethod
    def load(cls, path: str) -> "OneVsRestIsotonic":
        d = load(path)
        obj = cls()
        obj.models = d["models"]
        
        return obj

class MultinomialLogisticCalibrator:
    def __init__(self, class_weight: dict | None = None):
        self.model = LogisticRegression(max_iter=300, multi_class="multinomial", class_weight=class_weight)
        self.class_weight = class_weight

    def fit(self, P: np.ndarray, y: np.ndarray):
        eps = 1e-9
        X = np.log(np.clip(P, eps, 1.0))
        self.model.fit(X, y)
        return self

    def transform(self, P: np.ndarray) -> np.ndarray:
        eps = 1e-9
        X = np.log(np.clip(P, eps, 1.0))
        Q = self.model.predict_proba(X)
        Q = np.clip(Q, 1e-9, 1.0)
        Q = Q / Q.sum(axis=1, keepdims=True)
        return Q

    def save(self, path: str):
        dump({"model": self.model, "class_weight": self.class_weight}, path)

    @classmethod
    def load(cls, path: str) -> "MultinomialLogisticCalibrator":
        d = load(path)
        obj = cls(class_weight=d.get("class_weight"))
        obj.model = d["model"]
        return obj