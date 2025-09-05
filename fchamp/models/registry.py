from pathlib import Path
import time
import hashlib
import json

def _hash_obj(obj) -> str:
    m = hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode())
    return m.hexdigest()[:10]

class ModelRegistry:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)
        self.base.mkdir(parents=True, exist_ok=True)

    def _latest(self) -> str | None:
        items = sorted(self.base.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

        for p in items:
            if p.is_dir() and (p / "model.joblib").exists():
                return p.name

        return None

    def model_path(self, model_id: str) -> Path:
        p = self.base / model_id
        p.mkdir(parents=True, exist_ok=True)

        return p

    def model_dir(self, model_id: str) -> Path:
        return self.base / model_id

    def create_id(self, meta: dict) -> str:
        ts = time.strftime("%Y%m%d-%H%M%S")
        return f"{ts}-{_hash_obj(meta)}"

    def save(self, model_id: str, model_file: str, meta: dict):
        mp = self.model_path(model_id)
        (mp / "meta.json").write_text(json.dumps(meta, indent=2))

        return mp / model_file

    def get_latest_id(self) -> str | None:
        return self._latest()