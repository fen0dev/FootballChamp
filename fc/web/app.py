import json
import os
import pandas as pd
from copy import deepcopy
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
from fc.config import load_config, AppConfig
from fchamp.pipelines.predict import run_predict_df
import unicodedata, difflib

BASE_DIR = Path(__file__).parent
TEMPLATES = Jinja2Templates(directory=str(BASE_DIR / "templates"))
STATIC_DIR = BASE_DIR / "static"
ASSETS_DIR = BASE_DIR / "assets"

app = FastAPI(title="FootballChamp Web")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

def get_cfg() -> AppConfig:
    p = os.environ.get("FCHAMP_CONFIG", "config.yaml")

    if p and Path(p).exists():
        return load_config(p)

    for cand in ("config_ita.yaml", "config_epl.yaml", "config_dsl.yaml"):
        if Path(cand).exists():
            return load_config(cand)

    raise RuntimeError("Nessun file YAML di configurazione trovate. Imposta FCHAMP_CONFIG o rivedi i file di configurazione YAML.")

class FixtureIn(BaseModel):
    date: str
    home_team: str
    away_team: str
    psch: Optional[float] = None
    pscd: Optional[float] = None
    psca: Optional[float] = None
    b365h: Optional[float] = None
    b365d: Optional[float] = None
    b365a: Optional[float] = None

class PredictRequest(BaseModel):
    fixtures: List[FixtureIn]
    model_id: Optional[str] = None

def _scan_models(base="artifacts"):
    items = []
    basep = Path(base)
    
    if not basep.exists():
        return items
    
    for d in basep.rglob("meta.json"):
        try:
            meta = json.loads(d.read_text())
            mid = d.parent.name
            teams = set(meta.get("teams", []))
            league = meta.get("league", "unknown")
            ts = d.parent.stat().st_mtime
            items.append({"model_id": mid, "dir": str(d.parent), "teams": teams, "league": league, "mtime": ts})
    
        except Exception:
            continue
    
    items.sort(key=lambda x: x["mtime"], reverse=True)
    
    return items

def _pick_model_for_fixture(models, home, away):

    def _norm(s: str) -> str:
        x = unicodedata.normalize('NFKD', str(s)).encode('ascii','ignore').decode().lower()
        x = x.replace('.', ' ').replace('-', ' ').replace("'", ' ')
        for suf in [' fc',' afc',' cf',' bk',' if',' fk',' sk',' sc']:
            if x.endswith(suf):
                x = x[:-len(suf)]
        return ' '.join(x.split())
    
    alias = {
        "man city":"manchester city","man utd":"manchester united","man united":"manchester united",
        "nottm forest":"nottingham forest","newcastle":"newcastle united","wolves":"wolverhampton",
        "spurs":"tottenham","juve":"juventus"
    }
    
    def _key(s: str) -> str: 
        k = _norm(s); 
        return alias.get(k, k)

    h, a = _key(home), _key(away)

    # preferisci modello che contiene entrambe (con nomi normalizzati)
    for m in models:
        tnorm = { _key(t) for t in m["teams"] }
        if h in tnorm and a in tnorm:
            return m["model_id"]

    # fallback: almeno una squadra presente
    for m in models:
        tnorm = { _key(t) for t in m["teams"] }
        if h in tnorm or a in tnorm:
            return m["model_id"]
    return None

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return TEMPLATES.TemplateResponse("index.html", {"request": request})

@app.post("/api/predict")
def api_predict(payload: PredictRequest):
    if not payload.fixtures:
        raise HTTPException(400, "Nessuna partita fornita.")
    
    df = pd.DataFrame([f.model_dump() for f in payload.fixtures])
    cfg = get_cfg()

    if payload.model_id:
        out = run_predict_df(cfg, fixtures_df=df, model_id=payload.model_id)
    else:
        models = _scan_models("artifacts")
        groups = {}

        if models:
            # determina modello trovato: manda tutto e raggruppa per ridurre overhead
            for r in df.itertuples(index=False):
                mid = _pick_model_for_fixture(models, r.home_team, r.away_team)

                groups.setdefault(mid, []).append(r._asdict())
        else:
            # nessun modello trovato: manda tutto al default del config
            groups = {None: [r._asdict() for r in df.itertuples(index=False)]}

        parts = []

        for mid, rows in groups.items():
            part_df = pd.DataFrame(rows)
            local_cfg = deepcopy(cfg)

            # trove record del modello selezionato
            md = None
            if mid is not None:
                for m in models:
                    if m["model_id"] == mid:
                        md = m
                        break
            
            if mid is not None:
                model_dir = Path(md["dir"])
                meta = json.loads((model_dir / "meta.json").read_text())
                # Seleziona config per lega (assicurando coerenza dei parametri di post-processing)
                league = (meta.get("league") or "").lower()
                league_cfg_map = {
                    "ita": "config_ita.yaml",
                    "epl": "config_epl.yaml",
                    "dsl": "config_dsl.yaml",
                }
                sel = league_cfg_map.get(league)
                if sel and Path(sel).exists():
                    local_cfg = load_config(sel)
                # imposta artifacts_dir e dataset dai meta del modello
                local_cfg.artifacts_dir = str(model_dir.parent)
                data_files = meta.get("data_files")

                if data_files:
                    local_cfg.data.paths = data_files

            parts.append(run_predict_df(local_cfg, fixtures_df=part_df, model_id=mid))
        out = pd.concat(parts, axis=0, ignore_index=True) if parts else pd.DataFrame()
    
    cols = ["date","home_team","away_team","p_home","p_draw","p_away","lambda_home","lambda_away",
            "p_1x","p_12","p_x2","p_over_1_5","p_over_2_5","p_btts_yes","p_btts_no","p_home_scores","p_away_scores"]
    # include spiegazioni se presenti
    explain_cols = [c for c in [
        "explain_draw_boosted","explain_near_tie_promoted","explain_guardrails_applied","explain_market_estimated",
        "explain_favorite_gate_blocked","explain_flat_market_skipped"
    ] if c in out.columns]
    # includi anche model_id per debug/trasparenza
    cols = cols + [c for c in ["model_id"] if c in out.columns] + explain_cols
    out = out[cols].copy()
    
    for c in ["p_home","p_draw","p_away","lambda_home","lambda_away",
            "p_1x","p_12","p_x2","p_over_1_5","p_over_2_5","p_btts_yes","p_btts_no","p_home_scores","p_away_scores"]:
        
        out[c] = out[c].astype(float).round(3)

    # serializza la data come stringa ISO (evita Timestamp non serializzabile)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime('%Y-%m-%d')

    return JSONResponse(out.to_dict(orient="records"))

def run():
    import uvicorn
    uvicorn.run("fc.web.app:app", host="127.0.0.1", port=8000, reload=True)
