## FootballChamp

### Panoramica
FootballChamp è un motore di previsione per partite di calcio con:
- Modello Poisson avanzato (con correzione Dixon–Coles, ensemble opzionale GBM)
- Stacking OOF + calibrazione (isotonic/multinomial logistic)
- Integrazione dati mercato, xG reali (Understat scraping), tiri (shots) e feature avanzate
- Post‑processing per realismo: draw booster, meta “Draw vs No‑Draw”, market guardrails, favorite‑gate
- CLI, API (FastAPI) e UI web

Software con licenza proprietaria. Tutti i diritti riservati (vedi LICENSE).

### Struttura del progetto
- `fchamp/`:
  - `pipelines/` (train, predict)
  - `features/` (ELO, rolling form, advanced, market)
  - `models/` (Poisson, calibrators)
  - `evaluation/` (backtest)
  - `data/` (loader + web scraping xG/shots)
- `fc/`:
  - `cli.py` (comandi CLI)
  - `web/` (FastAPI + template UI)
  - `config.py` (schema Pydantic)
- `config_*.yaml` (config per lega)
- `artifacts/` (modelli, xg/shots CSV)
- `pyproject.toml`

### Requisiti
- Python 3.13
- Ambiente virtuale (in repo: `calcio/`) oppure crearne uno nuovo.

### Installazione
1) Attiva il venv incluso:
```bash
source calcio/bin/activate
```
2) (Opzionale) Installa in editable:
```bash
pip install -e .
```

### Dati
- Storici match CSV: es. `fchamp/data/epl_2015_2026.csv`, `fchamp/data/serie_a_2015_2025.csv`
- xG reali (Understat): salva in `artifacts/xg_{lega}.csv`
- Shots reali (Understat): salva in `artifacts/shots_{lega}.csv`

Scraping xG:
```bash
fchamp -c config_epl.yaml fetch-xg-scrape --league "Premier League" --seasons 2023,2024 -o artifacts/xg_epl.csv
```
Scraping shots:
```bash
fchamp -c config_epl.yaml fetch-shots-scrape --league "Premier League" --seasons 2023,2024 -o artifacts/shots_epl.csv --progress
```

### Configurazione (YAML)
Esempio (estratto):
```yaml
data:
  paths:
    - fchamp/data/epl_2015_2026.csv
  xg_path: artifacts/xg_epl.csv
  shots_path: artifacts/shots_epl.csv
  use_market: true
  delimiter: ","

features:
  rolling_n: 6
  ewm_alpha: 0.5
  use_advanced_stats: true
  use_xg_real: true
  use_h2h: true
  h2h_matches: 6

model:
  alpha: 0.8
  use_dixon_coles: true
  gbm:
    enabled: true
    blend_weight: 0.7
  calibration:
    enabled: true
    method: isotonic
  market_blend_weight: 0.45
  market_guardrails:
    enabled: true
    max_abs_diff_home: 0.14
    max_abs_diff_draw: 0.18
    max_abs_diff_away: 0.18
    blend_weight: 0.45
  draw_meta:
    enabled: true
    blend_weight: 0.70
  draw_booster:
    enabled: true
    elo_abs_diff_max: 7.0
    goals_ewm_sum_max: 2.15
    market_draw_min: 0.21
    weight: 0.65
    max_boost: 0.04
    promote_near_tie: true
    tie_margin: 0.17
    skip_booster_if_favorite: true
    favorite_prob_min: 0.62

backtest:
  n_splits: 5
  gap: 1
  tune: true
  tune_trials: 120
```

### Addestramento
```bash
fchamp -c config_ita.yaml train
fchamp -c config_epl.yaml train
```

### Backtest
```bash
fchamp -c config_ita.yaml backtest
fchamp -c config_epl.yaml backtest
```

### Predizione (CLI)
```bash
fchamp -c config_ita.yaml predict --fixtures fixtures.csv -o preds.csv
```
`fixtures.csv`: colonne minime `date,home_team,away_team` (+ opzionali PSCH/PSCD/PSCA).

### API e UI Web
Avvio server:
```bash
uvicorn fc.web.app:app --host 127.0.0.1 --port 8000
```
- POST `/api/predict` body:
```json
{
  "fixtures": [
    {"date":"2025-09-13","home_team":"Juventus","away_team":"Inter"}
  ]
}
```
UI: apri `http://127.0.0.1:8000/`

Suggerimento: puoi impostare `FCHAMP_CONFIG` per cambiare config default caricata dall’API.

### Note sui meccanismi di realismo
- Draw booster: aumenta X in contesti equilibrati (elo basso, pochi gol attesi, mercato pro‑X)
- Meta Draw vs No‑Draw: blending della X con un meta‐modello binario
- Market guardrails: limita scostamenti estremi dalle probabilità di mercato
- Favorite gate: non promuove X se c’è un favorito forte

### Troubleshooting
- Predizioni uniformi 1/3–1/3–1/3: fornire quote reali o abilitare stima mercato; verificare alias team
- 0–3–97 con λ minime: assicurarsi che sia attivo il clamp λ e che le quote stimate non guidino stacker/guardrails
- UI non aggiorna: riavvia server e hard refresh (Cmd+Shift+R)

### Scripting xG/shots
- xG: `fetch-xg-scrape` analizza la pagina Understat (blocchi JSON)
- Shots: `fetch-shots-scrape` visita match pages (rispetta i limiti con `--limit` e `--progress`)

### Licenza
Questo software è distribuito con licenza proprietaria. Vedi `LICENSE` per termini e restrizioni.

### Contatti
Per supporto interno/integrazioni, fare riferimento al responsabile del progetto o al team ML interno.
