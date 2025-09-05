import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load as joblib_load

from fchamp.data.loader import load_matches
from fchamp.features.engineering import add_elo, add_rolling_form, build_features
from fchamp.features.market import add_market_features
from fchamp.models.registry import ModelRegistry
from fchamp.models.calibration import OneVsRestIsotonic, MultinomialLogisticCalibrator
from fchamp.evaluation.metrics import expected_calibration_error, multi_log_loss, brier_score
from sklearn.model_selection import TimeSeriesSplit

# riuso helpers dal training
from fchamp.pipelines.train import _oof_probs, _oof_gbm

def run_calibration(cfg, model_id: str | None = None, ece_target: float = 0.08) -> dict:
    """
    Ricalibra il modello corrente (lega del cfg) usando OOF predictions coerenti all'inferenza.
    - Se esiste lo stacker, calibra sui suoi output OOF.
    - Altrimenti calibra sul fallback (Poisson -> GBM blend -> market blend).
    Salva calibrator.joblib nell'artifact del modello e aggiorna meta.json.
    """
    reg = ModelRegistry(cfg.artifacts_dir)
    model_id = model_id or reg.get_latest_id()
    if not model_id:
        raise RuntimeError("Nessun modello disponibile. Esegui train prima.")

    model_dir = reg.model_dir(model_id)
    meta = json.loads((model_dir / "meta.json").read_text())

    # 1) Dati e feature come in train
    df = load_matches(cfg.data.paths, delimiter=cfg.data.delimiter)
    if getattr(cfg.data, "use_market", False):
        df = add_market_features(df, cfg.data.paths, delimiter=cfg.data.delimiter)

    elo_params = {
        'start': cfg.elo.start, 'k': cfg.elo.k, 'hfa': cfg.elo.hfa, 'mov_factor': cfg.elo.mov_factor
    }
    if hasattr(cfg.elo, 'season_regression'): elo_params['season_regression'] = cfg.elo.season_regression
    if hasattr(cfg.elo, 'time_decay_days'): elo_params['time_decay_days'] = cfg.elo.time_decay_days
    if hasattr(cfg.elo, 'adaptive_k'): elo_params['adaptive_k'] = cfg.elo.adaptive_k
    if hasattr(cfg.elo, 'home_away_split'): elo_params['home_away_split'] = cfg.elo.home_away_split
    df = add_elo(df, **elo_params)

    form_params = {'rolling_n': cfg.features.rolling_n, 'ewm_alpha': cfg.features.ewm_alpha}
    if hasattr(cfg.features, 'add_features'): form_params['add_features'] = cfg.features.add_features
    df = add_rolling_form(df, **form_params)

    X, y_out, yh, ya = build_features(df, safe_fill=getattr(cfg.features, 'safe_fill', True), include_advanced=getattr(cfg.features, 'include_advanced', True))

    # 2) OOF probabilities coerenti al modello
    alpha = float(meta.get('alpha', cfg.model.alpha))
    use_dc = bool(meta.get('use_dixon_coles', cfg.model.use_dixon_coles))
    dc_rho = float(meta.get('dc_rho', cfg.model.dc_rho))
    max_goals = int(meta.get('max_goals', cfg.features.max_goals))

    P_poiss_oof, Y_oof = _oof_probs(X, y_out, yh, ya, cfg.backtest.n_splits, cfg.backtest.gap, alpha, use_dc, dc_rho, max_goals)

    P_gbm_oof = None
    try:
        P_gbm_oof, _ = _oof_gbm(X, y_out, cfg.backtest.n_splits, cfg.backtest.gap)
    except Exception:
        P_gbm_oof = None

    # market OOF
    MK_list = []
    tss = TimeSeriesSplit(n_splits=cfg.backtest.n_splits, gap=cfg.backtest.gap)
    for _, te in tss.split(X):
        if all(col in df.columns for col in ['book_p_home','book_p_draw','book_p_away']):
            MK_list.append(df.iloc[te][['book_p_home','book_p_draw','book_p_away']].astype(float).values)
        else:
            MK_list.append(np.full((len(te),3), 1/3, dtype=float))
    MK_oof = np.vstack(MK_list)

    # 3) Pre-cal probs come in inferenza: usa stacker se esiste, altrimenti fallback blend
    pre_P = None
    stacker_path = model_dir / "stacker.joblib"
    if stacker_path.exists():
        try:
            stacker = joblib_load(str(stacker_path))
            parts = [np.log(np.clip(P_poiss_oof, 1e-9, 1.0))]
            if P_gbm_oof is not None:
                parts.append(np.log(np.clip(P_gbm_oof, 1e-9, 1.0)))
            parts.append(MK_oof)
            X_stack_oof = np.column_stack(parts)
            pre_P = stacker.predict_proba(X_stack_oof)
        except Exception:
            pre_P = None

    if pre_P is None:
        pre_P = P_poiss_oof.copy()
        gbm_weight = float(meta.get("gbm", {}).get("blend_weight", getattr(cfg.model.gbm, 'blend_weight', 0.0)))
        gbm_weight = min(max(gbm_weight, 0.0), 1.0)
        if P_gbm_oof is not None and gbm_weight > 0:
            pre_P = (1.0 - gbm_weight) * pre_P + gbm_weight * P_gbm_oof
            pre_P = np.clip(pre_P, 1e-9, 1.0)
            pre_P = pre_P / pre_P.sum(axis=1, keepdims=True)
        w = float(getattr(cfg.model, 'market_blend_weight', 0.0) or 0.0)
        w = min(max(w, 0.0), 1.0)
        if w > 0 and MK_oof is not None:
            pre_P = (1.0 - w) * pre_P + w * MK_oof
            pre_P = np.clip(pre_P, 1e-9, 1.0)
            pre_P = pre_P / pre_P.sum(axis=1, keepdims=True)

    # 4) Fit calibrators e scegli il migliore (ECE poi LogLoss)
    results = {}
    def _metrics(P):
        return {
            'ece': expected_calibration_error(Y_oof, P),
            'log_loss': multi_log_loss(Y_oof, P),
            'brier': brier_score(np.eye(3)[Y_oof], P)
        }

    # Isotonic
    iso = OneVsRestIsotonic().fit(pre_P, Y_oof)
    P_iso = iso.transform(pre_P)
    m_iso = _metrics(P_iso)

    # Multinomial (vector scaling) pro-draw
    cw = {0: 1.0, 1: float(getattr(cfg.model, 'draw_weight', 1.0)), 2: 1.0}
    mcal = MultinomialLogisticCalibrator(class_weight=cw).fit(pre_P, Y_oof)
    P_m = mcal.transform(pre_P)
    m_m = _metrics(P_m)

    # scelta per ECE (target), tie-break logloss
    cand = [('isotonic', iso, m_iso), ('multinomial', mcal, m_m)]
    cand.sort(key=lambda x: (x[2]['ece'], x[2]['log_loss']))
    best_name, best_model, best_metrics = cand[0]
    if best_metrics['ece'] > ece_target and best_name != 'multinomial':
        best_name, best_model, best_metrics = ('multinomial', mcal, m_m)

    # 5) Save
    best_model.save(str(model_dir / 'calibrator.joblib'))
    # aggiorna meta
    meta['calibration'] = {
        'calibrated': True,
        'method': best_name,
        'metrics': best_metrics
    }
    (model_dir / 'meta.json').write_text(json.dumps(meta, indent=2))

    results.update({'model_id': model_id, 'chosen': best_name, **best_metrics})
    return results


