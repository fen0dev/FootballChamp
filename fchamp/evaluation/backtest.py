import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from fchamp.data.loader import load_matches
from fchamp.features.engineering import add_elo, add_rolling_form, build_features
from fchamp.models.goals_poisson import GoalsPoissonModel
from fchamp.features.market import add_market_features
from fchamp.evaluation.metrics import multi_log_loss, brier_score
import logging
from typing import Dict, List
from datetime import datetime
#from fchamp.models.registry import ModelRegistry
from fchamp.models.calibration import OneVsRestIsotonic, MultinomialLogisticCalibrator
from joblib import load as joblib_load

logger = logging.getLogger(__name__)

def _calculate_betting_metrics(y_true: np.ndarray, y_prob: np.ndarray, market_probs: np.ndarray = None) -> Dict[str, float]:
    """🚀 ENHANCED betting metrics per valutazione realistica"""
    metrics = {}
    
    # Predizioni discrete
    y_pred = np.argmax(y_prob, axis=1)
    
    # Accuracy base
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # Precision/Recall per classe
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    
    for i, outcome in enumerate(['home_win', 'draw', 'away_win']):
        metrics[f'precision_{outcome}'] = float(precision[i]) if i < len(precision) else 0.0
        metrics[f'recall_{outcome}'] = float(recall[i]) if i < len(recall) else 0.0
        metrics[f'f1_{outcome}'] = float(f1[i]) if i < len(f1) else 0.0
    
    # 🚀 BETTING SIMULATION (se abbiamo quote market)
    if market_probs is not None:
        # Trova dove il nostro modello è più confident del mercato
        model_confidence = np.max(y_prob, axis=1)
        market_confidence = np.max(market_probs, axis=1)
        
        # Bet quando siamo più confident del mercato
        confident_bets = model_confidence > market_confidence
        
        if confident_bets.sum() > 0:
            # ROI simulation (semplificato)
            correct_confident_bets = (y_pred[confident_bets] == y_true[confident_bets]).sum()
            metrics['confident_bets_count'] = int(confident_bets.sum())
            metrics['confident_bets_accuracy'] = float(correct_confident_bets / confident_bets.sum())
            
            # Estimated ROI (approssimazione)
            avg_odds = 1 / np.mean(market_probs[confident_bets], axis=0)
            estimated_roi = (correct_confident_bets * np.mean(avg_odds) - confident_bets.sum()) / confident_bets.sum()
            metrics['estimated_roi'] = float(estimated_roi)
        else:
            metrics.update({
                'confident_bets_count': 0,
                'confident_bets_accuracy': 0.0,
                'estimated_roi': 0.0
            })
    
    return metrics

def _calculate_calibration_metrics(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, float]:
    """🚀 ENHANCED calibration analysis"""
    metrics = {}
    
    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Per ogni classe
        for class_idx in range(3):
            # Trova predizioni in questo bin per questa classe
            in_bin = (y_prob[:, class_idx] > bin_lower) & (y_prob[:, class_idx] <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuratezza in questo bin
                accuracy_in_bin = (y_true[in_bin] == class_idx).mean()
                # Confidence media in questo bin
                avg_confidence_in_bin = y_prob[in_bin, class_idx].mean()
                # ECE contribution
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    metrics['expected_calibration_error'] = float(ece)
    
    # Reliability diagrams data (per plotting futuro)
    reliability_data = {}
    for class_idx, class_name in enumerate(['home_win', 'draw', 'away_win']):
        bin_acc, bin_conf, bin_count = [], [], []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob[:, class_idx] > bin_lower) & (y_prob[:, class_idx] <= bin_upper)
            if in_bin.sum() > 0:
                bin_acc.append((y_true[in_bin] == class_idx).mean())
                bin_conf.append(y_prob[in_bin, class_idx].mean())
                bin_count.append(in_bin.sum())
            else:
                bin_acc.append(0.0)
                bin_conf.append((bin_lower + bin_upper) / 2)
                bin_count.append(0)
        
        reliability_data[class_name] = {
            'accuracy': bin_acc,
            'confidence': bin_conf,
            'count': bin_count
        }
    
    metrics['reliability_data'] = reliability_data
    
    return metrics

def _apply_draw_booster_vectorized(P: np.ndarray,
                                   elo_abs_diff: np.ndarray,
                                   goals_ewm_sum: np.ndarray,
                                   market_draw: np.ndarray | None,
                                   elo_thr: float,
                                   goals_thr: float,
                                   market_min: float,
                                   weight: float,
                                   max_boost: float) -> np.ndarray:
    """
    Applica il draw-booster in modo vettorizzato su un insieme di probabilità P (n,3).
    Restituisce una nuova matrice di probabilità rinormalizzata.
    """
    if P.size == 0:
        return P

    P_new = P.astype(float).copy()
    ph = P_new[:, 0]
    px = P_new[:, 1]
    pa = P_new[:, 2]

    if market_draw is None:
        market_draw = np.full(px.shape, 1/3, dtype=float)

    mask = (
        (elo_abs_diff <= elo_thr) &
        (goals_ewm_sum <= goals_thr) &
        (market_draw >= market_min)
    )

    if np.any(mask):
        target_px = np.maximum(market_draw[mask], px[mask])
        delta = weight * (target_px - px[mask])
        delta = np.minimum(delta, max_boost)
        boosted_px = px[mask] + delta
        rem = 1.0 - boosted_px
        denom = np.maximum(ph[mask] + pa[mask], 1e-9)
        scale = np.where(rem > 0, rem / denom, 1.0)
        ph[mask] = ph[mask] * scale
        pa[mask] = pa[mask] * scale
        px[mask] = boosted_px

        s = ph + px + pa
        s = np.where(s > 0, s, 1.0)
        P_new[:, 0] = ph / s
        P_new[:, 1] = px / s
        P_new[:, 2] = pa / s

    return np.clip(P_new, 1e-12, 1.0)

def run_backtest(cfg) -> dict:
    """🚀 ENHANCED backtest con metriche comprehensive"""
    logger.info("🚀 Starting enhanced backtest")
    
    # 1. DATA LOADING (stesso di prima)
    df = load_matches(cfg.data.paths, delimiter=cfg.data.delimiter)
    logger.info(f"Loaded {len(df)} matches for backtest")
    
    if getattr(cfg.data, "use_market", False):
        df = add_market_features(df, cfg.data.paths, delimiter=cfg.data.delimiter)
        logger.info("Added market features for betting analysis")
    
    # ELO con parametri avanzati (coerente con train)
    elo_params = {
        'start': cfg.elo.start,
        'k': cfg.elo.k,
        'hfa': cfg.elo.hfa,
        'mov_factor': cfg.elo.mov_factor,
    }

    if hasattr(cfg.elo, 'season_regression'):
        elo_params['season_regression'] = cfg.elo.season_regression
    if hasattr(cfg.elo, 'time_decay_days'):
        elo_params['time_decay_days'] = cfg.elo.time_decay_days
    if hasattr(cfg.elo, 'adaptive_k'):
        elo_params['adaptive_k'] = cfg.elo.adaptive_k
    if hasattr(cfg.elo, 'home_away_split'):
        elo_params['home_away_split'] = cfg.elo.home_away_split

    df = add_elo(df, **elo_params)

    # Rolling form con flag avanzati
    form_params = {
        'rolling_n': cfg.features.rolling_n,
        'ewm_alpha': cfg.features.ewm_alpha,
    }

    if hasattr(cfg.features, 'add_features'):
        form_params['add_features'] = cfg.features.add_features
    df = add_rolling_form(df, **form_params)

    # Feature building con safe_fill/include_advanced
    feature_params = {}
    
    if hasattr(cfg.features, 'safe_fill'):
        feature_params['safe_fill'] = cfg.features.safe_fill
    include_adv = getattr(cfg.features, 'include_advanced', getattr(cfg.features, 'add_features', False))
    feature_params['include_advanced'] = bool(include_adv)

    X, y_out, yh, ya = build_features(df, **feature_params)
    
    logger.info(f"Built features: {X.shape[1]} features for {X.shape[0]} samples")

    # 2. 🚀 ENHANCED CROSS-VALIDATION
    tss = TimeSeriesSplit(n_splits=cfg.backtest.n_splits, gap=cfg.backtest.gap)
    
    # Metrics storage
    fold_metrics = []
    ll_list, br_list = [], []
    all_predictions, all_actuals = [], []
    all_market_probs = []
    tuned_params_fold: list[dict] = []
    
    logger.info(f"Running {cfg.backtest.n_splits}-fold time series cross-validation")
    
    for fold_idx, (tr, te) in enumerate(tss.split(X)):
        logger.info(f"🚀 Processing fold {fold_idx + 1}/{cfg.backtest.n_splits}")
        
        # Train model
        gm = GoalsPoissonModel(alpha=cfg.model.alpha, use_dixon_coles=cfg.model.use_dixon_coles, dc_rho=cfg.model.dc_rho)
        gm.fit(X.iloc[tr], yh.iloc[tr], ya.iloc[tr])
        
        # Predictions
        lh, la = gm.predict_lambdas(X.iloc[te])

        probs = []
        for lhi, lai in zip(lh, la):
            p1, px, p2 = gm.outcome_probs(lhi, lai, cfg.features.max_goals)
            probs.append([p1, px, p2])

        P = np.array(probs)

        # Calibratore per-fold: fit su train fold, apply su test fold
        try:
            y_tr = y_out.iloc[tr].values
            lh_tr, la_tr = gm.predict_lambdas(X.iloc[tr])
            probs_tr = np.array([
                list(gm.outcome_probs(lhi, lai, cfg.features.max_goals))
                for lhi, lai in zip(lh_tr, la_tr)
            ])
            
            def _fit_best_cal(P_tr: np.ndarray, y_tr: np.ndarray):
                cal1 = OneVsRestIsotonic().fit(P_tr, y_tr)
                e1 = _calculate_calibration_metrics(y_tr, cal1.transform(P_tr))['expected_calibration_error']
                cal2 = MultinomialLogisticCalibrator(class_weight='balanced').fit(P_tr, y_tr)
                e2 = _calculate_calibration_metrics(y_tr, cal2.transform(P_tr))['expected_calibration_error']

                return cal1 if e1 <= e2 else cal2

            cal = _fit_best_cal(probs_tr, y_tr)
            P = cal.transform(P)

            try:
                y_tr = y_out.iloc[tr].values
                lh_tr, la_tr = gm.predict_lambdas(X.iloc[tr])
                P_tr = np.array([gm.outcome_probs(lhi, lai, cfg.features.max_goals) for lhi, lai in zip(lh_tr, la_tr)])

                if all(c in df.columns for c in ['book_p_home','book_p_draw','book_p_away']):
                    MK_tr = df.iloc[tr][['book_p_home','book_p_draw','book_p_away']].astype(float).values
                else:
                    MK_tr = np.full((len(tr), 3), 1/3, dtype=float)

                X_draw_tr = np.column_stack([np.log(np.clip(P_tr, 1e-9, 1.0)), MK_tr])
                y_draw_tr = (y_tr == 1).astype(int)

                dm = LogisticRegression(max_iter=300, class_weight={0:1.0, 1: float(getattr(cfg.model, 'draw_weight', 1.0))})
                dm.fit(X_draw_tr, y_draw_tr)

                #test fold
                P_te_raw = np.array(probs)
                if all(c in df.columns for c in ['book_p_home','book_p_draw','book_p_away']):
                    MK_te = df.iloc[te][['book_p_home','book_p_draw','book_p_away']].astype(float).values
                else:
                    MK_te = np.full((len(te), 3), 1/3, dtype=float)

                elo_te = np.abs(X.iloc[te]['elo_diff'].values)[:, None]
                gsum_te = (X.iloc[te]['home_gf_ewm'].values + X.iloc[te]['away_gf_ewm'].values)[:, None]

                X_draw_te = np.column_stack([np.log(np.clip(P_te_raw, 1e-9, 1.0)), MK_te, elo_te, gsum_te])
                p_draw_hat = dm.predict_proba(X_draw_te)[:, 1]
                # tuning leggero blend_weight
                from sklearn.metrics import f1_score
                weights = [0.30,0.35,0.40,0.45,0.50,0.55,0.60]
                best = (1e9, -1.0, float(getattr(getattr(cfg.model, 'draw_meta', None), 'blend_weight', 0.4)))
                Y = y_out.iloc[te].values
                for w in weights:
                    Pw = P.copy()
                    for i in range(Pw.shape[0]):
                        ph, px, pa = float(P[i,0]), float(P[i,1]), float(P[i,2])
                        r_h = ph / max(ph + pa, 1e-9)
                        px_new = (1.0 - w) * px + w * float(p_draw_hat[i])
                        ph_new = (1.0 - px_new) * r_h
                        pa_new = 1.0 - px_new - ph_new
                        Pw[i,0], Pw[i,1], Pw[i,2] = ph_new, px_new, pa_new
                    ll_try = multi_log_loss(Y, Pw)
                    y_pred = np.argmax(Pw, axis=1)
                    f1d = f1_score(Y, y_pred, labels=[1], average='macro', zero_division=0)
                    if (ll_try < best[0]) or (abs(ll_try - best[0]) < 1e-6 and f1d > best[1]):
                        best = (ll_try, f1d, w)
                bw = best[2]
                for i in range(P.shape[0]):
                    ph, px, pa = float(P[i,0]), float(P[i,1]), float(P[i,2])
                    r_h = ph / max(ph + pa, 1e-9)
                    px_new = (1.0 - bw) * px + bw * float(p_draw_hat[i])
                    ph_new = (1.0 - px_new) * r_h
                    pa_new = 1.0 - px_new - ph_new
                    P[i,0], P[i,1], P[i,2] = ph_new, px_new, pa_new
            except Exception:
                pass

        except Exception:
            pass

        # 🔧 Optional draw booster (coerente con inferenza)
        booster = getattr(getattr(cfg, "model", None), "draw_booster", None)
        apply_booster = bool(getattr(booster, "enabled", False))
        if apply_booster and P.size:
            X_test = X.iloc[te]
            # serie opzionale dal mercato
            draw_series = None
            if 'book_p_draw' in df.columns:
                try:
                    draw_series = df.iloc[te]['book_p_draw'].astype(float).clip(0, 1).fillna(1/3).values
                except Exception:
                    draw_series = None

            elo_thr = float(getattr(booster, "elo_abs_diff_max", 35.0))
            goals_thr = float(getattr(booster, "goals_ewm_sum_max", 2.6))
            market_min = float(getattr(booster, "market_draw_min", 0.28))
            weight = float(getattr(booster, "weight", 0.25))
            max_boost = float(getattr(booster, "max_boost", 0.08))

            for i in range(P.shape[0]):
                ph, px, pa = float(P[i,0]), float(P[i,1]), float(P[i,2])

                elo_abs_diff = abs(float(X_test.iloc[i].get('elo_diff', 0.0))) if hasattr(X_test, 'iloc') else 0.0
                goals_ewm_sum = float(X_test.iloc[i].get('home_gf_ewm', 1.3)) + float(X_test.iloc[i].get('away_gf_ewm', 1.1)) if hasattr(X_test, 'iloc') else 2.4
                market_draw = float(draw_series[i]) if draw_series is not None and i < len(draw_series) else 1/3

                # gate su favorito di mercato
                fav_min = float(getattr(booster, 'favorite_prob_min', 0.60))
                skip_boost_if_fav = bool(getattr(booster, 'skip_booster_if_favorite', True))
                mh = float(df.iloc[te[i]].get('book_p_home', 1/3)) if 'book_p_home' in df.columns else 1/3
                ma = float(df.iloc[te[i]].get('book_p_away', 1/3)) if 'book_p_away' in df.columns else 1/3
                is_strong_fav = max(mh, ma) >= fav_min

                if (elo_abs_diff <= elo_thr and goals_ewm_sum <= goals_thr and market_draw >= market_min) and not (skip_boost_if_fav and is_strong_fav):
                    target_px = max(market_draw, px)
                    boosted_px = px + min(max_boost, weight * (target_px - px))
                    rem = 1.0 - boosted_px
                    if rem > 0:
                        scale = rem / max(ph + pa, 1e-9)
                        ph *= scale
                        pa *= scale
                        px = boosted_px
                    s = ph + px + pa
                    if s > 0:
                        P[i,0], P[i,1], P[i,2] = ph/s, px/s, pa/s

                # Near-tie promotion to draw (coerente con predict): valuta su P corrente
                if bool(getattr(booster, 'promote_near_tie', False)):
                    tie_margin = float(getattr(booster, 'tie_margin', 0.02))
                    pcur = P[i].astype(float)
                    top_idx = int(np.argmax(pcur))
                    if top_idx != 1:
                        diff = float(pcur[top_idx] - pcur[1])
                        # gate su favorito
                        fav_min = float(getattr(booster, 'favorite_prob_min', 0.60))
                        skip_nt_if_fav = bool(getattr(booster, 'skip_near_tie_if_favorite', True))
                        is_strong_fav = max(mh, ma) >= fav_min
                        if diff <= tie_margin and not (skip_nt_if_fav and is_strong_fav):
                            new_px = float(pcur[top_idx])
                            rem = 1.0 - new_px
                            hp = max(float(pcur[0]), 1e-12)
                            ap = max(float(pcur[2]), 1e-12)
                            if rem > 0 and (hp + ap) > 0:
                                scale = rem / (hp + ap)
                                hp *= scale
                                ap *= scale
                            s = hp + new_px + ap
                            if s > 0:
                                P[i,0], P[i,1], P[i,2] = hp/s, new_px/s, ap/s
                                
        # Market guardrails (coerenti con predict)
        mg = getattr(getattr(cfg, 'model', None), 'market_guardrails', None)
        if mg and bool(getattr(mg, 'enabled', False)) and all(c in df.columns for c in ['book_p_home','book_p_draw','book_p_away']):
            try:
                mk = df.iloc[te][['book_p_home','book_p_draw','book_p_away']].astype(float).values
                max_dh = float(getattr(mg, 'max_abs_diff_home', 0.18))
                max_dx = float(getattr(mg, 'max_abs_diff_draw', 0.14))
                max_da = float(getattr(mg, 'max_abs_diff_away', 0.18))
                bw = float(getattr(mg, 'blend_weight', 0.5))
                for i in range(P.shape[0]):
                    ph, px, pa = float(P[i,0]), float(P[i,1]), float(P[i,2])
                    mh, mx, ma = float(mk[i,0]), float(mk[i,1]), float(mk[i,2])
                    # skip se quote piatte (~1/3)
                    flat = abs(mh - 1.0/3.0) + abs(mx - 1.0/3.0) + abs(ma - 1.0/3.0) < 1e-6
                    if not flat:
                        ph = np.clip(ph, mh - max_dh, mh + max_dh)
                        px = np.clip(px, mx - max_dx, mx + max_dx)
                        pa = np.clip(pa, ma - max_da, ma + max_da)
                        s = ph + px + pa
                        if s > 0:
                            ph, px, pa = ph/s, px/s, pa/s
                        if bw > 0:
                            ph = (1.0 - bw) * ph + bw * mh
                            px = (1.0 - bw) * px + bw * mx
                            pa = (1.0 - bw) * pa + bw * ma
                            s = ph + px + pa
                            if s > 0:
                                ph, px, pa = ph/s, px/s, pa/s
                    P[i,0], P[i,1], P[i,2] = ph, px, pa
            except Exception:
                pass

        Y = y_out.iloc[te].values
        onehot = np.eye(3)[Y]

        # Optional tuning del draw_booster (se abilitato in cfg.backtest.tune)
        if getattr(cfg.backtest, 'tune', False):
            booster_cfg = getattr(cfg.model, 'draw_booster', None)
            if booster_cfg and getattr(booster_cfg, 'enabled', False):
                # feature per booster
                X_test = X.iloc[te]
                elo_abs = np.abs(X_test.get('elo_diff', pd.Series(np.zeros(len(te))))).values
                goals_sum = (X_test.get('home_gf_ewm', pd.Series(np.ones(len(te))*1.3)).values +
                             X_test.get('away_gf_ewm', pd.Series(np.ones(len(te))*1.1)).values)
                market_draw = None
                if all(c in df.columns for c in ['book_p_draw']):
                    try:
                        market_draw = df.iloc[te]['book_p_draw'].astype(float).clip(0,1).fillna(1/3).values
                    except Exception:
                        market_draw = None

                # griglia semplice; per velocità si limita il numero combinazioni
                elo_grid = [booster_cfg.elo_abs_diff_max*0.7, booster_cfg.elo_abs_diff_max, booster_cfg.elo_abs_diff_max*1.3]
                goals_grid = [booster_cfg.goals_ewm_sum_max*0.8, booster_cfg.goals_ewm_sum_max, booster_cfg.goals_ewm_sum_max*1.2]
                market_grid = [max(0.22, booster_cfg.market_draw_min*0.9), booster_cfg.market_draw_min, min(0.36, booster_cfg.market_draw_min*1.2)]
                weight_grid = [max(0.1, booster_cfg.weight*0.6), booster_cfg.weight, min(0.5, booster_cfg.weight*1.4)]
                maxb_grid = [max(0.04, booster_cfg.max_boost*0.6), booster_cfg.max_boost, min(0.15, booster_cfg.max_boost*1.4)]

                best_ll = np.inf
                best_params = (booster_cfg.elo_abs_diff_max, booster_cfg.goals_ewm_sum_max,
                                booster_cfg.market_draw_min, booster_cfg.weight, booster_cfg.max_boost)

                max_trials = int(getattr(cfg.backtest, 'tune_trials', 30))
                tried = 0
                for e in elo_grid:
                    for g in goals_grid:
                        for m in market_grid:
                            for w in weight_grid:
                                for mb in maxb_grid:
                                    if tried >= max_trials:
                                        break
                                    P_try = _apply_draw_booster_vectorized(P, elo_abs, goals_sum, market_draw, e, g, m, w, mb)
                                    ll_try = multi_log_loss(Y, P_try)
                                    if ll_try < best_ll:
                                        best_ll = ll_try
                                        best_params = (e, g, m, w, mb)
                                    tried += 1
                                if tried >= max_trials:
                                    break
                            if tried >= max_trials:
                                break
                        if tried >= max_trials:
                            break
                    if tried >= max_trials:
                        break

                # applica migliore configurazione trovata per questo fold
                e, g, m, w, mb = best_params
                P = _apply_draw_booster_vectorized(P, elo_abs, goals_sum, market_draw, e, g, m, w, mb)

                # salva best params per questo fold
                tuned_params_fold.append({
                    'fold': int(fold_idx + 1),
                    'elo_abs_diff_max': float(e),
                    'goals_ewm_sum_max': float(g),
                    'market_draw_min': float(m),
                    'weight': float(w),
                    'max_boost': float(mb),
                    'log_loss': float(best_ll),
                })

        # Basic metrics
        ll = multi_log_loss(Y, P)
        br = brier_score(onehot, P)
        ll_list.append(ll)
        br_list.append(br)
        
        # Store for aggregate analysis
        all_predictions.append(P)
        all_actuals.append(Y)
        
        # 🚀 MARKET COMPARISON (se disponibile)
        market_probs = None
        if 'book_p_home' in df.columns:
            # Extract market probabilities per test set
            test_df = df.iloc[te]
            if all(col in test_df.columns for col in ['book_p_home', 'book_p_draw', 'book_p_away']):
                market_probs = test_df[['book_p_home', 'book_p_draw', 'book_p_away']].values
                all_market_probs.append(market_probs)
        
        # 🚀 FOLD-SPECIFIC METRICS
        fold_betting_metrics = _calculate_betting_metrics(Y, P, market_probs)
        fold_calibration_metrics = _calculate_calibration_metrics(Y, P)
        
        fold_metric = {
            'fold': fold_idx + 1,
            'train_samples': len(tr),
            'test_samples': len(te),
            'log_loss': float(ll),
            'brier_score': float(br),
            'test_period': {
                'start': df.iloc[te]['date'].min().isoformat() if 'date' in df.columns else None,
                'end': df.iloc[te]['date'].max().isoformat() if 'date' in df.columns else None
            }
        }
        fold_metric.update(fold_betting_metrics)
        fold_metric.update(fold_calibration_metrics)
        
        fold_metrics.append(fold_metric)
        
        logger.info(f"   Fold {fold_idx + 1} - Log Loss: {ll:.4f}, Brier: {br:.4f}, Accuracy: {fold_betting_metrics.get('accuracy', 0):.3f}")

    # 3. 🚀 AGGREGATE ANALYSIS
    logger.info("🚀 Computing aggregate metrics")
    
    # Combine all predictions
    all_P = np.vstack(all_predictions)
    all_Y = np.concatenate(all_actuals)
    all_market_P = np.vstack(all_market_probs) if all_market_probs else None
    
    # Overall metrics
    overall_betting = _calculate_betting_metrics(all_Y, all_P, all_market_P)
    overall_calibration = _calculate_calibration_metrics(all_Y, all_P)
    
    # 🚀 TEMPORAL ANALYSIS (corretta)
    temporal_metrics = {}
    if 'date' in df.columns:
        # Crea DataFrame solo con i campioni di test
        test_indices = []
        for tr, te in tss.split(X):
            test_indices.extend(te)
        
        test_df = df.iloc[test_indices].copy()
        test_df['model_pred'] = np.argmax(all_P, axis=1)
        test_df['actual'] = all_Y
        test_df['year'] = pd.to_datetime(test_df['date']).dt.year
        
        yearly_performance = {}
        for year in test_df['year'].unique():
            year_mask = test_df['year'] == year
            if year_mask.sum() > 0:
                year_acc = (test_df.loc[year_mask, 'model_pred'] == test_df.loc[year_mask, 'actual']).mean()
                yearly_performance[int(year)] = float(year_acc)
        
        temporal_metrics['yearly_accuracy'] = yearly_performance

    # 4. 🚀 COMPREHENSIVE RESULTS
    results = {
        # Basic metrics (backward compatible)
        "log_loss_1x2": float(np.mean(ll_list)),
        "brier_1x2": float(np.mean(br_list)),
        "splits": cfg.backtest.n_splits,
        
        # 🚀 ENHANCED METRICS
        "backtest_timestamp": datetime.now().isoformat(),
        "total_samples": len(all_Y),
        "config_summary": {
            "elo": cfg.elo.__dict__,
            "features": cfg.features.__dict__,
            "model": cfg.model.__dict__
        },
        
        # Statistical metrics
        "log_loss_std": float(np.std(ll_list)),
        "brier_std": float(np.std(br_list)),
        "log_loss_min": float(np.min(ll_list)),
        "log_loss_max": float(np.max(ll_list)),
        
        # Betting & calibration
        **overall_betting,
        **overall_calibration,
        **temporal_metrics,
        
        # Fold-by-fold details
        "fold_details": fold_metrics,
        
        # 🚀 PERFORMANCE SUMMARY
        "performance_grade": _grade_performance(np.mean(ll_list), overall_betting.get('accuracy', 0)),
        "key_insights": _generate_insights(fold_metrics, overall_betting, overall_calibration)
    }

    # Se tuning attivo, raccogliamo i best params per fold
    if getattr(cfg.backtest, 'tune', False):
        # nota: per semplicità non abbiamo tracciato fold-wise i best params; qui ritorniamo i default usati
        # e suggeriamo di leggere i log per i valori per fold. In alternativa si può salvare durante il loop.
        booster_cfg = getattr(cfg.model, 'draw_booster', None)
        if booster_cfg and getattr(booster_cfg, 'enabled', False):
            results["draw_booster_used"] = True
            results["draw_booster_base_params"] = {
                "elo_abs_diff_max": float(booster_cfg.elo_abs_diff_max),
                "goals_ewm_sum_max": float(booster_cfg.goals_ewm_sum_max),
                "market_draw_min": float(booster_cfg.market_draw_min),
                "weight": float(booster_cfg.weight),
                "max_boost": float(booster_cfg.max_boost),
            }
            if tuned_params_fold:
                import numpy as _np
                results["tuned_draw_booster_params_per_fold"] = tuned_params_fold
                # suggerimento: mediana robusta tra i fold
                def _median_param(name: str) -> float:
                    return float(_np.median([d[name] for d in tuned_params_fold]))
                results["tuned_draw_booster_params_suggested"] = {
                    "elo_abs_diff_max": _median_param('elo_abs_diff_max'),
                    "goals_ewm_sum_max": _median_param('goals_ewm_sum_max'),
                    "market_draw_min": _median_param('market_draw_min'),
                    "weight": _median_param('weight'),
                    "max_boost": _median_param('max_boost'),
                }
    
    logger.info("🚀 Backtest completed successfully")
    logger.info(f"   Overall Log Loss: {results['log_loss_1x2']:.4f}")
    logger.info(f"   Overall Accuracy: {overall_betting.get('accuracy', 0):.3f}")
    logger.info(f"   Performance Grade: {results['performance_grade']}")
    
    return results

def _grade_performance(log_loss: float, accuracy: float) -> str:
    """🚀 Grade model performance"""
    if log_loss < 1.0 and accuracy > 0.55:
        return "Excellent"
    elif log_loss < 1.05 and accuracy > 0.50:
        return "Good"
    elif log_loss < 1.1 and accuracy > 0.45:
        return "Fair"
    else:
        return "Poor"

def _generate_insights(fold_metrics: List[Dict], betting_metrics: Dict, calibration_metrics: Dict) -> List[str]:
    """🚀 Generate actionable insights"""
    insights = []
    
    # Consistency check
    accuracies = [fm.get('accuracy', 0) for fm in fold_metrics]
    if max(accuracies) - min(accuracies) > 0.1:
        insights.append("High variance across folds - model may be unstable")
    
    # Betting potential
    if betting_metrics.get('confident_bets_accuracy', 0) > betting_metrics.get('accuracy', 0) + 0.05:
        insights.append("Model shows potential for selective betting strategy")
    
    # Calibration
    if calibration_metrics.get('expected_calibration_error', 1) < 0.05:
        insights.append("Model is well-calibrated - probabilities are trustworthy")
    elif calibration_metrics.get('expected_calibration_error', 1) > 0.1:
        insights.append("Model needs better calibration - consider isotonic regression")
    
    # Class-specific performance
    home_f1 = np.mean([fm.get('f1_home_win', 0) for fm in fold_metrics])
    draw_f1 = np.mean([fm.get('f1_draw', 0) for fm in fold_metrics])
    away_f1 = np.mean([fm.get('f1_away_win', 0) for fm in fold_metrics])
    
    if draw_f1 < min(home_f1, away_f1) - 0.1:
        insights.append("Model struggles with draw predictions - consider draw-specific features")
    
    if not insights:
        insights.append("Model performance is balanced across all metrics")
    
    return insights