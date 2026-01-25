import numpy as np
import pandas as pd
from joblib import dump
from fchamp.data.loader import load_matches, merge_xg_into_history, merge_shots_into_history
from fchamp.features.engineering import add_elo, add_rolling_form, add_xg_real_features, add_shots_real_features, build_features
from fchamp.features.market import add_market_features
from fchamp.models.goals_poisson import GoalsPoissonModel
from fchamp.models.registry import ModelRegistry
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
from fchamp.models.calibration import OneVsRestIsotonic, MultinomialLogisticCalibrator
from sklearn.linear_model import LogisticRegression
from fchamp.models.goals_advanced import GoalsBivariatePoissonModel, GoalsNegBinModel
from fchamp.models.market_prior_corrector import MarketPriorCorrector
from fchamp.models.learned_post_corrector import LearnedPostCorrector
import logging

logger = logging.getLogger(__name__)

def _make_goal_model(kind: str, alpha: float, use_dc: bool, dc_rho: float, max_sigma: float, model_params: dict):
    kind = (kind or "poisson").lower()
    model_params = model_params or {}
    if kind == "bivariate":
        return GoalsBivariatePoissonModel(
            alpha=alpha, use_dixon_coles=False, dc_rho=dc_rho, max_sigma=max_sigma, **model_params
        )
    if kind == "negbin":
        return GoalsNegBinModel(
            alpha=alpha, use_dixon_coles=use_dc, dc_rho=dc_rho, **model_params
        )
    return GoalsPoissonModel(alpha=alpha, use_dixon_coles=use_dc, dc_rho=dc_rho, **model_params)

def _cv_logloss(X, y_out, yh, ya, alpha, use_dc, dc_rho, max_goals, n_splits, gap, kind, max_sigma, model_params):
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []

    for tr, te in tss.split(X):
        gm = _make_goal_model(kind, alpha, use_dc, dc_rho, max_sigma, model_params)
        gm.fit(X.iloc[tr], yh.iloc[tr], ya.iloc[tr])

        lh, la = gm.predict_lambdas(X.iloc[te])
        
        probs = [gm.outcome_probs(lhi, lai, max_goals) for lhi, lai in zip(lh, la)]
        
        P = np.array(probs)
        scores.append(log_loss(y_out.iloc[te], P, labels=[0, 1, 2]))

    return float(np.mean(scores))

def _oof_probs(
    X,
    y_out,
    yh,
    ya,
    n_splits: int,
    gap: int,
    alpha: float,
    use_dc: bool,
    dc_rho: float,
    max_goals: int,
    kind: str,
    max_sigma: float,
    model_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    P_list, Y_list = [], []

    for tr, te in tss.split(X):
        gm = _make_goal_model(kind, alpha, use_dc, dc_rho, max_sigma, model_params)
        gm.fit(X.iloc[tr], yh.iloc[tr], ya.iloc[tr])
        lh, la = gm.predict_lambdas(X.iloc[te])
        P = np.array([gm.outcome_probs(lhi, lai, max_goals) for lhi, lai in zip(lh, la)])
        P_list.append(P)  # shape (len(te), 3)
        Y_list.append(y_out.iloc[te].values)

    return np.vstack(P_list), np.concatenate(Y_list)

def _oof_gbm(X, y_out, n_splits, gap) -> tuple[np.ndarray, np.ndarray]:
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    P_list, Y_list = [], []

    for tr, te in tss.split(X):
        gbm = HistGradientBoostingClassifier(
            loss="log_loss", early_stopping=True, random_state=42
        )

        gbm.fit(X.iloc[tr], y_out.iloc[tr])
        proba = gbm.predict_proba(X.iloc[te])   # columns in gbm.classes_ order
        # map in ordine [0,1,2] se necessario
        classes = list(gbm.classes_)
        P = np.zeros((proba.shape[0], 3))

        for i, cls in enumerate(classes):
            P[:, int(cls)] = proba[:, i]

        P_list.append(P)
        Y_list.append(y_out.iloc[te].values)

    return np.vstack(P_list), np.concatenate(Y_list)

def run_train(cfg) -> str:
    """🚀 ENHANCED training pipeline con features avanzate opzionali"""
    logger.info("🚀 Starting enhanced training pipeline")
    
    # 1. LOAD DATA (identico a prima) + optional xG merge
    df = load_matches(cfg.data.paths, delimiter=cfg.data.delimiter)
    # Merge xG reali se disponibile nel config
    try:
        if getattr(cfg.data, 'xg_path', None):
            import pandas as _pd
            xg_df = _pd.read_csv(cfg.data.xg_path)
            # normalizza colonne attese
            # attese: date, home_team, away_team, home_xg, away_xg
            rename_map = {c.lower(): c for c in xg_df.columns}
            for need in ['date','home_team','away_team','home_xg','away_xg']:
                if need not in xg_df.columns:
                    # prova da lowercase
                    if need in rename_map:
                        pass
                    else:
                        # se mancano colonne fondamentali, salta merge
                        raise KeyError('missing xg columns')
            # assicurati lowercase
            xg_df.columns = [c.lower() for c in xg_df.columns]
            df = merge_xg_into_history(df, xg_df)
            # crea rolling xG reali
            if bool(getattr(cfg.features, 'use_xg_real', True)):
                df = add_xg_real_features(df)
            logger.info("🎯 Merged real xG into history and added rolling xG features")
    except Exception as _e:
        logger.warning(f"xG merge skipped: {_e}")
    logger.info(f"Loaded {len(df)} matches from {len(cfg.data.paths)} files")

    # 1.5 Merge shots reali
    try:
        if getattr(cfg.data, 'shots_path', None):
            sh_df = _pd.read_csv(cfg.data.shots_path)
            sh_df.columns = [c.lower() for c in sh_df.columns]
            df = merge_shots_into_history(df, sh_df)

            if bool(getattr(cfg.features, 'use_shots_real', True)):
                df = add_shots_real_features(df)
            logger.info("🎯 Merged real shots into history and added rolling shots features")
    except Exception as _e:
        logger.warning(f"shots merge skipped: {_e}")

    # 2. MARKET FEATURES (identico a prima)
    if getattr(cfg.data, "use_market", False):
        df = add_market_features(df, cfg.data.paths, delimiter=cfg.data.delimiter)
        logger.info("Added market features")

    # 3. 🚀 ENHANCED ELO (con parametri opzionali)
    elo_params = {
        'start': cfg.elo.start, 
        'k': cfg.elo.k, 
        'hfa': cfg.elo.hfa, 
        'mov_factor': cfg.elo.mov_factor
    }
    
    # Aggiungi parametri avanzati se presenti nel config
    if hasattr(cfg.elo, 'season_regression'):
        elo_params['season_regression'] = cfg.elo.season_regression
    if hasattr(cfg.elo, 'time_decay_days'):
        elo_params['time_decay_days'] = cfg.elo.time_decay_days
    if hasattr(cfg.elo, 'adaptive_k'):
        elo_params['adaptive_k'] = cfg.elo.adaptive_k
    if hasattr(cfg.elo, 'home_away_split'):
        elo_params['home_away_split'] = cfg.elo.home_away_split
    
    df = add_elo(df, **elo_params)
    logger.info(f"🚀 Added ELO features with {len(elo_params)} parameters")

    # 4. 🚀 ENHANCED ROLLING FORM (con features avanzate opzionali)
    form_params = {
        'rolling_n': cfg.features.rolling_n,
        'ewm_alpha': cfg.features.ewm_alpha
    }
    
    if hasattr(cfg.features, 'add_features'):
        form_params['add_features'] = cfg.features.add_features

    df = add_rolling_form(df, **form_params)
    logger.info("🚀 Added rolling form features")

    # 🎯 NEW: Advanced statistics features (solo se disponibili)
    try:
        from fchamp.features.advanced_stats import (
            add_shots_and_corners_features,
            add_referee_bias,
            add_head_to_head_stats,
            add_xg_proxy_features,
            add_advanced_proxy_features
        )
        from fchamp.features.engineering import create_composite_features

        # Applica solo se ci sono le colonne necessarie
        # controlla flag feature dal config
        use_adv = bool(getattr(cfg.features, 'use_advanced_stats', True))
        use_xg = bool(getattr(cfg.features, 'use_xg_proxy', True))
        # Shots/advanced proxy
        if use_adv and any(col in df.columns for col in ['HS', 'AS', 'HST', 'AST']):
            df = add_shots_and_corners_features(df)
            if use_xg:
                df = add_xg_proxy_features(df)
                logger.info("🎯 Added shots and xG features")
            df = add_advanced_proxy_features(df)
            logger.info("🎯 Added advanced proxy features")
            df = create_composite_features(df)
            logger.info("🎯 Added composite features")
        
        if 'Referee' in df.columns:
            df = add_referee_bias(df)
            logger.info("⚖️ Added referee bias features")
        
        # H2H opzionale
        if bool(getattr(cfg.features, 'use_h2h', True)):
            df = add_head_to_head_stats(df, n_matches=int(getattr(cfg.features, 'h2h_matches', 5)))
            logger.info("🤝 Added head-to-head features")
        
    except ImportError as e:
        logger.warning(f"Advanced features not available: {e}")
    
    logger.info(f"🚀 Enhanced dataset with {len(df.columns)} total columns")

    # 5. 🚀 ENHANCED FEATURE BUILDING
    feature_params = {}
    if hasattr(cfg.features, 'safe_fill'):
        feature_params['safe_fill'] = cfg.features.safe_fill
    include_adv = getattr(cfg.features, 'include_advanced', getattr(cfg.features, 'add_features', False))
    feature_params['include_advanced'] = bool(include_adv)

    X, y_out, yh, ya = build_features(df, **feature_params)
    logger.info(f"🚀 Built feature matrix: {X.shape[1]} features, {X.shape[0]} samples")

    # 6. PARAMETER SETUP (con defaults avanzati)
    alpha = cfg.model.alpha
    use_dc = cfg.model.use_dixon_coles
    dc_rho = cfg.model.dc_rho
    kind = str(getattr(getattr(cfg.model, "goal_model", None), "kind", "poisson")).lower()
    max_sigma = float(getattr(getattr(cfg.model, "goal_model", None), "max_sigma", 0.30))
    
    # 🚀 PARAMETRI MODELLO AVANZATI (opzionali)
    model_advanced_params = {}
    if hasattr(cfg.model, 'use_ensemble'):
        model_advanced_params['use_ensemble'] = cfg.model.use_ensemble
    if hasattr(cfg.model, 'ensemble_weight'):
        model_advanced_params['ensemble_weight'] = cfg.model.ensemble_weight
    if hasattr(cfg.model, 'robust_sanitization'):
        model_advanced_params['robust_sanitization'] = cfg.model.robust_sanitization
    if hasattr(cfg.model, 'adaptive_clipping'):
        model_advanced_params['adaptive_clipping'] = cfg.model.adaptive_clipping

    # 7. 🚀 ENHANCED HYPERPARAMETER TUNING
    if cfg.backtest.tune:
        logger.info("🚀 Starting enhanced hyperparameter tuning")
        try:
            import optuna
            
            def objective(trial):
                # Parametri base
                a = trial.suggest_float("alpha", 0.01, 5.0, log=True)
                rho = trial.suggest_float("dc_rho", 0.0, 0.2)
                
                # 🚀 PARAMETRI AVANZATI (se ensemble è abilitato)
                if model_advanced_params.get('use_ensemble', False):
                    ensemble_weight = trial.suggest_float("ensemble_weight", 0.3, 0.9)
                    model_advanced_params['ensemble_weight'] = ensemble_weight
                
                return _cv_logloss(
                    X, y_out, yh, ya, a, use_dc, rho, cfg.features.max_goals,
                    cfg.backtest.n_splits, cfg.backtest.gap, kind, max_sigma, model_advanced_params
                )
            
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=cfg.backtest.tune_trials)
            
            alpha = study.best_params["alpha"]
            dc_rho = study.best_params.get("dc_rho", dc_rho)
            
            if "ensemble_weight" in study.best_params:
                model_advanced_params['ensemble_weight'] = study.best_params["ensemble_weight"]
            
            logger.info(f"🚀 Optimized parameters: alpha={alpha:.4f}, dc_rho={dc_rho:.4f}")
            
        except Exception as e:
            logger.warning(f"Tuning failed: {e}, using default parameters")

    # 8. 🚀 ENHANCED MODEL CREATION 
    # with bivariate and negbin + default model
    if kind == "bivariate":
        gm = GoalsBivariatePoissonModel(
            alpha=alpha, 
            use_dixon_coles=False, 
            dc_rho=dc_rho,
            max_sigma=max_sigma,
            **model_advanced_params
        )
    elif kind == "negbin":
        gm = GoalsNegBinModel(
            alpha=alpha, 
            use_dixon_coles=use_dc, 
            dc_rho=dc_rho,
            **model_advanced_params
        )
    else:
        gm = GoalsPoissonModel(
            alpha=alpha, 
            use_dixon_coles=use_dc, 
            dc_rho=dc_rho,
            **model_advanced_params
        )
    
    logger.info(f"🚀 Created model with {len(model_advanced_params)} advanced parameters")
    gm.fit(X, yh, ya)

    # 9. CALIBRATION verrà applicata dopo eventuale stacking
    cal_meta = {}
    cal = None

    # 10. 🚀 ENHANCED GBM (con parametri migliorati)
    gbm = None
    gbm_cal = None
    gbm_meta = {}

    if getattr(cfg.model, "gbm", None) and cfg.model.gbm.enabled:
        logger.info("🚀 Training standalone GBM ensemble")
        # OOF GBM per calibrazione
        P_gbm_oof, Y_oof = _oof_gbm(X, y_out, cfg.backtest.n_splits, cfg.backtest.gap)
        gbm_cal = OneVsRestIsotonic().fit(P_gbm_oof, Y_oof)

        # Train finale GBM su tutto
        # class_weight per rinforzare la classe X (1)
        class_weight = {0: 1.0, 1: float(getattr(cfg.model, 'draw_weight', 1.0)), 2: 1.0}
        # HistGradientBoostingClassifier non supporta class_weight direttamente -> oversampling semplice
        gbm = HistGradientBoostingClassifier(
            loss="log_loss", 
            early_stopping=True, 
            random_state=42,
            max_iter=200,
            learning_rate=0.1,
            max_depth=6,
            # class_weight=class_weight
        )
        # Oversample la classe 1 in fit
        try:
            y_vals = y_out.values if hasattr(y_out, 'values') else y_out
            draw_idx = np.where(y_vals == 1)[0]
            if len(draw_idx) > 0:
                # ripeti le righe draw per aumentare il peso
                repeat_factor = int(max(1, getattr(cfg.model, 'draw_weight', 1.0)))
                idx_extra = np.random.choice(draw_idx, size=len(draw_idx) * (repeat_factor - 1), replace=True)
                X_aug = pd.concat([X, X.iloc[idx_extra]], axis=0)
                y_aug = np.concatenate([y_vals, y_vals[idx_extra]])
            else:
                X_aug, y_aug = X, y_vals
        except Exception:
            X_aug, y_aug = X, y_out
        gbm.fit(X_aug, y_aug)
        gbm_meta = {"enabled": True, "blend_weight": cfg.model.gbm.blend_weight}
        logger.info("🚀 Standalone GBM trained")

    # 11. 🚀 STACKING OOF (Poisson + opzionalmente GBM + Market)
    stacker = None
    stacker_meta = {}
    P_base_oof = None
    
    try:
        logger.info("🚀 Training OOF stacking (multinomial LR)")
        # OOF Poisson
        P_poiss_oof, Y_oof = _oof_probs(
            X, y_out, yh, ya,
            cfg.backtest.n_splits, cfg.backtest.gap,
            alpha, use_dc, dc_rho, cfg.features.max_goals,
            kind, max_sigma, model_advanced_params
        )

        # OOF GBM (se disponibile)
        P_gbm_oof = None
        if gbm is not None:
            P_gbm_oof, _ = _oof_gbm(X, y_out, cfg.backtest.n_splits, cfg.backtest.gap)

        # OOF market probs
        MK_list = []
        tss = TimeSeriesSplit(n_splits=cfg.backtest.n_splits, gap=cfg.backtest.gap)
        for _, te in tss.split(X):
            if all(col in df.columns for col in ['book_p_home','book_p_draw','book_p_away']):
                MK_list.append(df.iloc[te][['book_p_home','book_p_draw','book_p_away']].astype(float).values)
            else:
                MK_list.append(np.full((len(te),3), 1/3, dtype=float))
        MK_oof = np.vstack(MK_list)

        parts = [np.log(np.clip(P_poiss_oof, 1e-9, 1.0))]
        if P_gbm_oof is not None:
            parts.append(np.log(np.clip(P_gbm_oof, 1e-9, 1.0)))
        if MK_oof is not None:
            parts.append(np.log(np.clip(MK_oof, 1e-9, 1.0)))

        # --- MarketPriorCorrector (OOF) ---
        prior = None
        prior_meta = { "enabled": False }

        try:
            mp_cfg = getattr(cfg.model, "market_prior", None)
            if mp_cfg and bool(getattr(mp_cfg, "enabled", False)):
                Z_parts = [np.log(np.clip(P_poiss_oof, 1e-12, 1.0))]

                if P_gbm_oof is not None and bool(getattr(mp_cfg, "use_gbm", True)):
                    Z_parts.append(np.log(np.clip(P_gbm_oof, 1e-12, 1.0)))
                
                Z = np.column_stack(Z_parts)
                prior = MarketPriorCorrector(l2=float(getattr(mp_cfg, "l2", 1.0))).fit(Z, MK_oof, Y_oof)
                prior_meta = {"enabled": True, "l2": float(getattr(mp_cfg, "l2", 1.0)) }
        except Exception as e:
            prior = None
            prior_meta = { "enabled": False, "err": str(e) }

        X_stack_oof = np.column_stack(parts)
        # class_weight to enphatize 'draw' within stacker
        cw = {0: 1.0, 1: float(getattr(cfg.model, 'draw_weight', 1.0)), 2: 1.0}
        stacker = LogisticRegression(max_iter=300, solver="lbfgs", class_weight=cw)
        stacker.fit(X_stack_oof, Y_oof)
        stacker_meta = {"enabled": True, "inputs": ["poisson", "gbm" if P_gbm_oof is not None else None, "market"]}
        logger.info("🚀 Stacker trained")

        # Base OOF probs per post-processing (market prior se attivo, altrimenti stacker)
        P_base_oof = None
        try:
            if prior is not None and bool(getattr(getattr(cfg.model, "market_prior", None), "enabled", False)):
                Z_parts = [np.log(np.clip(P_poiss_oof, 1e-12, 1.0))]
                if P_gbm_oof is not None and bool(getattr(getattr(cfg.model, "market_prior", None), "use_gbm", True)):
                    Z_parts.append(np.log(np.clip(P_gbm_oof, 1e-12, 1.0)))
                Z = np.column_stack(Z_parts)
                P_base_oof = prior.predict_proba(Z, MK_oof)
            else:
                P_base_oof = stacker.predict_proba(X_stack_oof)
                try:
                    classes = list(stacker.classes_)
                    P_ord = np.zeros_like(P_base_oof)
                    for j, cls in enumerate(classes):
                        P_ord[:, int(cls)] = P_base_oof[:, j]
                    P_base_oof = P_ord
                except Exception:
                    pass
        except Exception:
            P_base_oof = None

        # 12. 🚀 CALIBRATION (post-stacking/prior)
        if getattr(cfg.model, "calibration", None) and cfg.model.calibration.enabled:
            logger.info("🚀 Training calibration (post-stacking/prior)")
            P_for_cal = np.array(P_base_oof, dtype=float) if P_base_oof is not None else None
            if P_for_cal is not None:
                if getattr(cfg.model.calibration, 'method', 'isotonic') == "isotonic":
                    cal = OneVsRestIsotonic().fit(P_for_cal, Y_oof)
                    cal_meta = {"calibrated": True, "method": "isotonic"}
                else:
                    # Calibratore multinomiale con class_weight pro-draw
                    cw = {0: 1.0, 1: float(getattr(cfg.model, 'draw_weight', 1.0)), 2: 1.0}
                    cal = MultinomialLogisticCalibrator(class_weight=cw).fit(P_for_cal, Y_oof)
                    cal_meta = {"calibrated": True, "method": "multinomial"}
                logger.info("🚀 Calibration trained")
        if cal is not None and P_base_oof is not None:
            try:
                P_base_oof = cal.transform(P_base_oof)
            except Exception:
                pass

        # 12.b Meta-modello Draw vs No-Draw (binario) su OOF Poisson (+ mercato + feature extra)
        try:
            from sklearn.model_selection import TimeSeriesSplit as _TSS
            from sklearn.linear_model import LogisticRegression as _LR
            
            # Construct OOF, elo_abs_diff and goals_ewm_sum
            _tss = _TSS(n_splits=cfg.backtest.n_splits, gap=cfg.backtest.gap)
            ELO_oof, GSUM_oof = [], []
            
            for _, _te in _tss.split(X):
                ELO_oof.append(np.abs(X.iloc[_te]['elo_diff'].values)[:, None])
                GSUM_oof.append((X.iloc[_te]['home_gf_ewm'].values + X.iloc[_te]['away_gf_ewm'].values)[:, None])
            
            ELO_oof = np.vstack(ELO_oof)
            GSUM_oof = np.vstack(GSUM_oof)

            parts_draw = [np.log(np.clip(P_poiss_oof, 1e-9, 1.0))]
            if 'MK_oof' in locals() and MK_oof is not None:
                parts_draw.append(MK_oof)
            
            parts_draw += [ELO_oof, GSUM_oof]
            X_draw = np.column_stack(parts_draw)
            y_draw = (Y_oof == 1).astype(int)

            _cw = {0: 1.0, 1: float(getattr(cfg.model, 'draw_weight', 1.0))}
            draw_meta = _LR(max_iter=300, class_weight=_cw)
            draw_meta.fit(X_draw, y_draw)
            stacker_meta['draw_meta'] = {'enabled': True}
        except Exception as _e:
            draw_meta = None
            stacker_meta['draw_meta'] = {'enabled': False, 'err': str(_e)}

    except Exception as e:
        logger.warning(f"Stacking training failed: {e}")
        P_base_oof = np.clip(P_poiss_oof, 1e-9, 1.0)
        P_base_oof = P_base_oof / P_base_oof.sum(axis=1, keepdims=True)

    # --- 12.c FINAL CALIBRATION (OOF, post-processing)
    final_cal = None
    final_cal_meta = { "enabled": False }

    try:
        # base OOF = prior/stacker/poisson (già calibrato se cal presente)
        if P_base_oof is None:
            P_base_oof = np.clip(P_poiss_oof, 1e-9, 1.0)
            P_base_oof = P_base_oof / P_base_oof.sum(axis=1, keepdims=True)

        P_oof = np.array(P_base_oof, dtype=float)

        post = None
        post_meta = { "enabled": False }

        try:
            lp_cfg = getattr(cfg.model, "learned_post", None)
            if lp_cfg and bool(getattr(lp_cfg, "enabled", False)):
                # feature input F = [log(P_base), log(MK), elo_abs, gsum, entropy]
                P_base = np.clip(P_oof, 1e-12, 1.0)
                P_base = P_base / P_base.sum(axis=1, keepdims=True)
                if MK_oof is None:
                    mk = np.full((len(P_base), 3), 1/3, dtype=float)
                else:
                    mk = np.clip(MK_oof, 1e-12, 1.0); mk = mk / mk.sum(axis=1, keepdims=True)

                elo_abs = np.abs(X["elo_diff"].values)[:, None] if "elo_diff" in X.columns else np.zeros((len(X), 1))
                gsum = ((X.get("home_gf_ewm", pd.Series(np.ones(len(X)) * 1.3)).values +
                        X.get("away_gf_ewm", pd.Series(np.ones(len(X)) * 1.1)).values)[:, None])
                ent = (-np.sum(P_base * np.log(P_base + 1e-12), axis=1, keepdims=True))

                F = np.column_stack([
                    np.log(P_base), np.log(mk), elo_abs, gsum, ent
                ])

                post = LearnedPostCorrector(l2=float(getattr(lp_cfg, "l2", 1.0))).fit(F, Y_oof)
                post_meta = { "enabled": True, "l2": float(getattr(lp_cfg,"l2",1.0))}
        except Exception as e:
            post = None
            post_meta = {"enabled": False, "err": str(e)}

        if post is not None:
            dump(post, reg.model_path(model_id) / "learned_post.joblib")
        meta["learned_post"] = post_meta

        # same logic as in "predict.py"
        # applying draw_meta on OOF
        P_final = np.array(P_oof, dtype=float, copy=True)
        use_learned_post = post is not None and bool(getattr(getattr(cfg.model, "learned_post", None), "enabled", False))

        if use_learned_post:
            if MK_oof is None:
                mk = np.full((len(P_oof), 3), 1/3, dtype=float)
            else:
                mk = np.clip(MK_oof, 1e-12, 1.0); mk = mk / mk.sum(axis=1, keepdims=True)
            P_base = np.clip(P_oof, 1e-12, 1.0)
            P_base = P_base / P_base.sum(axis=1, keepdims=True)

            elo_abs = np.abs(X["elo_diff"].values)[:, None] if "elo_diff" in X.columns else np.zeros((len(X), 1))
            gsum = ((X.get("home_gf_ewm", pd.Series(np.ones(len(X))*1.3)).values +
                    X.get("away_gf_ewm", pd.Series(np.ones(len(X))*1.1)).values)[:, None])
            ent = (-np.sum(P_base * np.log(P_base + 1e-12), axis=1, keepdims=True))
            F = np.column_stack([np.log(P_base), np.log(mk), elo_abs, gsum, ent])
            P_final = post.predict_proba(F)
        else:
            if draw_meta is not None and bool(getattr(getattr(cfg.model, "draw_meta", None), "enabled", True)):
                parts_draw = [np.log(np.clip(P_poiss_oof, 1e-9, 1.0))]

                if MK_oof is not None:
                    parts_draw.append(MK_oof)   # coherent with predict (mk raw, not log)
                elo_abs = np.abs(X["elo_diff"].values)[:, None] if "elo_diff" in X.columns else np.zeros((len(X), 1))
                gsum = ((X.get("home_gf_ewm", pd.Series(np.ones(len(X))*1.3)).values +
                        X.get("away_gf_ewm", pd.Series(np.ones(len(X))*1.1)).values)[:, None])
                X_draw = np.column_stack(parts_draw + [elo_abs, gsum])

                p_draw_hat = draw_meta.predict_proba(X_draw)[:, 1]
                bw = float(getattr(cfg.model.draw_meta, "blend_weight", 0.4))

                for i in range(P_final.shape[0]):
                    ph, px, pa = float(P_final[i, 0]), float(P_final[i, 1]), float(P_final[i, 2])
                    r_h = ph / max(ph + pa, 1e-9)
                    px_new = (1.0 - bw) * px + bw * float(p_draw_hat[i])
                    ph_new = (1.0 - px_new) * r_h
                    pa_new = 1.0 - px_new - ph_new

                    P_final[i, 0], P_final[i, 1], P_final[i, 2] = ph_new, px_new, pa_new

            # booster + near-tie + guardrails on OOF (using MK_oof and X)
            booster = getattr(getattr(cfg, "model", None), "draw_booster", None)
            mg = getattr(getattr(cfg, "model", None), "market_guardrails", None)

            for i in range(P_final.shape[0]):
                ph, px, pa = float(P_final[i, 0]), float(P_final[i, 1]), float(P_final[i, 2])

                mh, mx, ma = (1/3, 1/3, 1/3)
                if MK_oof is not None:
                    mh, mx, ma = float(MK_oof[i, 0]), float(MK_oof[i, 1]), float(MK_oof[i, 2])

                # draw_booster (copied from predict basically)
                if booster and bool(getattr(booster, "enabled", False)):
                    elo_abs_diff = float(abs(X.iloc[i].get("elo_diff", 0.0)))
                    goals_ewm_sum = float(X.iloc[i].get("home_gf_ewm", 1.3)) + float(X.iloc[i].get("away_gf_ewm", 1.1))
                    market_draw = float(mx)

                    fav_min = float(getattr(booster, "favorite_prob_min", 0.60))
                    skip_boost_if_fav = bool(getattr(booster, "skip_booster_if_favorite", True))
                    is_strong_fav = max(mh, ma) >= fav_min

                    can_boost = (
                        elo_abs_diff <= float(getattr(booster, "elo_abs_diff_max", 35.0)) and
                        goals_ewm_sum <= float(getattr(booster, "goals_ewm_sum_max", 2.6)) and
                        market_draw >= float(getattr(booster, "market_draw_min", 0.28))
                    )

                    if can_boost and (skip_boost_if_fav and is_strong_fav):
                        pass
                    elif can_boost:
                        w = float(getattr(booster, "weight", 0.25))
                        max_boost = float(getattr(booster, "max_boost", 0.08))
                        target_px = max(market_draw, px)
                        boosted_px = px + min(max_boost, w * (target_px - px))
                        rem = 1.0 - boosted_px

                        if rem > 0:
                            scale = rem / max(ph + pa, 1e-9)
                            ph *= scale
                            pa *= scale
                            px = boosted_px

                        s = ph + px + pa
                        ph, px, pa = ph/s, px/s, pa/s

                # near-tie promotion
                nt_enabled = bool(getattr(booster, "promote_near_tie", False)) if booster else False
                if nt_enabled:
                    tie_margin = float(getattr(booster, "tie_margin", 0.02))
                    fav_min = float(getattr(booster, "favorite_prob_min", 0.60)) if booster else 0.60
                    skip_nt_if_fav = bool(getattr(booster, "skip_near_tie_if_favorite", True)) if booster else True
                    is_strong_fav = max(mh, ma) >= fav_min

                    pvec = np.array([ph, px, pa], dtype=float)
                    top_idx = int(np.argmax(pvec))

                    if top_idx != 1:
                        diff = float(pvec[top_idx] - px)
                        if diff <= tie_margin:
                            if skip_nt_if_fav and is_strong_fav:
                                pass
                            else:
                                px = pvec[top_idx]
                                rem = 1.0 - px
                                hp = max(ph, 1e-12)
                                ap = max(pa, 1e-12)

                                if rem > 0 and (hp + ap) > 0:
                                    scale = rem / (hp + ap)
                                    ph = hp * scale
                                    pa = ap * scale
                                s = ph + px + pa
                                ph, px, pa = ph/s, px/s, pa/s

                # guardrails (clipping + blend, come in predict.py)
                if mg and bool(getattr(mg, "enabled", False)):
                    flat = abs(mh - 1/3) + abs(mx - 1/3) + abs(ma - 1/3) < 1e-6
                    if not flat:
                        max_dh = float(getattr(mg, "max_abs_diff_home", 0.18))
                        max_dx = float(getattr(mg, "max_abs_diff_draw", 0.14))
                        max_da = float(getattr(mg, "max_abs_diff_away", 0.18))

                        ph = np.clip(ph, mh - max_dh, mh + max_dh)
                        px = np.clip(px, mx - max_dx, mx + max_dx)
                        pa = np.clip(pa, ma - max_da, ma + max_da)

                        s = max(ph + px + pa, 1e-12)
                        ph, px, pa = ph/s, px/s, pa/s

                        bw = float(getattr(mg, "blend_weight", 0.5))
                        if bw > 0:
                            ph = (1.0 - bw) * ph + bw * mh
                            px = (1.0 - bw) * px + bw * mx
                            pa = (1.0 - bw) * pa + bw * ma
                            s = max(ph + px + pa, 1e-12)
                            ph, px, pa = ph/s, px/s, pa/s

                P_final[i, 0], P_final[i, 1], P_final[i, 2] = ph, px, pa

        # fit final calibrator onP_final
        fc_cfg = getattr(cfg.model, "final_calibration", None)
        if fc_cfg and bool(getattr(fc_cfg, "enabled", False)):
            method = str(getattr(fc_cfg, "method", "multinomial"))

            if method == "isotonic":
                final_cal = OneVsRestIsotonic().fit(P_final, Y_oof)
                final_cal_meta = { "enabled": True, "method": "isotonic" }
            else:
                cw = {0: 1.0, 1: float(getattr(cfg.model, "draw_weight", 1.0)), 2: 1.0}
                final_cal = MultinomialLogisticCalibrator(class_weight=cw).fit(P_final, Y_oof)
                final_cal_meta = {"enabled": True, "method": "multinomial"}
    except Exception as _e:
        final_cal = None
        final_cal_meta = {"enabled": False, "err": str(_e)}


    # 13. METADATA COLLECTION (🚀 enhanced)
    try:
        teams = sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]], axis=0).dropna().astype(str).str.strip()))
    except Exception:
        teams = []
    
    def _guess_league(paths: list[str]) -> str:
        p = " ".join(paths).lower()
        if "epl" in p or "premier" in p: return "epl"
        if "serie" in p or "ita" in p or "i1" in p: return "ita"
        if "dsl" in p or "denmark" in p or "superliga" in p: return "dsl"
        return "unknown"
    
    league = _guess_league(cfg.data.paths)

    # 🚀 ENHANCED METADATA
    meta = {
        "alpha": alpha, "use_dixon_coles": use_dc, "dc_rho": dc_rho,
        "features": list(X.columns), "max_goals": cfg.features.max_goals,
        "elo": cfg.elo.__dict__, "features_cfg": cfg.features.__dict__,
        "data_files": cfg.data.paths,
        "league": league,
        "teams": teams,
        
        # 🚀 ENHANCED METADATA
        "training_samples": len(X),
        "feature_count": len(X.columns),
        "unique_teams": len(teams),
        "date_range": {
            "start": df['date'].min().isoformat() if 'date' in df.columns else None,
            "end": df['date'].max().isoformat() if 'date' in df.columns else None
        },
        "model_advanced_params": model_advanced_params,
        "tuning_trials": cfg.backtest.tune_trials if cfg.backtest.tune else 0,
        "model_info": gm.get_model_info()
    }

    meta.update({"calibration": cal_meta})
    meta["final_calibration"] = final_cal_meta
    meta["gbm"] = gbm_meta
    meta["stacker"] = stacker_meta
    meta["market_prior"] = prior_meta
    meta["goal_model"] = kind

    # 14. SAVE MODEL (identico a prima, con stacker)
    reg = ModelRegistry(cfg.artifacts_dir)
    model_id = reg.create_id(meta)

    # salva calibratore (se presente) nella cartella del modello appena creato
    if cal is not None:
        cal_path = reg.model_path(model_id) / "calibrator.joblib"
        cal.save(str(cal_path))

    if final_cal is not None:
        final_cal_path = reg.model_path(model_id) / "final_calibrator.joblib"
        final_cal.save(str(final_cal_path))

    # salva GBM + calibratore GBM se presente
    if gbm is not None:
        mp = reg.model_path(model_id)
        dump(gbm, mp / "gbm.joblib")
        if gbm_cal is not None:
            gbm_cal.save(str(mp / "gbm_cal.joblib"))

    # salva stacker se presente
    if stacker is not None:
        mp = reg.model_path(model_id)
        dump(stacker, mp / "stacker.joblib")

    if prior is not None:
        dump(prior, reg.model_path(model_id) / "prior_corrector.joblib")

    # salva draw_meta se presente
    try:
        if stacker_meta.get('draw_meta', {}).get('enabled', False):
            mp = reg.model_path(model_id)
            dump(draw_meta, mp / 'draw_meta.joblib')
    except Exception:
        pass

    path = reg.save(model_id, "model.joblib", meta)
    gm.save(str(path))
    
    logger.info(f"🚀 Enhanced model saved: {model_id}")
    logger.info(f"🚀 Model features: {len(X.columns)}")
    logger.info(f"🚀 Training samples: {len(X)}")
    logger.info(f"🚀 Teams: {len(teams)}")
    
    return model_id
