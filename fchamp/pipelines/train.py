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
import logging

logger = logging.getLogger(__name__)

def _cv_logloss(X, y_out, yh, ya, alpha, use_dc, dc_rho, max_goals, n_splits, gap):
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    scores = []

    for tr, te in tss.split(X):
        gm = GoalsPoissonModel(alpha=alpha, use_dixon_coles=use_dc, dc_rho=dc_rho)
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
) -> tuple[np.ndarray, np.ndarray]:
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    P_list, Y_list = [], []

    for tr, te in tss.split(X):
        gm = GoalsPoissonModel(alpha=alpha, use_dixon_coles=use_dc, dc_rho=dc_rho)
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
                
                return _cv_logloss(X, y_out, yh, ya, a, use_dc, rho, cfg.features.max_goals, cfg.backtest.n_splits, cfg.backtest.gap)
            
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
    try:
        logger.info("🚀 Training OOF stacking (multinomial LR)")
        # OOF Poisson
        P_poiss_oof, Y_oof = _oof_probs(
            X, y_out, yh, ya,
            cfg.backtest.n_splits, cfg.backtest.gap,
            alpha, use_dc, dc_rho, cfg.features.max_goals
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

        X_stack_oof = np.column_stack(parts)
        # class_weight per enfatizzare il draw nello stacker
        cw = {0: 1.0, 1: float(getattr(cfg.model, 'draw_weight', 1.0)), 2: 1.0}
        stacker = LogisticRegression(max_iter=300, solver="lbfgs", class_weight=cw)
        stacker.fit(X_stack_oof, Y_oof)
        stacker_meta = {"enabled": True, "inputs": ["poisson", "gbm" if P_gbm_oof is not None else None, "market"]}
        logger.info("🚀 Stacker trained")

        # 12. 🚀 CALIBRATION (post-stacking)
        if getattr(cfg.model, "calibration", None) and cfg.model.calibration.enabled:
            logger.info("🚀 Training calibration (post-stacking)")
            P_for_cal = stacker.predict_proba(X_stack_oof)
            if getattr(cfg.model.calibration, 'method', 'isotonic') == "isotonic":
                cal = OneVsRestIsotonic().fit(P_for_cal, Y_oof)
                cal_meta = {"calibrated": True, "method": "isotonic"}
            else:
                # Calibratore multinomiale con class_weight pro-draw
                cw = {0: 1.0, 1: float(getattr(cfg.model, 'draw_weight', 1.0)), 2: 1.0}
                cal = MultinomialLogisticCalibrator(class_weight=cw).fit(P_for_cal, Y_oof)
                cal_meta = {"calibrated": True, "method": "multinomial"}
            logger.info("🚀 Calibration trained")

        # 12.b Meta-modello Draw vs No-Draw (binario) su OOF Poisson (+ mercato + feature extra)
        try:
            from sklearn.model_selection import TimeSeriesSplit as _TSS
            from sklearn.linear_model import LogisticRegression as _LR
            # costruisci OOF elo_abs_diff e goals_ewm_sum
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
    meta["gbm"] = gbm_meta
    meta["stacker"] = stacker_meta

    # 14. SAVE MODEL (identico a prima, con stacker)
    reg = ModelRegistry(cfg.artifacts_dir)
    model_id = reg.create_id(meta)

    # salva calibratore (se presente) nella cartella del modello appena creato
    if cal is not None:
        cal_path = reg.model_path(model_id) / "calibrator.joblib"
        cal.save(str(cal_path))

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
