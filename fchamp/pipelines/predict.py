from ast import alias
from math import isfinite
import pandas as pd
import numpy as np
from pathlib import Path
import unicodedata
import difflib
from fchamp.data.loader import load_matches, merge_xg_into_history, merge_shots_into_history
from fchamp.features.engineering import add_elo, add_rolling_form, add_shots_real_features, add_xg_real_features
from fchamp.features.market import attach_market_to_fixtures
from fchamp.models.goals_poisson import GoalsPoissonModel
from fchamp.models.goals_advanced import GoalsBivariatePoissonModel, GoalsNegBinModel
from fchamp.models.registry import ModelRegistry
from fchamp.models.calibration import OneVsRestIsotonic, MultinomialLogisticCalibrator
from fchamp.models.market_prior_corrector import MarketPriorCorrector
from fchamp.models.learned_post_corrector import LearnedPostCorrector
import json
import logging
from typing import Dict, List
from joblib import load as joblib_load

logger = logging.getLogger(__name__)

def _latest_or(model_id, reg: ModelRegistry) -> str:
    return model_id or reg.get_latest_id()

def _validate_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    """🚀 ENHANCED fixture validation con error handling robusto"""
    required_cols = ['date', 'home_team', 'away_team']
    missing_cols = [col for col in required_cols if col not in fixtures.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validazione date
    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures['date'], dayfirst=True, errors='coerce', format='mixed')
    invalid_dates = fixtures['date'].isna()
    
    if invalid_dates.any():
        logger.warning(f"Found {invalid_dates.sum()} fixtures with invalid dates, removing them")
        fixtures = fixtures[~invalid_dates]
    
    # Validazione squadre
    null_teams = fixtures['home_team'].isna() | fixtures['away_team'].isna()
    if null_teams.any():
        logger.warning(f"Found {null_teams.sum()} fixtures with null team names, removing them")
        fixtures = fixtures[~null_teams]
    
    # Controllo squadre identiche
    same_teams = fixtures['home_team'] == fixtures['away_team']
    if same_teams.any():
        logger.warning(f"Found {same_teams.sum()} fixtures with same home/away teams, removing them")
        fixtures = fixtures[~same_teams]
    
    if len(fixtures) == 0:
        raise ValueError("No valid fixtures remaining after validation")
    
    logger.info(f"Validated {len(fixtures)} fixtures successfully")
    return fixtures

def _extract_team_stats(hist: pd.DataFrame, teams: List[str], asof_date: pd.Timestamp | None = None) -> Dict[str, Dict[str, float]]:
    """🚀 ENHANCED team stats extraction con fallback robusti"""
    team_stats = {}

    hist = hist.copy()
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce")

    if asof_date is not None:
        asof_date = pd.to_datetime(asof_date).normalize()
        hist = hist.loc[hist["date"] < asof_date]

    # se esiste ancora storico prima di asof_date, fallback 'vuoto'
    if hist.empty:
        return { t: {} for t in teams }

    # Normalizza nomi + alias per allineare fixtures e storico
    def _norm_name(name: str) -> str:
        if not isinstance(name, str):
            return ""

        s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
        s = s.lower().replace('.', ' ').replace('-', ' ').replace("'", ' ')
        for suf in [' fc',' afc',' cf',' bk',' if',' fk',' sk',' sc',' u23',' u21']:
            if s.endswith(suf):
                s = s[:-len(suf)]

        return ' '.join(s.split())

    # Alias comuni per allineare nomi UI a storico
    alias_map = {
        # EPL
        "man city": "manchester city",
        "man utd": "manchester united",
        "man united": "manchester united",
        "spurs": "tottenham",
        "wolves": "wolverhampton",
        "wolverhampton wanderers": "wolverhampton",
        "nottm forest": "nottingham forest",
        "nott m forest": "nottingham forest",
        "nottingham": "nottingham forest",
        # ITA
        "juve": "juventus",
        "inter milan": "inter",
        "ac milan": "milan",
    }
    def _canon(key: str) -> str:
        return alias_map.get(key, key)
    hist_teams = pd.unique(pd.concat([hist['home_team'], hist['away_team']], axis=0).dropna().astype(str))
    hist_norm_map = { _canon(_norm_name(t)): str(t) for t in hist_teams }
    
    # Stats base sempre disponibili
    elo_home_map = hist.groupby("home_team")["elo_home_pre"].last().to_dict()
    elo_away_map = hist.groupby("away_team")["elo_away_pre"].last().to_dict()
    
    hg_map = hist.groupby("home_team")["home_gf_roll"].last().to_dict()
    hga_map = hist.groupby("home_team")["home_ga_roll"].last().to_dict()
    ag_map = hist.groupby("away_team")["away_gf_roll"].last().to_dict()
    aga_map = hist.groupby("away_team")["away_ga_roll"].last().to_dict()
    
    h_ewm = hist.groupby("home_team")["home_gf_ewm"].last().to_dict()
    a_ewm = hist.groupby("away_team")["away_gf_ewm"].last().to_dict()
    
    # 🚀 STATS AVANZATE (se disponibili)
    advanced_stats = {}
    advanced_cols = [
        'home_gf_venue', 'home_ga_venue', 'away_gf_venue', 'away_ga_venue',
        'home_rest_days', 'away_rest_days', 'home_games_14d', 'away_games_14d',
        'home_shots_roll','home_shots_on_target_roll',
        'away_shots_roll','away_shots_on_target_roll',
        "home_xg_roll", "away_xg_roll", "home_xg_ewm", "away_xg_ewm",
        # raw denominations
        "home_xg", "away_xg",
        # proxy/derivate avanzate che il modello potrebbe usare
        'home_shot_efficiency','away_shot_efficiency',
        'home_shot_accuracy','away_shot_accuracy',
        'att_home_strength','def_home_strength',
        'att_away_strength','def_away_strength',
        'att_strength_diff','home_dominance','away_dominance',
        'home_offensive_pressure','away_defensive_score'
    ]
    
    for col in advanced_cols:
        if col in hist.columns:
            if 'home_' in col:
                advanced_stats[col] = hist.groupby("home_team")[col].last().to_dict()
            else:
                advanced_stats[col] = hist.groupby("away_team")[col].last().to_dict()
    
    # Costruisci stats per ogni squadra con risoluzione alias
    for team in teams:
        key = _canon(_norm_name(team))
        tname = hist_norm_map.get(key)
        if tname is None and key:
            try:
                import difflib as _difflib
                match = _difflib.get_close_matches(key, list(hist_norm_map.keys()), n=1, cutoff=0.80)
                if match:
                    tname = hist_norm_map.get(match[0])
            except Exception:
                tname = None
        if tname is None:
            tname = team
        team_stats[team] = {
            # ELO
            'elo_home': elo_home_map.get(tname, 1500.0),
            'elo_away': elo_away_map.get(tname, 1500.0),
            
            # Form tradizionale
            'home_gf_roll': hg_map.get(tname, 1.3),
            'home_ga_roll': hga_map.get(tname, 1.3),
            'away_gf_roll': ag_map.get(tname, 1.1),
            'away_ga_roll': aga_map.get(tname, 1.1),
            'home_gf_ewm': h_ewm.get(tname, 1.3),
            'away_gf_ewm': a_ewm.get(tname, 1.1),
        }
        
        # 🚀 AGGIUNGI STATS AVANZATE
        for stat_name, stat_dict in advanced_stats.items():
            team_stats[team][stat_name] = stat_dict.get(tname, 0.0)
    
    logger.info(f"Extracted stats for {len(team_stats)} teams with {len(team_stats[list(team_stats.keys())[0]])} features each")
    return team_stats

def run_predict(cfg, fixtures_path: Path, model_id: str | None = None) -> pd.DataFrame:
    """🚀 ENHANCED prediction pipeline con error handling e performance ottimizzate"""
    logger.info(f"🚀 Starting enhanced prediction pipeline")
    
    # NOTE: Manteniamo una sola implementazione "vera" (run_predict_df) per evitare divergenze
    # tra CLI e UI/API (stacker inputs, draw_meta/booster/guardrails, final calibration, as-of cache).
    try:
        fixtures_df = pd.read_csv(fixtures_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read fixtures CSV: {fixtures_path} ({e})")
    
    return run_predict_df(cfg, fixtures_df, model_id=model_id)
    
    # 1. MODEL LOADING con validation
    reg = ModelRegistry(cfg.artifacts_dir)
    model_id = _latest_or(model_id, reg)
    
    if not model_id:
        raise RuntimeError("No available model. Train model first!")

    model_dir = reg.model_dir(model_id)
    model_file = model_dir / "model.joblib"
    
    if not model_file.exists():
        raise RuntimeError(f"Model file not found: {model_file}")
    
    try:
        meta = json.loads((model_dir / "meta.json").read_text())
        goal_kind = (meta.get("goal_model") or "poisson").lower()
        if goal_kind == "bivariate":
            gm = GoalsBivariatePoissonModel.load(str(model_file))
        elif goal_kind == "negbin":
            gm = GoalsNegBinModel.load(str(model_file))
        else:
            gm = GoalsPoissonModel.load(str(model_file))
        logger.info(f"🚀 Loaded model {model_id} with {len(meta.get('features', []))} features")
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_id}: {e}")

    cal = None
    cal_path = model_dir / "calibrator.joblib"
    if cal_path.exists():
        logger.info("🚀 Calibrator artifact found")
    
    # Final calibrator (post-processing)
    final_cal_path = model_dir / "final_calibrator.joblib"

    # GBM ensemble
    gbm = None
    gbm_cal = None
    gbm_path = model_dir / "gbm.joblib"
    
    if gbm_path.exists():
        try:
            gbm = joblib_load(str(gbm_path))
            logger.info("🚀 Loaded GBM model for blending")
        except Exception as e:
            logger.warning(f"[X] Failed to load GBM: {e}")
        
    gbm_cal_path = model_dir / "gbm_cal.joblib"
    if gbm_cal_path.exists():
        try:
            gbm_cal = OneVsRestIsotonic.load(str(gbm_cal_path))
            logger.info("🚀 Loaded GBM calibrator")
        except Exception as e:
            logger.warning(f"[X] Failed to load GBM calibrator: {e}")

    # 2. 🚀 PERFORMANCE OPTIMIZATION: Cache historical data processing
    logger.info("🚀 Processing historical data for team stats")
    try:
        hist = load_matches(cfg.data.paths, delimiter=cfg.data.delimiter)
        # Merge xG reali se presente nel config e nei file
        try:
            if getattr(cfg.data, 'xg_path', None):
                xg_df = pd.read_csv(cfg.data.xg_path)
                xg_df.columns = [c.lower() for c in xg_df.columns]
                if all(c in xg_df.columns for c in ['date','home_team','away_team','home_xg','away_xg']):
                    hist = merge_xg_into_history(hist, xg_df)
                    # aggiunge rolling xG reali
                    if bool(getattr(cfg.features, 'use_xg_real', True)):
                        hist = add_xg_real_features(hist)
        except Exception as _e:
            logger.warning(f"[-] xG merge in predict skipped: {_e}")

        # SHOTS merge
        try:
            if getattr(cfg.data, 'shots_path', None):
                sh_df = pd.read_csv(cfg.data.shots_path)
                sh_df.columns = [c.lower() for c in sh_df.columns]
                hist = merge_shots_into_history(hist, sh_df)

                if bool(getattr(cfg.features, 'use_shots_real', True)):
                    hist = add_shots_real_features(hist)

        except Exception as _e:
            logger.warning(f"[-] Shots merge in predict skipped: {_e}")
        
        # Replica feature engineering avanzata usata in train (se abilitata)
        try:
            from fchamp.features.advanced_stats import (
                add_shots_and_corners_features,
                add_head_to_head_stats,
                add_xg_proxy_features,
                add_advanced_proxy_features,
            )
            from fchamp.features.engineering import create_composite_features
            use_adv = bool(getattr(cfg.features, 'use_advanced_stats', True))
            use_xg = bool(getattr(cfg.features, 'use_xg_proxy', True))
            if use_adv and any(col in hist.columns for col in ['HS','AS','HST','AST']):
                hist = add_shots_and_corners_features(hist)
                if use_xg:
                    hist = add_xg_proxy_features(hist)
                hist = add_advanced_proxy_features(hist)
                hist = create_composite_features(hist)
            if bool(getattr(cfg.features, 'use_h2h', True)):
                try:
                    from fchamp.features.advanced_stats import add_head_to_head_stats as _h2h
                    hist = _h2h(hist, n_matches=int(getattr(cfg.features, 'h2h_matches', 5)))
                except Exception:
                    pass
        except Exception as _e:
            logger.warning(f"[-] Advanced feature build in predict skipped: {_e}")
        
        # Usa parametri dal metadata se disponibili (consistency)
        elo_config = meta.get('elo', cfg.elo.__dict__)
        hist = add_elo(hist, 
                        start=elo_config.get('start', cfg.elo.start),
                        k=elo_config.get('k', cfg.elo.k), 
                        hfa=elo_config.get('hfa', cfg.elo.hfa),
                        mov_factor=elo_config.get('mov_factor', cfg.elo.mov_factor),
                        season_regression=elo_config.get('season_regression', 0.0),
                        time_decay_days=elo_config.get('time_decay_days', 0.0),
                        adaptive_k=elo_config.get('adaptive_k', False),
                        home_away_split=elo_config.get('home_away_split', False))
        
        features_config = meta.get('features_cfg', cfg.features.__dict__)
        hist = add_rolling_form(hist, 
                                rolling_n=features_config.get('rolling_n', cfg.features.rolling_n),
                                ewm_alpha=features_config.get('ewm_alpha', cfg.features.ewm_alpha),
                                add_features=features_config.get('add_features', features_config.get('include_advanced', False)))
        
        logger.info(f"🚀 Processed {len(hist)} historical matches")
    except Exception as e:
        logger.error(f"Failed to process historical data: {e}")
        raise

    # 3. FIXTURES LOADING con validation robusta
    try:
        fixtures = pd.read_csv(fixtures_path)
        fixtures = fixtures.rename(columns={c: c.lower() for c in fixtures.columns})
        fixtures = _validate_fixtures(fixtures)
        fixtures = attach_market_to_fixtures(fixtures)
        
        # Keep only relevant columns
        keep_cols = ["date","home_team","away_team"] + [c for c in fixtures.columns if c.startswith("book_")]
        fixtures = fixtures[keep_cols]
        
        logger.info(f"🚀 Loaded and validated {len(fixtures)} fixtures")
    except Exception as e:
        logger.error(f"Failed to load fixtures from {fixtures_path}: {e}")
        raise

    # 4. 🚀 EFFICIENT TEAM STATS EXTRACTION
    all_teams = set(fixtures['home_team'].unique()) | set(fixtures['away_team'].unique())

    # normalizing a column date to obtain clearer keys
    fixtures = fixtures.copy()
    fixtures["date_norm"] = pd.to_datetime(fixtures["date"], errors="coerce").dt.normalize()

    # cache: for each date in fixtures, storico < specific date
    unique_dates = [d for d in sorted(fixtures["date_norm"].dropna().unique())]

    stats_cache = {}
    for d in unique_dates:
        stats_cache[d] = _extract_team_stats(hist, list(all_teams), asof_date=d)

    # Helper: normalizzazione nomi e lookup robusto
    def _norm_name(name: str) -> str:
        if not isinstance(name, str):
            return ""
        
        s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
        s = s.lower().replace('.', ' ').replace('-', ' ').replace("'", ' ')
        
        for suf in [' fc', ' afc', ' cf', ' bk', ' if', ' fk', ' sk', ' sc']:
            if s.endswith(suf):
                s = s[: -len(suf)]
        s = ' '.join(s.split())
        return s

    alias_map = {
        "man city":"manchester city","man utd":"manchester united","man united":"manchester united",
        "nottm forest":"nottingham forest","newcastle":"newcastle united","wolves":"wolverhampton",
        "spurs":"tottenham","juve":"juventus"
    }

    def _canon(key: str) -> str:
        return alias_map.get(key, key)

    def _get_stats(team: str, team_stats_for_date: dict) -> dict:
        # normalized mapping --> original name in team_stats_for_date
        norm_map = { _canon(_norm_name(t)): t for t in team_stats_for_date.keys() }
        norm_keys = list(norm_map.keys())
        
        # exact
        st = team_stats_for_date.get(team)
        if st is not None:
            return st
        # normalized
        key = _canon(_norm_name(team))
        tname = norm_map.get(key)
        
        if tname is not None:
            return team_stats_for_date.get(tname, {})
        
        # fuzzy fallback
        if key:
            match = difflib.get_close_matches(key, norm_keys, n=1, cutoff=0.82)
            if match:
                tname = norm_map.get(match[0])
                if tname is not None:
                    return team_stats_for_date.get(tname, {})
        
        return {}

    # 5. 🚀 VECTORIZED FEATURE BUILDING
    logger.info("🚀 Building prediction features")
    rows = []
    est_bph, est_bpd, est_bpa = [], [], []
    est_used: list[bool] = []
    for r in fixtures.itertuples(index=False):
        d = getattr(r, "date_norm", None)
        team_stats_for_date = stats_cache.get(d, stats_cache.get(unique_dates[-1], {}))   # fallback to last available date

        h, a = r.home_team, r.away_team
        # Get team stats con fallback sicuri
        h_stats = _get_stats(h, team_stats_for_date)
        a_stats = _get_stats(a, team_stats_for_date)
        
        # Build feature row
        row = {
            "elo_home_pre": h_stats.get('elo_home', 1500.0),
            "elo_away_pre": a_stats.get('elo_away', 1500.0),
            "elo_diff": h_stats.get('elo_home', 1500.0) - a_stats.get('elo_away', 1500.0),
            "home_gf_roll": h_stats.get('home_gf_roll', 1.3),
            "home_ga_roll": h_stats.get('home_ga_roll', 1.3),
            "away_gf_roll": a_stats.get('away_gf_roll', 1.1),
            "away_ga_roll": a_stats.get('away_ga_roll', 1.1),
            "home_gf_ewm": h_stats.get('home_gf_ewm', 1.3),
            "away_gf_ewm": a_stats.get('away_gf_ewm', 1.1),
        }
        
        # 🚀 ADVANCED FEATURES
        model_features = set(gm.feature_cols) if hasattr(gm, 'feature_cols') and gm.feature_cols else set()
        
        # Venue-specific features
        if 'home_gf_venue' in model_features:
            row.update({
                'home_gf_venue': h_stats.get('home_gf_venue', 1.4),
                'home_ga_venue': h_stats.get('home_ga_venue', 1.2),
                'away_gf_venue': a_stats.get('away_gf_venue', 1.0),
                'away_ga_venue': a_stats.get('away_ga_venue', 1.4),
            })
        
        # Schedule features
        if 'home_rest_days' in model_features or 'home_games_14d' in model_features:
            row.update({
                'home_rest_days': h_stats.get('home_rest_days', 7.0),
                'away_rest_days': a_stats.get('away_rest_days', 7.0),
                'rest_advantage': h_stats.get('home_rest_days', 7.0) - a_stats.get('away_rest_days', 7.0),
                'home_games_14d': h_stats.get('home_games_14d', 1.0),
                'away_games_14d': a_stats.get('away_games_14d', 1.0),
                'fatigue_differential': a_stats.get('away_games_14d', 1.0) - h_stats.get('home_games_14d', 1.0)
            })
        
        if 'home_games_14d' in model_features:
            row.update({
                'home_games_14d': h_stats.get('home_games_14d', 1.0),
                'away_games_14d': a_stats.get('away_games_14d', 1.0),
                'fatigue_differential': a_stats.get('away_games_14d', 1.0) - h_stats.get('home_games_14d', 1.0),
            })
        
        # xG total roll (se richiesto dal modello)
        if 'xg_total_roll' in model_features:
            hxg = h_stats.get("home_xg_roll", np.nan)
            axg = a_stats.get("away_xg_roll", np.nan)
            if np.isfinite(hxg) and np.isfinite(axg):
                row['xg_total_roll'] = float(hxg) + float(axg)

        # Shots features (se richieste dal modello)
        for fname, val in [
            ('home_shots_roll', h_stats.get('home_shots_roll')),
            ('home_shots_on_target_roll', h_stats.get('home_shots_on_target_roll')),
            ('away_shots_roll', a_stats.get('away_shots_roll')),
            ('away_shots_on_target_roll', a_stats.get('away_shots_on_target_roll')),
        ]:
            if fname in model_features and val is not None:
                row[fname] = float(val)

        # 🚀 MARKET FEATURES (gestione robusta)
        market_features = [c for c in ["book_p_home","book_p_draw","book_p_away","book_logit_diff"] if c in model_features]
        
        if market_features:
            # Calcolo sicuro del logit con fallback
            bph = float(getattr(r, "book_p_home", 1/3))
            bpd = float(getattr(r, "book_p_draw", 1/3))
            bpa = float(getattr(r, "book_p_away", 1/3))
            
            # Sanity check
            if not (0 < bph <= 1): bph = 1/3
            if not (0 < bpd <= 1): bpd = 1/3
            if not (0 < bpa <= 1): bpa = 1/3
            
            row.update({
                "book_p_home": bph,
                "book_p_draw": bpd,
                "book_p_away": bpa,
                "book_logit_diff": float(np.log((bph + 1e-9) / (bpa + 1e-9)))
            })
            
            # Enhanced market features
            if 'market_margin' in model_features:
                row['market_margin'] = bph + bpd + bpa - 1.0
            if 'favorite_prob' in model_features:
                row['favorite_prob'] = max(bph, bpa)
            if 'favorite_edge' in model_features:
                row['favorite_edge'] = max(bph, bpa) - min(bph, bpa)
            if 'draw_tendency' in model_features:
                row['draw_tendency'] = bpd / (bph + bpa)
        
        rows.append(row)

    # 6. 🚀 PREDICTION con error handling
    logger.info("🚀 Generating predictions")
    try:
        # Generate predictions
        Xf = pd.DataFrame(rows)

        if hasattr(gm, "feature_cols") and gm.feature_cols:
            defaults = {
                "elo_home_pre": 1500.0, "elo_away_pre": 1500.0, "elo_diff": 0.0,
                "home_gf_roll": 1.3, "home_ga_roll": 1.3, "away_gf_roll": 1.1, "away_ga_roll": 1.1,
                "home_gf_ewm": 1.3, "away_gf_ewm": 1.1,
                "home_gf_venue": 1.4, "home_ga_venue": 1.2, "home_gd_venue": 0.2,
                "away_gf_venue": 1.0, "away_ga_venue": 1.4, "away_gd_venue": -0.4,
                "home_rest_days": 7.0, "away_rest_days": 7.0, "rest_advantage": 0.0,
                "home_games_14d": 1.0, "away_games_14d": 1.0, "fatigue_differential": 0.0,
                "book_p_home": 1/3, "book_p_draw": 1/3, "book_p_away": 1/3, "book_logit_diff": 0.0,
                "market_margin": 0.05, "favorite_prob": 0.4, "favorite_edge": 0.1, "draw_tendency": 1.0,
                # defaults per shots
                "home_shots_roll": 12.0, "away_shots_roll": 10.5,
                "home_shots_on_target_roll": 4.5, "away_shots_on_target_roll": 4.0,
            }

            for col in gm.feature_cols:
                if col not in Xf.columns:
                    Xf[col] = defaults.get(col, 0.0)

            Xf = Xf[gm.feature_cols]

        # GBM ensemble probilities
        P_gbm = None
        if gbm is not None:
            try:
                proba = gbm.predict_proba(Xf)
                classes = list(gbm.classes_)
                P_gbm = np.zeros((proba.shape[0], 3), dtype=float)

                for i, cls in enumerate(classes):
                    P_gbm[:, int(cls)] = proba[:, i]

                P_gbm = np.clip(P_gbm, 1e-9, 1.0)
                P_gbm = P_gbm / P_gbm.sum(axis=1, keepdims=True)

                if gbm_cal is not None:
                    P_gbm = gbm_cal.transform(P_gbm)
            except Exception as e:
                logger.warning(f"GBM calibration transform failed: {e}")
                P_gbm = None
                
        lh, la = gm.predict_lambdas(Xf)
        # Generate all market probabilities
        probs = []
        max_goals = meta.get("max_goals", 8)
        
        for i, (lhi, lai) in enumerate(zip(lh, la)):
            try:
                m = gm.market_probs(lhi, lai, max_goals=max_goals)
                prob_row = {
                    "p_home": m["p1"], "p_draw": m["px"], "p_away": m["p2"],
                    "lambda_home": lhi, "lambda_away": lai,
                    "p_1x": m["p_1x"], "p_12": m["p_12"], "p_x2": m["p_x2"],
                    "p_over_1_5": m["p_over_1_5"], "p_over_2_5": m["p_over_2_5"],
                    "p_btts_yes": m["p_btts_yes"], "p_btts_no": m["p_btts_no"],
                    "p_home_scores": m["p_home_scores"], "p_away_scores": m["p_away_scores"],
                }
                probs.append(prob_row)
            except Exception as e:
                logger.warning(f"Failed to compute probabilities for fixture {i}: {e}")
                probs.append({
                    "p_home": 1/3, "p_draw": 1/3, "p_away": 1/3,
                    "lambda_home": 1.0, "lambda_away": 1.0,
                    "p_1x": 2/3, "p_12": 2/3, "p_x2": 2/3,
                    "p_over_1_5": 0.5, "p_over_2_5": 0.3,
                    "p_btts_yes": 0.5, "p_btts_no": 0.5,
                    "p_home_scores": 0.7, "p_away_scores": 0.7,
                })

        # Compose Poisson matrix
        P_poiss = np.array([[r["p_home"], r["p_draw"], r["p_away"]] for r in probs], dtype=float)

        # Market matrix (se disponibile)
        mk = None
        if all(c in fixtures.columns for c in ["book_p_home","book_p_draw","book_p_away"]):
            try:
                mk = fixtures[["book_p_home","book_p_draw","book_p_away"]].astype(float).values
                mk = np.clip(mk, 1e-9, 1.0)
                mk = mk / mk.sum(axis=1, keepdims=True)
            except Exception:
                mk = None

        # STACKING / MARKET PRIOR
        P = P_poiss
        used_prior = False
        try:
            use_prior = bool(getattr(getattr(cfg.model, "market_prior", None), "enabled", False)) and prior is not None
            league = (meta.get('league') or '').lower()
            mk_estimated_all = False
            try:
                mk_estimated_all = isinstance(est_used, list) and (len(est_used) == len(fixtures)) and all(bool(v) for v in est_used)
            except Exception:
                pass

            if use_prior:
                if mk is None or (league == 'epl' and mk_estimated_all):
                    mk_for_prior = np.full((P_poiss.shape[0], 3), 1/3, dtype=float)
                else:
                    mk_for_prior = mk

                Z_parts = [np.log(np.clip(P_poiss, 1e-9, 1.0))]
                if P_gbm is not None and bool(getattr(getattr(cfg.model, "market_prior", None), "use_gbm", True)):
                    Z_parts.append(np.log(np.clip(P_gbm, 1e-9, 1.0)))
                Z = np.column_stack(Z_parts)
                P = prior.predict_proba(Z, mk_for_prior)
                used_prior = True

                if cal_path.exists():
                    try:
                        try:
                            cal = MultinomialLogisticCalibrator.load(str(cal_path))
                        except Exception:
                            cal = OneVsRestIsotonic.load(str(cal_path))
                        P = cal.transform(P)
                    except Exception as e:
                        logger.warning(f"Calibration transform failed: {e}")

            if not used_prior:
                stacker = None
                stacker_path = model_dir / "stacker.joblib"
                if stacker_path.exists():
                    stacker = joblib_load(str(stacker_path))
                    parts = [np.log(np.clip(P_poiss, 1e-9, 1.0))]
                    if P_gbm is not None:
                        parts.append(np.log(np.clip(P_gbm, 1e-9, 1.0)))
                    # NON usare MK nello stacker se è tutto stimato in EPL
                    # Se manca market (es. fixtures senza quote), usa prior uniforme per rispettare le dimensioni dello stacker
                    if mk is None:
                        mk = np.full((P_poiss.shape[0], 3), 1/3, dtype=float)

                    if mk is not None and not (league == 'epl' and mk_estimated_all):
                        parts.append(np.log(np.clip(mk, 1e-9, 1.0)))
                    X_stack = np.column_stack(parts)
                    P = stacker.predict_proba(X_stack)

                    # allinea le colonne all’ordine canonico [0,1,2]
                    try:
                        classes = list(stacker.classes_)
                        P_ord = np.zeros_like(P)

                        for j, cls in enumerate(classes):
                            P_ord[:, int(cls)] = P[:, j]
                        
                        P = P_ord
                    except Exception:
                        pass

                    # Calibrazione post-stacking
                    if cal_path.exists():
                        try:
                            try:
                                cal = MultinomialLogisticCalibrator.load(str(cal_path))
                            except Exception:
                                cal = OneVsRestIsotonic.load(str(cal_path))
                            P = cal.transform(P)
                        except Exception as e:
                            logger.warning(f"Calibration transform failed: {e}")
                else:
                    # Fallback: Poisson -> (cal) -> GBM -> market blend
                    if cal_path.exists():
                        try:
                            try:
                                cal = MultinomialLogisticCalibrator.load(str(cal_path))
                            except Exception:
                                cal = OneVsRestIsotonic.load(str(cal_path))
                            P = cal.transform(P)
                        except Exception as e:
                            logger.warning(f"Calibration transform failed: {e}")
                    gbm_weight = 0.0
                    gbm_meta = meta.get("gbm", {})
                    if P_gbm is not None and gbm_meta.get("enabled", False):
                        default_weight = getattr(getattr(getattr(cfg, "model", None), "gbm", None), "blend_weight", 0.0)
                    gbm_weight = float(gbm_meta.get("blend_weight", default_weight))
                    gbm_weight = min(max(gbm_weight, 0.0), 1.0)
                    if gbm_weight > 0:
                        P = (1.0 - gbm_weight) * P + gbm_weight * P_gbm
                        P = np.clip(P, 1e-9, 1.0)
                        P = P / P.sum(axis=1, keepdims=True)
                w = float(getattr(cfg.model, "market_blend_weight", 0.0) or 0.0)
                w = min(max(w, 0.0), 1.0)
                league = (meta.get('league') or '').lower()
                # disattiva blend se il “mercato” è interamente stimato in EPL
                if league == 'epl' and mk is not None and mk_estimated_all:
                    w = 0.0
                if w > 0 and mk is not None:
                    # Applica blend solo se le quote non sono piatte (~1/3)
                    diff = np.abs(mk - 1.0/3.0).sum(axis=1)
                    mask = diff > 1e-6
                    if mask.any():
                        P_mix = (1.0 - w) * P + w * mk
                        P = np.where(mask[:, None], P_mix, P)
                        P = np.clip(P, 1e-9, 1.0)
                        P = P / P.sum(axis=1, keepdims=True)
        except Exception as e:
            logger.warning(f"Stacking pipeline failed, using Poisson-only: {e}")
            P = P_poiss
            if cal_path.exists():
                try:
                    cal = OneVsRestIsotonic.load(str(cal_path))
                    P = cal.transform(P)
                except Exception:
                    pass

        # Optional draw booster (parametrico, no retrain)
        booster = getattr(getattr(cfg, "model", None), "draw_booster", None)
        apply_booster = bool(getattr(booster, "enabled", False))

        # Write back final probs and confidence
        P_post = np.zeros_like(P, dtype=float)
        for i, r in enumerate(probs):
            ph, px, pa = float(P[i,0]), float(P[i,1]), float(P[i,2])

            if apply_booster:
                # proxy per bassa differenza di forza e bassa somma gol attesi
                elo_abs_diff = abs(Xf.iloc[i].get("elo_diff", 0.0))
                goals_ewm_sum = float(Xf.iloc[i].get("home_gf_ewm", 1.3)) + float(Xf.iloc[i].get("away_gf_ewm", 1.1))
                market_draw = float(Xf.iloc[i].get("book_p_draw", 1/3))

                if (
                    elo_abs_diff <= float(getattr(booster, "elo_abs_diff_max", 35.0)) and
                    goals_ewm_sum <= float(getattr(booster, "goals_ewm_sum_max", 2.6)) and
                    market_draw >= float(getattr(booster, "market_draw_min", 0.28))
                ):
                    w = float(getattr(booster, "weight", 0.25))
                    max_boost = float(getattr(booster, "max_boost", 0.08))
                    target_px = max(market_draw, px)
                    # blend verso market draw con clipping
                    boosted_px = px + min(max_boost, w * (target_px - px))
                    rem = 1.0 - boosted_px
                    if rem > 0:
                        scale = rem / max(ph + pa, 1e-9)
                        ph *= scale
                        pa *= scale
                        px = boosted_px
                    # rinormalizza (robust)
                    s = ph + px + pa
                    ph, px, pa = ph/s, px/s, pa/s

            P_post[i, 0], P_post[i, 1], P_post[i, 2] = ph, px, pa

        # --- FINAL CALIBRATION (ultima trasformazione) ---
        if final_cal_path.exists() and bool(getattr(getattr(cfg.model, "final_calibration", None), "enabled", False)):
            try:
                try:
                    fcal = MultinomialLogisticCalibrator.load(str(final_cal_path))
                except Exception:
                    fcal = OneVsRestIsotonic.load(str(final_cal_path))
                P_post = fcal.transform(P_post)
            except Exception as e:
                logger.warning(f"Final calibration transform failed: {e}")

        # write into output rows + recompute summary stats
        for i, r in enumerate(probs):
            ph, px, pa = float(P_post[i, 0]), float(P_post[i, 1]), float(P_post[i, 2])
            r["p_home"], r["p_draw"], r["p_away"] = ph, px, pa
            r["prediction_confidence"] = max(ph, px, pa)
            r["prediction_entropy"] = -sum(p * np.log(p + 1e-10) for p in [ph, px, pa])

        logger.info(f"🚀 Generated {len(probs)} predictions successfully")
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        raise

    # 7. 🚀 RESULT ASSEMBLY con metadata
    result = pd.concat([fixtures.reset_index(drop=True), pd.DataFrame(probs)], axis=1)
    
    # Add prediction metadata
    result['model_id'] = model_id
    result['prediction_timestamp'] = pd.Timestamp.now()
    
    # 🚀 QUALITY METRICS
    avg_confidence = result['prediction_confidence'].mean()
    low_confidence_count = (result['prediction_confidence'] < 0.4).sum()
    
    logger.info(f"🚀 Prediction Summary:")
    logger.info(f"   - Total predictions: {len(result)}")
    logger.info(f"   - Average confidence: {avg_confidence:.3f}")
    logger.info(f"   - Low confidence predictions: {low_confidence_count}")
    logger.info(f"   - Model used: {model_id}")
    
    return result

def run_predict_df(cfg, fixtures_df: pd.DataFrame, model_id: str | None = None) -> pd.DataFrame:
    """🚀 ENHANCED DataFrame prediction con stessa logica ottimizzata"""
    logger.info("🚀 Starting DataFrame prediction")
    
    # Usa stessa logica di run_predict ma con DataFrame input
    reg = ModelRegistry(cfg.artifacts_dir)
    model_id = _latest_or(model_id, reg)
    
    if not model_id:
        raise RuntimeError("No available model. Train model first!")
    
    model_dir = reg.model_dir(model_id)
    model_file = model_dir / "model.joblib"
    meta = json.loads((model_dir / "meta.json").read_text())
    goal_kind = (meta.get("goal_model") or "poisson").lower()
    if goal_kind == "bivariate":
        gm = GoalsBivariatePoissonModel.load(str(model_file))
    elif goal_kind == "negbin":
        gm = GoalsNegBinModel.load(str(model_file))
    else:
        gm = GoalsPoissonModel.load(str(model_file))
    cal = None
    cal_path = model_dir / "calibrator.joblib"
    final_cal_path = model_dir / "final_calibrator.joblib"
    if cal_path.exists():
        try:
            try:
                cal = MultinomialLogisticCalibrator.load(str(cal_path))
            except Exception:
                cal = OneVsRestIsotonic.load(str(cal_path))
        except Exception:
            cal = None
    gbm = None
    gbm_cal = None
    gbm_path = model_dir / "gbm.joblib"
    if gbm_path.exists(): gbm = joblib_load(str(gbm_path))
    gbm_cal_path = model_dir / "gbm_cal.joblib"
    if gbm_cal_path.exists(): gbm_cal = OneVsRestIsotonic.load(str(gbm_cal_path))

    prior = None
    prior_path = model_dir / "prior_corrector.joblib"
    if prior_path.exists():
        try:
            prior = joblib_load(str(prior_path))
        except Exception as e:
            logger.warning(f"Failed to load market prior corrector: {e}")

    post = None
    post_path = model_dir / "learned_post.joblib"
    if post_path.exists():
        try:
            post = joblib_load(str(post_path))
        except Exception as e:
            logger.warning(f"Failed to load learned post-corrector: {e}")


    # Process historical data (stesso del run_predict)
    hist = load_matches(cfg.data.paths, delimiter=cfg.data.delimiter)

    # --- merge real xG (ITA only) ---
    try:
        if getattr(cfg.data, "xg_path", None):
            xg_df = pd.read_csv(cfg.data.xg_path)
            xg_df.columns = [c.lower() for c in xg_df.columns]

            if all(c in xg_df.columns for c in ["date", "home_team", "away_team", "home_xg", "away_xg"]):
                hist = merge_xg_into_history(hist, xg_df)

                if bool(getattr(cfg.features, "use_xg_real", True)):
                    hist = add_xg_real_features(hist)
    except Exception as _e:
        logger.warning(f"[-] xG merge in predict_df skipped: {_e}")

    # --- merge real shots (ITA only) ---
    try:
        if getattr(cfg.data, "shots_path", None):
            sh_df = pd.read_csv(cfg.data.shots_path)
            sh_df.columns = [c.lower() for c in sh_df.columns]
            hist = merge_shots_into_history(hist, sh_df)

            if bool(getattr(cfg.features, "use_shots_real", True)):
                hist = add_shots_real_features(hist)
    except Exception as _e:
        logger.warning(f"[-] shots merge in predict_df skipped: {_e}")

    elo_config = meta.get('elo', cfg.elo.__dict__)
    hist = add_elo(hist, 
                  start=elo_config.get('start', cfg.elo.start),
                  k=elo_config.get('k', cfg.elo.k), 
                  hfa=elo_config.get('hfa', cfg.elo.hfa),
                  mov_factor=elo_config.get('mov_factor', cfg.elo.mov_factor),
                  season_regression=elo_config.get('season_regression', 0.0),
                  time_decay_days=elo_config.get('time_decay_days', 0.0),
                  adaptive_k=elo_config.get('adaptive_k', False),
                  home_away_split=elo_config.get('home_away_split', False))
    
    features_config = meta.get('features_cfg', cfg.features.__dict__)
    hist = add_rolling_form(hist, 
                          rolling_n=features_config.get('rolling_n', cfg.features.rolling_n),
                          ewm_alpha=features_config.get('ewm_alpha', cfg.features.ewm_alpha),
                          add_features=features_config.get('add_features', features_config.get('include_advanced', False)))

    # Process fixtures DataFrame
    fixtures = fixtures_df.rename(columns={c: c.lower() for c in fixtures_df.columns}).copy()
    fixtures = _validate_fixtures(fixtures)
    fixtures = attach_market_to_fixtures(fixtures)
    fixtures = fixtures[["date","home_team","away_team"] + [c for c in fixtures.columns if c.startswith("book_")]]

    # Extract team stats (AS-OF date cache)
    all_teams = set(fixtures['home_team'].unique()) | set(fixtures['away_team'].unique())
    
    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures["date_norm"] = fixtures["date"].dt.normalize()

    unique_dates = [d for d in sorted(fixtures["date_norm"].dropna().unique())]

    stats_cache = {}
    for d in unique_dates:
        stats_cache[d] = _extract_team_stats(hist, list(all_teams), asof_date=d)

    # League priors per fallback per squadre nuove/ignote
    def _safe_med(col: str, default: float) -> float:
        return float(pd.to_numeric(hist.get(col, pd.Series([default])), errors='coerce').dropna().median()) if col in hist.columns else float(default)

    fallback_stats = {
        'elo_home': _safe_med('elo_home_pre', 1500.0),
        'elo_away': _safe_med('elo_away_pre', 1500.0),
        'home_gf_roll': _safe_med('home_gf_roll', 1.3),
        'home_ga_roll': _safe_med('home_ga_roll', 1.3),
        'away_gf_roll': _safe_med('away_gf_roll', 1.1),
        'away_ga_roll': _safe_med('away_ga_roll', 1.1),
        'home_gf_ewm': _safe_med('home_gf_ewm', 1.3),
        'away_gf_ewm': _safe_med('away_gf_ewm', 1.1),
    }

    # Normalizzazione nomi + alias + fuzzy fallback (come in run_predict)
    def _norm_name(name: str) -> str:
        if not isinstance(name, str):
            return ""
        s = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
        s = s.lower().replace('.', ' ').replace('-', ' ').replace("'", ' ')
        for suf in [' fc', ' afc', ' cf', ' bk', ' if', ' fk', ' sk', ' sc']:
            if s.endswith(suf):
                s = s[: -len(suf)]
        s = ' '.join(s.split())
        return s

    alias_map = {
        # EPL comuni
        "man city": "manchester city",
        "man utd": "manchester united",
        "man united": "manchester united",
        "nottm forest": "nottingham forest",
        "nott m forest": "nottingham forest",
        "nottingham": "nottingham forest",
        "newcastle": "newcastle united",
        "wolves": "wolverhampton",
        "wolverhampton wanderers": "wolverhampton",
        "spurs": "tottenham",
        # ITA comuni
        "juve": "juventus",
        "inter milan": "inter",
    }

    def _canon(key: str) -> str:
        return alias_map.get(key, key)

    def _get_stats(team: str, team_stats_for_date: dict) -> dict:
        # mappa normalizzata -> nome originale presente nello storico
        norm_map = { _canon(_norm_name(t)): t for t in team_stats_for_date.keys() }
        norm_keys = list(norm_map.keys())
        # exact
        st = team_stats_for_date.get(team)
        if st is not None:
            return st
        # normalized
        key = _canon(_norm_name(team))
        tname = norm_map.get(key)
        if tname is not None:
            return team_stats_for_date.get(tname, {})
        # fuzzy fallback (cutoff basso per alias strani tipo "Nott'm Forest")
        if key:
            import difflib as _difflib
            match = _difflib.get_close_matches(key, norm_keys, n=1, cutoff=0.80)
            if match:
                tname = norm_map.get(match[0])
                if tname is not None:
                    return team_stats_for_date.get(tname, {})
        # league prior fallback
        return fallback_stats.copy()

    # Build features e prediction (stessa logica di run_predict)
    rows = []
    est_bph, est_bpd, est_bpa = [], [], []
    est_used: list[bool] = []
    for r in fixtures.itertuples(index=False):
        d = getattr(r, "date_norm", None)
        team_stats_for_date = stats_cache.get(d, stats_cache.get(unique_dates[-1], {})) if unique_dates else {}

        h, a = r.home_team, r.away_team
        h_stats = _get_stats(h, team_stats_for_date)
        a_stats = _get_stats(a, team_stats_for_date)
        
        row = {
            "elo_home_pre": h_stats.get('elo_home', 1500.0),
            "elo_away_pre": a_stats.get('elo_away', 1500.0),
            "elo_diff": h_stats.get('elo_home', 1500.0) - a_stats.get('elo_away', 1500.0),
            "home_gf_roll": h_stats.get('home_gf_roll', 1.3),
            "home_ga_roll": h_stats.get('home_ga_roll', 1.3),
            "away_gf_roll": a_stats.get('away_gf_roll', 1.1),
            "away_ga_roll": a_stats.get('away_ga_roll', 1.1),
            "home_gf_ewm": h_stats.get('home_gf_ewm', 1.3),
            "away_gf_ewm": a_stats.get('away_gf_ewm', 1.1),
        }

        model_features = set(gm.feature_cols) if hasattr(gm, 'feature_cols') and gm.feature_cols else set()

        # xG total roll (as-of) if required by model
        if "xg_total_roll" in model_features:
            hxg = h_stats.get("home_xg_roll", np.nan)
            axg = a_stats.get("away_xg_roll", np.nan)

            if np.isfinite(hxg) and np.isfinite(axg):
                row["xg_total_roll"] = float(hxg) + float(axg)

        # Venue-specific features
        if 'home_gf_venue' in model_features:
            row.update({
                'home_gf_venue': h_stats.get('home_gf_venue', 1.4),
                'home_ga_venue': h_stats.get('home_ga_venue', 1.2),
                'away_gf_venue': a_stats.get('away_gf_venue', 1.0),
                'away_ga_venue': a_stats.get('away_ga_venue', 1.4),
            })

        # Schedule features
        if 'home_rest_days' in model_features or 'home_games_14d' in model_features:
            row.update({
                'home_rest_days': h_stats.get('home_rest_days', 7.0),
                'away_rest_days': a_stats.get('away_rest_days', 7.0),
                'rest_advantage': h_stats.get('home_rest_days', 7.0) - a_stats.get('away_rest_days', 7.0),
                'home_games_14d': h_stats.get('home_games_14d', 1.0),
                'away_games_14d': a_stats.get('away_games_14d', 1.0),
                'fatigue_differential': a_stats.get('away_games_14d', 1.0) - h_stats.get('home_games_14d', 1.0),
            })

        # Shots features (se richieste dal modello)
        for fname, val in [
            ('home_shots_roll', h_stats.get('home_shots_roll')),
            ('home_shots_on_target_roll', h_stats.get('home_shots_on_target_roll')),
            ('away_shots_roll', a_stats.get('away_shots_roll')),
            ('away_shots_on_target_roll', a_stats.get('away_shots_on_target_roll'))]:
            
            if fname in model_features and val is not None:
                row[fname] = float(val)

        # Market features
        market_features = [c for c in ["book_p_home","book_p_draw","book_p_away","book_logit_diff"] if c in model_features]
        if market_features:
            bph = float(getattr(r, "book_p_home", 1/3))
            bpd = float(getattr(r, "book_p_draw", 1/3))
            bpa = float(getattr(r, "book_p_away", 1/3))
            if not (0 < bph <= 1): bph = 1/3
            if not (0 < bpd <= 1): bpd = 1/3
            if not (0 < bpa <= 1): bpa = 1/3
            row.update({
                "book_p_home": bph, "book_p_draw": bpd, "book_p_away": bpa,
                "book_logit_diff": float(np.log((bph + 1e-9) / (bpa + 1e-9)))
            })
            if 'market_margin' in model_features:
                row['market_margin'] = bph + bpd + bpa - 1.0
            if 'favorite_prob' in model_features:
                row['favorite_prob'] = max(bph, bpa)
            if 'favorite_edge' in model_features:
                row['favorite_edge'] = max(bph, bpa) - min(bph, bpa)
            if 'draw_tendency' in model_features:
                row['draw_tendency'] = bpd / (bph + bpa)

        # Stima market probs se mancanti (per guardrails/blend e per evitare 1/3-1/3-1/3)
        # Basata su elo_diff e somma gol attesa
        bph_cur = float(getattr(r, "book_p_home", np.nan)) if hasattr(r, 'book_p_home') else np.nan
        bpd_cur = float(getattr(r, "book_p_draw", np.nan)) if hasattr(r, 'book_p_draw') else np.nan
        bpa_cur = float(getattr(r, "book_p_away", np.nan)) if hasattr(r, 'book_p_away') else np.nan
        # Se sono valori placeholder ~1/3, considerali mancanti per stimare dal modello
        flat_defaults = (np.isfinite(bph_cur) and np.isfinite(bpd_cur) and np.isfinite(bpa_cur) and 
                         abs(bph_cur - 1/3) + abs(bpd_cur - 1/3) + abs(bpa_cur - 1/3) < 1e-6)
        if flat_defaults:
            bph_cur = bpd_cur = bpa_cur = np.nan
        if not (np.isfinite(bph_cur) and np.isfinite(bpd_cur) and np.isfinite(bpa_cur)):
            elo_diff = float(row["elo_diff"])  # positivo favore casa
            goals_sum = float(row.get("home_gf_ewm", 1.3)) + float(row.get("away_gf_ewm", 1.1))
            league = (meta.get('league') or '').lower()
            scale = 60.0 if league == 'epl' else 75.0
            px_cap_hi = 0.34 if league == 'epl' else 0.36

            px_est = float(np.clip(0.30 - 0.08 * (goals_sum - 2.4), 0.18, px_cap_hi))
            base_home = 1.0 / (1.0 + np.exp(- elo_diff / scale))

            rem = max(1.0 - px_est, 1e-6)
            ph_est = float(np.clip(rem * base_home, 1e-6, 1.0))
            pa_est = float(max(rem - ph_est, 1e-6))
            s = ph_est + px_est + pa_est
            ph_est, px_est, pa_est = ph_est/s, px_est/s, pa_est/s
            est_used.append(True)
        else:
            ph_est, px_est, pa_est = bph_cur, bpd_cur, bpa_cur
            est_used.append(False)
        est_bph.append(ph_est); est_bpd.append(px_est); est_bpa.append(pa_est)

        rows.append(row)

    # Genera/aggiorna market probs stimati nel DataFrame fixtures per l'uso dei guardrails
    try:
        import numpy as _np
        fixtures["book_p_home"] = pd.to_numeric(fixtures.get("book_p_home", _np.nan), errors='coerce')
        fixtures["book_p_draw"] = pd.to_numeric(fixtures.get("book_p_draw", _np.nan), errors='coerce')
        fixtures["book_p_away"] = pd.to_numeric(fixtures.get("book_p_away", _np.nan), errors='coerce')

        if len(est_bph) == len(fixtures):
            est_bph_s = pd.Series(est_bph, index=fixtures.index)
            est_bpd_s = pd.Series(est_bpd, index=fixtures.index)
            est_bpa_s = pd.Series(est_bpa, index=fixtures.index)

            h = fixtures["book_p_home"]; x = fixtures["book_p_draw"]; a = fixtures["book_p_away"]
            mask_h = h.isna() | (h.sub(1.0/3.0).abs() < 1e-6)
            mask_x = x.isna() | (x.sub(1.0/3.0).abs() < 1e-6)
            mask_a = a.isna() | (a.sub(1.0/3.0).abs() < 1e-6)

            fixtures.loc[mask_h, "book_p_home"] = est_bph_s[mask_h]
            fixtures.loc[mask_x,  "book_p_draw"] = est_bpd_s[mask_x]
            fixtures.loc[mask_a,  "book_p_away"] = est_bpa_s[mask_a]
    except Exception:
        pass

    # Generate predictions
    Xf = pd.DataFrame(rows)

    if hasattr(gm, "feature_cols") and gm.feature_cols:
        defaults = {
            "elo_home_pre": 1500.0, "elo_away_pre": 1500.0, "elo_diff": 0.0,
            "home_gf_roll": 1.3, "home_ga_roll": 1.3, "away_gf_roll": 1.1, "away_ga_roll": 1.1,
            "home_gf_ewm": 1.3, "away_gf_ewm": 1.1,
            "home_gf_venue": 1.4, "home_ga_venue": 1.2, "home_gd_venue": 0.2,
            "away_gf_venue": 1.0, "away_ga_venue": 1.4, "away_gd_venue": -0.4,
            "home_rest_days": 7.0, "away_rest_days": 7.0, "rest_advantage": 0.0,
            "home_games_14d": 1.0, "away_games_14d": 1.0, "fatigue_differential": 0.0,
            "book_p_home": 1/3, "book_p_draw": 1/3, "book_p_away": 1/3, "book_logit_diff": 0.0,
            "market_margin": 0.05, "favorite_prob": 0.4, "favorite_edge": 0.1, "draw_tendency": 1.0,
            # defaults per shots
            "home_shots_roll": 12.0, "away_shots_roll": 10.5,
            "home_shots_on_target_roll": 4.5, "away_shots_on_target_roll": 4.0,
        }

        for col in gm.feature_cols:
            if col not in Xf.columns:
                Xf[col] = defaults.get(col, 0.0)

        Xf = Xf[gm.feature_cols]

    lh, la = gm.predict_lambdas(Xf)
    league = (meta.get('league') or '').lower()
    if league == 'epl':
        lh = np.clip(lh, 0.25, 10.0)
        la = np.clip(la, 0.25, 10.0)

    probs = []
    max_goals = meta.get("max_goals", 8)
    
    for lhi, lai in zip(lh, la):
        m = gm.market_probs(lhi, lai, max_goals=max_goals)
        probs.append({
            "p_home": m["p1"], "p_draw": m["px"], "p_away": m["p2"],
            "lambda_home": lhi, "lambda_away": lai,
            "p_1x": m["p_1x"], "p_12": m["p_12"], "p_x2": m["p_x2"],
            "p_over_1_5": m["p_over_1_5"], "p_over_2_5": m["p_over_2_5"],
            "p_btts_yes": m["p_btts_yes"], "p_btts_no": m["p_btts_no"],
            "p_home_scores": m["p_home_scores"], "p_away_scores": m["p_away_scores"],
            "prediction_confidence": max(m["p1"], m["px"], m["p2"]),
            "prediction_entropy": -sum(p * np.log(p + 1e-10) for p in [m["p1"], m["px"], m["p2"]]),
            # spiegazioni (di default false, saranno aggiornate sotto se applicate)
            "explain_draw_boosted": False,
            "explain_near_tie_promoted": False,
            "explain_guardrails_applied": False,
            "explain_favorite_gate_blocked": False,
            "explain_flat_market_skipped": False,
        })

    # Compose Poisson probability matrix
    P_poiss = np.array([[r["p_home"], r["p_draw"], r["p_away"]] for r in probs], dtype=float)

    # GBM probabilities (optional)
    P_gbm = None
    if gbm is not None:
        proba = gbm.predict_proba(Xf)
        classes = list(gbm.classes_)
        P_gbm = np.zeros((proba.shape[0], 3), dtype=float)

        for i, cls in enumerate(classes):
            P_gbm[:, int(cls)] = proba[:, i]
        
        P_gbm = np.clip(P_gbm, 1e-9, 1.0)
        P_gbm = P_gbm / P_gbm.sum(axis=1, keepdims=True)

        if gbm_cal is not None:
            P_gbm = gbm_cal.transform(P_gbm)

    # Market matrix
    mk = None
    if all(c in fixtures.columns for c in ["book_p_home","book_p_draw","book_p_away"]):
        mk = fixtures[["book_p_home","book_p_draw","book_p_away"]].astype(float).values
        mk = np.clip(mk, 1e-9, 1.0)
        mk = mk / mk.sum(axis=1, keepdims=True)

    # STACKING / MARKET PRIOR
    P = P_poiss
    mk_estimated_all = False
    try:
        mk_estimated_all = isinstance(est_used, list) and (len(est_used) == len(fixtures)) and all(bool(v) for v in est_used)
    except Exception:
        mk_estimated_all = False

    try:
        use_prior = bool(getattr(getattr(cfg.model, "market_prior", None), "enabled", False)) and prior is not None
        league = (meta.get('league') or '').lower()

        if use_prior:
            if mk is None or (league == 'epl' and mk_estimated_all):
                mk_for_prior = np.full((P_poiss.shape[0], 3), 1/3, dtype=float)
            else:
                mk_for_prior = mk

            Z_parts = [np.log(np.clip(P_poiss, 1e-9, 1.0))]
            if P_gbm is not None and bool(getattr(getattr(cfg.model, "market_prior", None), "use_gbm", True)):
                Z_parts.append(np.log(np.clip(P_gbm, 1e-9, 1.0)))
            Z = np.column_stack(Z_parts)
            P = prior.predict_proba(Z, mk_for_prior)

            if cal_path.exists():
                try:
                    try:
                        cal = MultinomialLogisticCalibrator.load(str(cal_path))
                    except Exception:
                        cal = OneVsRestIsotonic.load(str(cal_path))
                    P = cal.transform(P)
                except Exception:
                    pass
        else:
            stacker = None
            stacker_path = model_dir / "stacker.joblib"
            if stacker_path.exists():
                stacker = joblib_load(str(stacker_path))
                parts = [np.log(np.clip(P_poiss, 1e-9, 1.0))]
                if P_gbm is not None:
                    parts.append(np.log(np.clip(P_gbm, 1e-9, 1.0)))
                # NON usare MK nello stacker se è tutto stimato in EPL
                if mk is not None and not (league == 'epl' and mk_estimated_all):
                    parts.append(np.log(np.clip(mk, 1e-9, 1.0)))

                X_stack = np.column_stack(parts)
                P = stacker.predict_proba(X_stack)
                # allinea le colonne all’ordine canonico [0,1,2]
                try:
                    classes = list(stacker.classes_)
                    P_ord = np.zeros_like(P)
                    for j, cls in enumerate(classes):
                        P_ord[:, int(cls)] = P[:, j]
                    P = P_ord
                except Exception:
                    pass
                # calibration post-stacking
                if cal_path.exists():
                    try:
                        try:
                            cal = MultinomialLogisticCalibrator.load(str(cal_path))
                        except Exception:
                            cal = OneVsRestIsotonic.load(str(cal_path))
                        P = cal.transform(P)
                    except Exception:
                        pass
            else:
                # fallback: Poisson -> (cal) -> GBM blend -> market blend
                if cal is not None and P.size:
                    P = cal.transform(P)

                gbm_weight = float(meta.get("gbm", {}).get(
                    "blend_weight",
                    getattr(getattr(getattr(cfg, "model", None), "gbm", None), "blend_weight", 0.0)
                ))

                gbm_weight = min(max(gbm_weight, 0.0), 1.0)
                if P_gbm is not None and gbm_weight > 0:
                    P = (1.0 - gbm_weight) * P + gbm_weight * P_gbm
                    P = np.clip(P, 1e-9, 1.0)
                    P = P / P.sum(axis=1, keepdims=True)

                w = float(getattr(cfg.model, "market_blend_weight", 0.0) or 0.0)
                w = min(max(w, 0.0), 1.0)
                
                # disattiva blend se il “mercato” è interamente stimato in EPL
                if league == 'epl' and mk is not None and mk_estimated_all:
                    w = 0.0
                if w > 0 and mk is not None:
                    P = (1.0 - w) * P + w * mk
                    P = np.clip(P, 1e-9, 1.0)
                    P = P / P.sum(axis=1, keepdims=True)
    except Exception:
        P = P_poiss
        if cal is not None and P.size:
            try:
                P = cal.transform(P)
            except Exception:
                pass

    use_learned_post = bool(getattr(getattr(cfg.model, "learned_post", None), "enabled", False)) and post is not None
    if use_learned_post:
        if mk is None:
            mk_lp = np.full((len(P), 3), 1/3, dtype=float)
        else:
            mk_lp = mk
        P_base = np.clip(P, 1e-12, 1.0)
        P_base = P_base / P_base.sum(axis=1, keepdims=True)
        elo_abs = np.abs(Xf['elo_diff'].values)[:, None] if 'elo_diff' in Xf.columns else np.zeros((len(Xf),1))
        gsum = ((Xf.get('home_gf_ewm', pd.Series(np.ones(len(Xf))*1.3)).values +
                 Xf.get('away_gf_ewm', pd.Series(np.ones(len(Xf))*1.1)).values)[:, None])
        ent = (-np.sum(P_base * np.log(P_base + 1e-12), axis=1, keepdims=True))
        F = np.column_stack([np.log(P_base), np.log(np.clip(mk_lp, 1e-12, 1.0)), elo_abs, gsum, ent])
        P = post.predict_proba(F)

    # ---- Draw vs No-Draw meta blend (binario) con feature extra ----
    if not use_learned_post:
        try:
            dm_path = model_dir / "draw_meta.joblib"
            if dm_path.exists() and getattr(getattr(cfg.model, 'draw_meta', None), 'enabled', True):
                from joblib import load as _load
                draw_meta = _load(str(dm_path))
                parts_draw = [np.log(np.clip(P_poiss, 1e-9, 1.0))]
                if mk is not None:
                    parts_draw.append(mk)
                elo_abs = np.abs(Xf['elo_diff'].values)[:, None] if 'elo_diff' in Xf.columns else np.zeros((len(Xf),1))
                gsum = ((Xf.get('home_gf_ewm', pd.Series(np.ones(len(Xf))*1.3)).values +
                         Xf.get('away_gf_ewm', pd.Series(np.ones(len(Xf))*1.1)).values)[:, None])
                parts_draw += [elo_abs, gsum]
                X_draw = np.column_stack(parts_draw)
                p_draw_hat = draw_meta.predict_proba(X_draw)[:, 1]
                bw = float(getattr(cfg.model.draw_meta, 'blend_weight', 0.4))
                for i in range(P.shape[0]):
                    ph, px, pa = float(P[i,0]), float(P[i,1]), float(P[i,2])
                    r_h = ph / max(ph + pa, 1e-9)
                    px_new = (1.0 - bw) * px + bw * float(p_draw_hat[i])
                    ph_new = (1.0 - px_new) * r_h
                    pa_new = 1.0 - px_new - ph_new
                    P[i,0], P[i,1], P[i,2] = ph_new, px_new, pa_new
        except Exception:
            pass

    # Optional draw booster (parametrico, no retrain)
    booster = getattr(getattr(cfg, "model", None), "draw_booster", None)
    apply_booster = bool(getattr(booster, "enabled", False)) and not use_learned_post

    # write back final probs
    P_post = np.zeros((len(probs), 3), dtype=float)
    for i, r in enumerate(probs):
        ph, px, pa = float(P[i,0]), float(P[i,1]), float(P[i,2])

        # market helper (fallback a fixtures se non presente in Xf)
        def _get_market_probs_row(idx: int) -> tuple[float, float, float]:
            mh = Xf.iloc[idx].get("book_p_home", np.nan)
            mx = Xf.iloc[idx].get("book_p_draw", np.nan)
            ma = Xf.iloc[idx].get("book_p_away", np.nan)
            if not np.isfinite(mh):
                mh = getattr(fixtures.iloc[idx], "book_p_home", 1/3)
            if not np.isfinite(mx):
                mx = getattr(fixtures.iloc[idx], "book_p_draw", 1/3)
            if not np.isfinite(ma):
                ma = getattr(fixtures.iloc[idx], "book_p_away", 1/3)
            return float(mh), float(mx), float(ma)

        mh, mx, ma = _get_market_probs_row(i)

        # flags di spiegazione
        draw_boosted = False
        near_tie_promoted = False
        guardrails_applied = False
        fav_gate_blocked = False
        flat_market_skipped = False

        if apply_booster:
            elo_abs_diff = abs(Xf.iloc[i].get("elo_diff", 0.0))
            goals_ewm_sum = float(Xf.iloc[i].get("home_gf_ewm", 1.3)) + float(Xf.iloc[i].get("away_gf_ewm", 1.1))
            market_draw = float(mx)

            # gate su favorito di mercato
            fav_min = float(getattr(booster, "favorite_prob_min", 0.60))
            skip_boost_if_fav = bool(getattr(booster, "skip_booster_if_favorite", True))
            is_strong_fav = max(mh, ma) >= fav_min

            can_boost = (
                elo_abs_diff <= float(getattr(booster, "elo_abs_diff_max", 35.0)) and
                goals_ewm_sum <= float(getattr(booster, "goals_ewm_sum_max", 2.6)) and
                market_draw >= float(getattr(booster, "market_draw_min", 0.28))
            )

            if can_boost and (skip_boost_if_fav and is_strong_fav):
                fav_gate_blocked = True
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
                draw_boosted = True

        # Near-tie promotion to draw (parametrico, coerente con cfg)
        nt_enabled = bool(getattr(booster, "promote_near_tie", False)) if booster else False
        if nt_enabled:
            tie_margin = float(getattr(booster, "tie_margin", 0.02))
            # gate su favorito di mercato
            fav_min = float(getattr(booster, "favorite_prob_min", 0.60)) if booster else 0.60
            skip_nt_if_fav = bool(getattr(booster, "skip_near_tie_if_favorite", True)) if booster else True
            is_strong_fav = max(mh, ma) >= fav_min
            # se X è entro tie_margin dal top1, promuovi X
            pvec = np.array([ph, px, pa], dtype=float)
            top_idx = int(np.argmax(pvec))

            if top_idx != 1:
                diff = float(pvec[top_idx] - px)
                if diff <= tie_margin:
                    if skip_nt_if_fav and is_strong_fav:
                        fav_gate_blocked = True
                    else:
                        # alza px al livello del top1 e rinormalizza
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
                        near_tie_promoted = True

        # Market guardrails: limita scostamento e blend verso mercato
        mg = getattr(getattr(cfg, "model", None), "market_guardrails", None)

        if (not use_learned_post) and mg and bool(getattr(mg, "enabled", False)):
            mh, mx, ma = _get_market_probs_row(i)
            flat = abs(mh - 1/3) + abs(mx - 1/3) + abs(ma - 1/3) < 1e-6

            # in EPL, se riga con quote stimate, salta guardrails
            is_estimated = False
            try:
                is_estimated = bool(est_used[i])
            except Exception:
                is_estimated = False

            if league == 'epl' and is_estimated:
                flat = True

            if flat:
                flat_market_skipped = True
            else:
                # clipping + blend (invariato)
                max_dh = float(getattr(mg, "max_abs_diff_home", 0.18))
                max_dx = float(getattr(mg, "max_abs_diff_draw", 0.14))
                max_da = float(getattr(mg, "max_abs_diff_away", 0.18))
                ph = np.clip(ph, mh - max_dh, mh + max_dh)
                px = np.clip(px, mx - max_dx, mx + max_dx)
                pa = np.clip(pa, ma - max_da, ma + max_da)
                # rinormalizza
                s = max(ph + px + pa, 1e-12)
                ph, px, pa = ph/s, px/s, pa/s
                # blend dolce verso mercato se oltre soglia
                bw = float(getattr(mg, "blend_weight", 0.5))
                if bw > 0:
                    ph = (1.0 - bw) * ph + bw * mh
                    px = (1.0 - bw) * px + bw * mx
                    pa = (1.0 - bw) * pa + bw * ma
                    s = max(ph + px + pa, 1e-12)
                    ph, px, pa = ph/s, px/s, pa/s
                guardrails_applied = True

        P_post[i, 0], P_post[i, 1], P_post[i, 2] = ph, px, pa
        r["explain_draw_boosted"] = bool(draw_boosted)
        r["explain_near_tie_promoted"] = bool(near_tie_promoted)
        r["explain_guardrails_applied"] = bool(guardrails_applied)
        r["explain_favorite_gate_blocked"] = bool(fav_gate_blocked)
        r["explain_flat_market_skipped"] = bool(flat_market_skipped)
        # nuova spiegazione: quote di mercato stimate
        try:
            r["explain_market_estimated"] = bool(est_used[i]) if i < len(est_used) else False
        except Exception:
            r["explain_market_estimated"] = False

    # --- FINAL CALIBRATION (ultima trasformazione) ---
    if final_cal_path.exists() and bool(getattr(getattr(cfg.model, "final_calibration", None), "enabled", False)):
        try:
            try:
                fcal = MultinomialLogisticCalibrator.load(str(final_cal_path))
            except Exception:
                fcal = OneVsRestIsotonic.load(str(final_cal_path))
            P_post = fcal.transform(P_post)
        except Exception as e:
            logger.warning(f"Final calibration transform failed: {e}")

    # overwrite final probabilities + recompute confidence/entropy
    for i, r in enumerate(probs):
        ph, px, pa = float(P_post[i, 0]), float(P_post[i, 1]), float(P_post[i, 2])
        r["p_home"], r["p_draw"], r["p_away"] = ph, px, pa
        r["prediction_confidence"] = max(ph, px, pa)
        r["prediction_entropy"] = -sum(p * np.log(p + 1e-10) for p in [ph, px, pa])
    
    result = pd.concat([fixtures.reset_index(drop=True), pd.DataFrame(probs)], axis=1)
    result['model_id'] = model_id
    result['prediction_timestamp'] = pd.Timestamp.now()
    
    logger.info(f"🚀 DataFrame prediction completed: {len(result)} predictions")
    return result