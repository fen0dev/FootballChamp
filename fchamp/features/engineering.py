import pandas as pd
import numpy as np
from fchamp.features.elo import Elo
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

def add_elo(
    df: pd.DataFrame,
    *,
    start: float,
    k: float,
    hfa: float,
    mov_factor: float,
    season_regression: float = 0.0,
    time_decay_days: float = 0.0,
    adaptive_k: bool = False,
    home_away_split: bool = False,
) -> pd.DataFrame:
    elo = Elo(start=start, k=k, hfa=hfa, mov_factor=mov_factor, adaptive_k=adaptive_k, home_away_split=home_away_split)
    # track last match date per team per time decay
    last_match_date = {}
    elo_home, elo_away = [], []
    df_sorted = df.sort_values("date").copy()
    df_sorted['date'] = pd.to_datetime(df_sorted['date'])

    # detect season breaks per regression automatica
    season_breaks = set()
    if season_regression > 0:
        for i in range(1, len(df_sorted)):
            current_date = df_sorted.iloc[i]['date']
            prev_date = df_sorted.iloc[i - 1]['date']
            # break > 60 giorni = nuova stagione
            if (current_date - prev_date).days > 60:
                season_breaks.add(current_date)

        if season_breaks:
            logger.info(f"🚀 Detected {len(season_breaks)} season breaks for ELO regression")

    for i, r in enumerate(df_sorted.itertuples(index=False)):
        h, a, hg, ag = r.home_team, r.away_team, int(r.home_goals), int(r.away_goals)
        match_date = r.date

        # apply season regression se siamo a inizio stagione
        if season_regression > 0 and match_date in season_breaks:
            for team in elo.ratings:
                current_rating = elo.ratings[team]
                elo.ratings[team] = start + (current_rating - start) * (1 - season_regression)
            logger.debug(f"Applied season regression (factor={season_regression}) at {match_date}")

        # apply time decay per inattivita'
        if time_decay_days > 0:
            for team in [h, a]:
                if team in last_match_date:
                    days_inactive = (match_date - last_match_date[team]).days
                    if days_inactive > 0:
                        decay_factor = np.exp(-days_inactive / time_decay_days)
                        current_rating = elo.get(team)
                        elo.ratings[team] = start + (current_rating - start) * decay_factor

        # get ratings PRE-match
        elo_home.append(elo.get(h))
        elo_away.append(elo.get(a))

        # update POST-match
        elo.update(h, a, hg, ag)

        # track last game date
        if time_decay_days > 0:
            last_match_date[h] = match_date
            last_match_date[a] = match_date

    # costruisci data frame
    out = df.copy()
    out["elo_home_pre"] = elo_home
    out["elo_away_pre"] = elo_away
    out["elo_diff"] = out["elo_home_pre"] - out["elo_away_pre"]

    return out

def add_rolling_form(df: pd.DataFrame, rolling_n: int, ewm_alpha: float, add_features: bool = False) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    df['date'] = pd.to_datetime(df['date'])

    def _roll(g: pd.Series) -> pd.Series:
        return g.shift().rolling(int(rolling_n), min_periods=1).mean()

    out = df.copy()

    out["home_gf_roll"] = out.groupby("home_team")["home_goals"].apply(_roll).reset_index(level=0, drop=True)
    out["home_ga_roll"] = out.groupby("home_team")["away_goals"].apply(_roll).reset_index(level=0, drop=True)
    out["away_gf_roll"] = out.groupby("away_team")["away_goals"].apply(_roll).reset_index(level=0, drop=True)
    out["away_ga_roll"] = out.groupby("away_team")["home_goals"].apply(_roll).reset_index(level=0, drop=True)

    # EWM as recent form
    out["home_gf_ewm"] = out.groupby("home_team")["home_goals"].apply(lambda s: s.shift().ewm(alpha=ewm_alpha, min_periods=1).mean()).reset_index(level=0, drop=True)
    out["away_gf_ewm"] = out.groupby("away_team")["away_goals"].apply(lambda s: s.shift().ewm(alpha=ewm_alpha, min_periods=1).mean()).reset_index(level=0, drop=True)

    if add_features:
        logger.info("🚀 Adding advanced form features")

        # separate home/away form
        def _roll_venue_specific(group, is_home=True):
            if is_home:
                gf = group['home_goals']
                ga = group['away_goals']
            else:
                gf = group['away_goals']
                ga = group['home_goals']

            return pd.DataFrame({
                'gf_venue': gf.shift().rolling(rolling_n, min_periods=1).mean(),
                'ga_venue': ga.shift().rolling(rolling_n, min_periods=1).mean(),
                'gd_venue': (gf - ga).shift().rolling(rolling_n, min_periods=1).mean()
            }, index=group.index)

        # Home venue form
        home_venue = out.groupby("home_team").apply(lambda g: _roll_venue_specific(g, True), include_groups=False).reset_index(level=0, drop=True)
        out["home_gf_venue"] = home_venue['gf_venue']
        out["home_ga_venue"] = home_venue['ga_venue'] 
        out["home_gd_venue"] = home_venue['gd_venue']
        
        # Away venue form
        away_venue = out.groupby("away_team").apply(lambda g: _roll_venue_specific(g, False), include_groups=False).reset_index(level=0, drop=True)
        out["away_gf_venue"] = away_venue['gf_venue']
        out["away_ga_venue"] = away_venue['ga_venue']
        out["away_gd_venue"] = away_venue['gd_venue']

        # schedule density
        def _calculate_rest_days(group):
            dates = pd.to_datetime(group['date'])
            rest_days = dates.diff().dt.days.fillna(7)  # 7 giorni default per prima partita
            return rest_days

        out['home_rest_days'] = out.groupby('home_team').apply(_calculate_rest_days, include_groups=False).reset_index(level=0, drop=True)
        out['away_rest_days'] = out.groupby('away_team').apply(_calculate_rest_days, include_groups=False).reset_index(level=0, drop=True)
        out['rest_advantage'] = out['home_rest_days'] - out['away_rest_days']

        # conto delle partite recenti (fatigue)
        def _games_in_window(group, days=14):
            dates = pd.to_datetime(group['date'])
            games_count = []

            for i, current_date in enumerate(dates):
                window_start = current_date - timedelta(days=days)
                recent_games = ((dates[:i] >= window_start) & (dates[:i] < current_date)).sum()
                games_count.append(recent_games)
            
            return pd.Series(games_count, index=group.index)

        out['home_games_14d'] = out.groupby('home_team').apply(lambda g: _games_in_window(g, 14), include_groups=False).reset_index(level=0, drop=True)
        out['away_games_14d'] = out.groupby('away_team').apply(lambda g: _games_in_window(g, 14), include_groups=False).reset_index(level=0, drop=True)
        out['fatigue_differential'] = out['away_games_14d'] - out['home_games_14d']  # Più alto = away più stanca

        logger.info(f"🚀 Added {len([c for c in out.columns if 'venue' in c or 'rest' in c or 'games_' in c])} advanced features")

    return out

def add_xg_real_features(df: pd.DataFrame, rolling_n: int = 6, ewm_alpha: float = 0.5) -> pd.DataFrame:
    """Aggiunge feature xG reali se presenti: rolling mean e EWM per squadra/venue.
    Richiede colonne 'home_xg' e 'away_xg'.
    """
    if not all(c in df.columns for c in ['home_xg','away_xg']):
        return df
    out = df.sort_values('date').copy()
    out['date'] = pd.to_datetime(out['date'])

    # rolling venue-specific xG
    out['home_xg_roll'] = (
        out.groupby('home_team')['home_xg']
        .transform(lambda s: s.shift().rolling(int(rolling_n), min_periods=1).mean())
    )
    out['away_xg_roll'] = (
        out.groupby('away_team')['away_xg']
        .transform(lambda s: s.shift().rolling(int(rolling_n), min_periods=1).mean())
    )

    # EWM xG (più reattivo)
    out['home_xg_ewm'] = out.groupby('home_team')['home_xg'].apply(lambda s: s.shift().ewm(alpha=ewm_alpha, min_periods=1).mean()).reset_index(level=0, drop=True)
    out['away_xg_ewm'] = out.groupby('away_team')['away_xg'].apply(lambda s: s.shift().ewm(alpha=ewm_alpha, min_periods=1).mean()).reset_index(level=0, drop=True)

    return out

def add_shots_real_features(df: pd.DataFrame, rolling_n: int = 6) -> pd.DataFrame:
    """ Crea rolling per tiri reali se presenti colonne normalizzate home_shots/away_shots (+ opzionale home_sot/away_sot) """
    req = ['home_shots', 'away_shots']

    if not all(c in df.columns for c in req):
        return df
    
    out = df.sort_values('date').copy()
    out['home_shots_roll'] = (
        out.groupby('home_team')['home_shots']
        .transform(lambda s: s.shift().rolling(int(rolling_n), min_periods=1).mean())
    )
    out['away_shots_roll'] = (
        out.groupby('away_team')['away_shots']
        .transform(lambda s: s.shift().rolling(int(rolling_n), min_periods=1).mean())
    )

    if 'home_sot' in out.columns and 'away_sot' in out.columns:
        out['home_shots_on_target_roll'] = (
            out.groupby('home_team')['home_sot']
            .transform(lambda s: s.shift().rolling(int(rolling_n), min_periods=1).mean())
        )
        out['away_shots_on_target_roll'] = (
            out.groupby('away_team')['away_sot']
            .transform(lambda s: s.shift().rolling(int(rolling_n), min_periods=1).mean())
        )

    return out

def build_features(df: pd.DataFrame, safe_fill: bool = True, include_advanced: bool = False) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    yh = df["home_goals"].astype(int)
    ya = df["away_goals"].astype(int)
    y_outcome = pd.Series(np.where(yh > ya, 0, np.where(yh < ya, 2, 1)), index=df.index, name="outcome")

    base_cols = [
        "elo_home_pre","elo_away_pre","elo_diff",
        "home_gf_roll","home_ga_roll",
        "away_gf_roll","away_ga_roll",
        "home_gf_ewm","away_gf_ewm",
    ]

    # 👇 Feature pro-X (pareggio): gap di forza ridotto e bassa somma attesa gol
    try:
        df["elo_abs_diff"] = (df["elo_diff"]).abs()
    except Exception:
        pass

    advanced_cols = []
    if include_advanced:
        potential_advanced = [
            "home_gf_venue", "home_ga_venue", "home_gd_venue",
            "away_gf_venue", "away_ga_venue", "away_gd_venue", 
            "rest_advantage", "fatigue_differential",
            "home_rest_days", "away_rest_days",
            "home_games_14d", "away_games_14d",
            "att_home_strength", "def_home_strength",
            "att_away_strength", "def_away_strength",
            "att_strength_diff",
            # shots (da HS/HST o merge_shots)
            "home_shots_roll","away_shots_roll",
            "home_shots_on_target_roll","away_shots_on_target_roll",
            "home_shot_efficiency","away_shot_efficiency",
            "home_shot_accuracy","away_shot_accuracy",
        ]

        advanced_cols = [col for col in potential_advanced if col in df.columns]
        if advanced_cols:
            logger.info(f"🚀 Including {len(advanced_cols)} advanced features: {advanced_cols}")
    
    # usa solo il segnale robusto dalle quote
    market_cols = [c for c in ["book_logit_diff"] if c in df.columns]

    if include_advanced and all(col in df.columns for col in ['book_p_home', 'book_p_draw', 'book_p_away']):
        enhanced_market = []

        # Market margin (close)
        if 'book_p_home' in df.columns:
            df['market_margin'] = df['book_p_home'] + df['book_p_draw'] + df['book_p_away'] - 1.0
            enhanced_market.append('market_margin')

        # Favorite strength
        df['favorite_prob'] = df[['book_p_home', 'book_p_away']].max(axis=1)
        df['favorite_edge'] = df[['book_p_home', 'book_p_away']].max(axis=1) - df[['book_p_home', 'book_p_away']].min(axis=1)
        enhanced_market.extend(['favorite_prob', 'favorite_edge'])

        # Draw tendency e supporto diretto al pareggio
        df['draw_tendency'] = df['book_p_draw'] / (df['book_p_home'] + df['book_p_away'])
        enhanced_market.append('draw_tendency')
        enhanced_market.append('book_p_draw')

        # Drift features (se disponibili)
        for c in ["prob_drift_home","prob_drift_draw","prob_drift_away","prob_drift_abs_sum",
                  "market_margin_open","market_margin_close",
                  "book_p_home_open","book_p_draw_open","book_p_away_open",
                  "book_p_home_close","book_p_draw_close","book_p_away_close"]:
            if c in df.columns:
                enhanced_market.append(c)

        market_cols.extend([col for col in enhanced_market if col in df.columns])

        if enhanced_market:
            logger.info(f"🚀 Added enhanced market features: {enhanced_market}")
    
    # Combina tutte le colonne
    # Costruiamo alcune feature sintetiche pro-draw
    synth_cols = []
    if 'elo_abs_diff' in df.columns:
        synth_cols.append('elo_abs_diff')
    if 'home_xg_roll' in df.columns and 'away_xg_roll' in df.columns:
        df['xg_total_roll'] = df['home_xg_roll'] + df['away_xg_roll']
        synth_cols.append('xg_total_roll')
    if 'home_gd_venue' in df.columns and 'away_gd_venue' in df.columns:
        df['gd_venue_abs_gap'] = (df['home_gd_venue'] - df['away_gd_venue']).abs()
        synth_cols.append('gd_venue_abs_gap')

    cols = base_cols + advanced_cols + market_cols + synth_cols
    
    # Filtra solo colonne esistenti
    existing_cols = [col for col in cols if col in df.columns]
    
    # 🚨 FIX CRITICO DATA LEAKAGE
    X = df[existing_cols].replace([np.inf, -np.inf], np.nan)
    
    if safe_fill:
        # 🚀 SAFE FILLING: Usa defaults ragionevoli
        safe_defaults = {
            # ELO defaults
            "elo_home_pre": 1500.0, "elo_away_pre": 1500.0, "elo_diff": 0.0,
            
            # Form defaults (media ragionevole)
            "home_gf_roll": 1.3, "home_ga_roll": 1.3, "away_gf_roll": 1.1, "away_ga_roll": 1.1,
            "home_gf_ewm": 1.3, "away_gf_ewm": 1.1,
            
            # Venue defaults
            "home_gf_venue": 1.4, "home_ga_venue": 1.2, "home_gd_venue": 0.2,
            "away_gf_venue": 1.0, "away_ga_venue": 1.4, "away_gd_venue": -0.4,
            
            # Schedule defaults
            "rest_advantage": 0.0, "fatigue_differential": 0.0,
            "home_rest_days": 7.0, "away_rest_days": 7.0,
            "home_games_14d": 1.0, "away_games_14d": 1.0,
            
            # Market defaults
            "book_logit_diff": 0.0, "market_margin": 0.05, 
            "favorite_prob": 0.4, "favorite_edge": 0.1, "draw_tendency": 1.0,
            # Pro-draw synthetics
            "elo_abs_diff": 0.0,
            "xg_total_roll": 2.4,
            "gd_venue_abs_gap": 0.0,
            "book_p_draw": 0.28,
        }
        
        # Applica defaults solo per colonne presenti
        for col in existing_cols:
            if col in safe_defaults:
                X[col] = X[col].fillna(safe_defaults[col])
            else:
                X[col] = X[col].fillna(0.0)  # Fallback generico
        
        logger.info("🚀 Applied safe filling (no data leakage)")
    else:
        # Metodo originale (per backward compatibility)
        X = X.ffill().fillna(0.0)
        logger.warning("⚠️  Using original .ffill() method (potential data leakage)")
    
    X = X.astype(float)
    
    # EXTREME VALUE CLIPPING per robustezza
    X = X.clip(lower=-15.0, upper=15.0)
    
    logger.info(f"🚀 Built feature matrix: {X.shape[1]} features for {X.shape[0]} samples")
    if len(existing_cols) != len(cols):
        missing = set(cols) - set(existing_cols)
        logger.warning(f"Missing features: {missing}")

    return X, y_outcome, yh, ya

def create_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features composite che combinano multiple statistiche per migliorare la previsione
    """
    df = df.copy()
    logger.info("🔧 Creating composite features")

    # attack-defense balance
    if all(col in df.columns for col in ['home_gf_roll', 'home_ga_roll', 'away_gf_roll', 'away_ga_roll']):
        df['home_balance_score'] = df['home_gf_roll'] - df['home_ga_roll']
        df['away_balance_score'] = df['away_gf_roll'] - df['away_ga_roll']
        df['balance_diff'] = df['home_balance_score'] - df['away_balance_score']

    # form momentum (cobina risultati recenti con xG proxy)
    if all(col in df.columns for col in ['home_xg_proxy', 'home_xg_roll', 'home_gf_roll']):
        df['home_form_momentum'] = (
            df['home_gf_roll'] * 0.4 +
            df['home_xg_roll'] * 0.3 +
            df.get('home_shot_quality_roll', df.get('home_shot_quality', 0.1)) * 10 * 0.3
        )

    if all(col in df.columns for col in ['away_xg_proxy', 'away_xg_roll', 'away_gf_roll']):
        df['away_form_momentum'] = (
            df['away_gf_roll'] * 0.4 +
            df['away_xg_roll'] * 0.3 +
            df.get('away_shot_quality_roll', df.get('away_shot_quality' ,0.1)) * 10 * 0.3
        )

    # pressure differential
    if all(col in df.columns for col in ['home_offensive_pressure', 'away_defensive_score']):
        df['pressure_differential'] = (
            df['home_offensive_pressure'] - df.get('away_defensive_score', 0) * 10
        )

    # expected outcome score
    if all(col in df.columns for col in ['elo_diff', 'home_dominance']):
        df['home_expected_score'] = (
            (df['elo_diff'] / 100) * 0.3 +
            df['home_dominance'] * 0.3 +
            (df.get('home_xg_proxy', 1.3) / 3) * 0.4
        )

    return df