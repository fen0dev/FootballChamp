import pandas as pd
import logging

logger = logging.getLogger(__name__)

def _get_goals_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Helper per ottenere i nomi corretti delle colonne goals"""
    if 'home_goals' in df.columns:
        return 'home_goals', 'away_goals'
    elif 'FTHG' in df.columns:
        return 'FTHG', 'FTAG'
    else:
        raise ValueError("No goals columns found in dataframe")

def add_shots_and_corners_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature basate su tiri e corner dai dati CSV
    """
    df = df.copy()
    
    # Verifica colonne disponibili
    shots_cols = ['HS', 'AS', 'HST', 'AST']  # Home/Away Shots (on Target)
    corners_cols = ['HC', 'AC']  # Home/Away Corners
    fouls_cols = ['HF', 'AF']  # Home/Away Fouls
    cards_cols = ['HY', 'AY', 'HR', 'AR']  # Yellow/Red cards
    
    # Calcola medie mobili per squadra
    if all(col in df.columns for col in shots_cols):
        logger.info("🎯 Adding shots-based features")
        
        # Shots efficiency (conversione tiri in gol)
        goals_home_col, goals_away_col = _get_goals_columns(df)
        df['home_shot_efficiency'] = df[goals_home_col] / (df['HS'] + 1)  # +1 per evitare divisione per 0
        df['away_shot_efficiency'] = df[goals_away_col] / (df['AS'] + 1)
        
        # Shots on target accuracy
        df['home_shot_accuracy'] = df['HST'] / (df['HS'] + 1)
        df['away_shot_accuracy'] = df['AST'] / (df['AS'] + 1)
        
        # Rolling averages per team
        for team_col, prefix in [('home_team', 'home'), ('away_team', 'away')]:
            # Shots rolling average
            df[f'{prefix}_shots_roll'] = (
                df.groupby(team_col)[f'{prefix[0].upper()}S']
                .transform(lambda x: x.shift().rolling(6, min_periods=1).mean())
            )
            
            # Shots on target rolling
            df[f'{prefix}_shots_on_target_roll'] = (
                df.groupby(team_col)[f'{prefix[0].upper()}ST']
                .transform(lambda x: x.shift().rolling(6, min_periods=1).mean())
            )
            
            # Shot efficiency rolling
            df[f'{prefix}_shot_eff_roll'] = (
                df.groupby(team_col)[f'{prefix}_shot_efficiency']
                .transform(lambda x: x.shift().rolling(6, min_periods=1).mean())
            )
    
    if all(col in df.columns for col in corners_cols):
        logger.info("⚽ Adding corner-based features")
        
        # Corner dominance
        df['corner_diff'] = df['HC'] - df['AC']
        
        # Rolling corner stats
        df['home_corners_roll'] = (
            df.groupby('home_team')['HC']
            .transform(lambda x: x.shift().rolling(6, min_periods=1).mean())
        )
        df['away_corners_roll'] = (
            df.groupby('away_team')['AC']
            .transform(lambda x: x.shift().rolling(6, min_periods=1).mean())
        )
    
    if all(col in df.columns for col in cards_cols):
        logger.info("🟨 Adding discipline features")
        
        # Disciplinary index (weighted cards)
        df['home_discipline_index'] = df['HY'] + df['HR'] * 3  # Red = 3 yellows
        df['away_discipline_index'] = df['AY'] + df['AR'] * 3
        
        # Rolling discipline
        df['home_discipline_roll'] = (
            df.groupby('home_team')['home_discipline_index']
            .transform(lambda x: x.shift().rolling(6, min_periods=1).mean())
        )
        df['away_discipline_roll'] = (
            df.groupby('away_team')['away_discipline_index']
            .transform(lambda x: x.shift().rolling(6, min_periods=1).mean())
        )
    
    return df


def add_referee_bias(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge feature per bias arbitrale (solo EPL ha referee data)
    """
    if 'Referee' not in df.columns:
        return df
    
    df = df.copy()
    logger.info("⚖️ Adding referee bias features")
    
    # Calcola statistiche per arbitro
    goals_home_col, goals_away_col = _get_goals_columns(df)
    ref_stats = df.groupby('Referee').agg({
        goals_home_col: 'mean',  # Media gol casa
        goals_away_col: 'mean',  # Media gol trasferta
        'HY': 'mean',     # Media cartellini gialli casa
        'AY': 'mean',     # Media cartellini gialli trasferta
        'FTR': lambda x: (x == 'H').mean()  # % vittorie casa
    }).rename(columns={
        goals_home_col: 'ref_avg_home_goals',
        goals_away_col: 'ref_avg_away_goals',
        'HY': 'ref_avg_home_cards',
        'AY': 'ref_avg_away_cards',
        'FTR': 'ref_home_win_rate'
    })
    
    # Merge con dataset principale
    df = df.merge(ref_stats, left_on='Referee', right_index=True, how='left')
    
    # Fill missing con medie generali
    for col in ref_stats.columns:
        df[col] = df[col].fillna(df[col].mean())
    
    return df

def add_head_to_head_stats(df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """
    Statistiche H2H robuste (niente dipendenza da FTR/Res): calcola esiti da gol.
    """
    if not all(c in df.columns for c in ['home_team','away_team','date']):
        return df

    # mappa colonne gol
    if 'home_goals' in df.columns and 'away_goals' in df.columns:
        gh, ga = 'home_goals', 'away_goals'
    elif 'FTHG' in df.columns and 'FTAG' in df.columns:
        gh, ga = 'FTHG', 'FTAG'
    else:
        return df

    dfx = df.copy()
    dfx['date'] = pd.to_datetime(dfx['date'], errors='coerce')

    h2h_stats = []
    for i, row in dfx.iterrows():
        home, away, date = row['home_team'], row['away_team'], row['date']
        if pd.isna(date):
            h2h_stats.append({
                'h2h_home_wins': 0.33, 'h2h_away_wins': 0.33, 'h2h_draws': 0.34,
                'h2h_home_goals_avg': 1.3, 'h2h_away_goals_avg': 1.1, 'h2h_matches_count': 0
            })
            continue

        h2h = dfx[
            (((dfx['home_team'] == home) & (dfx['away_team'] == away)) |
             ((dfx['home_team'] == away) & (dfx['away_team'] == home)))
            & (dfx['date'] < date)
        ].sort_values('date').tail(n_matches)

        if len(h2h) > 0:
            # vittorie del team 'home' in qualunque venue
            home_wins = (
                ((h2h['home_team'] == home) & (h2h[gh] > h2h[ga])).sum() +
                ((h2h['away_team'] == home) & (h2h[ga] > h2h[gh])).sum()
            )
            away_wins = (
                ((h2h['home_team'] == away) & (h2h[gh] > h2h[ga])).sum() +
                ((h2h['away_team'] == away) & (h2h[ga] > h2h[gh])).sum()
            )
            draws = (h2h[gh] == h2h[ga]).sum()

            home_goals_h2h = h2h.loc[h2h['home_team'] == home, gh].mean()
            away_goals_h2h = h2h.loc[h2h['away_team'] == away, ga].mean()

            h2h_stats.append({
                'h2h_home_wins': home_wins / len(h2h),
                'h2h_away_wins': away_wins / len(h2h),
                'h2h_draws': draws / len(h2h),
                'h2h_home_goals_avg': (home_goals_h2h if not pd.isna(home_goals_h2h) else 1.3),
                'h2h_away_goals_avg': (away_goals_h2h if not pd.isna(away_goals_h2h) else 1.1),
                'h2h_matches_count': len(h2h)
            })
        else:
            h2h_stats.append({
                'h2h_home_wins': 0.33, 'h2h_away_wins': 0.33, 'h2h_draws': 0.34,
                'h2h_home_goals_avg': 1.3, 'h2h_away_goals_avg': 1.1, 'h2h_matches_count': 0
            })

    h2h_df = pd.DataFrame(h2h_stats, index=dfx.index)
    return pd.concat([df, h2h_df], axis=1)

def add_xg_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea proxy per Expected Goals basato su shots e shots on target
    """
    if not all(col in df.columns for col in ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF']):
        logger.warning("⚠️ Missing required columns for advanced proxy features")
        return df
    
    df = df.copy()
    logger.info("📊 Adding xG proxy features")
    
    # Formula empirica per xG proxy
    # Shots on target valgono di più dei tiri normali
    goals_home_col, goals_away_col = _get_goals_columns(df)
    df['home_xg_proxy'] = df['HST'] * 0.35 + (df['HS'] - df['HST']) * 0.05
    df['away_xg_proxy'] = df['AST'] * 0.35 + (df['AS'] - df['AST']) * 0.05
    
    # xG over/underperformance
    df['home_xg_overperformance'] = df[goals_home_col] - df['home_xg_proxy']
    df['away_xg_overperformance'] = df[goals_away_col] - df['away_xg_proxy']
    
    # Rolling xG stats
    for team_col, prefix in [('home_team', 'home'), ('away_team', 'away')]:
        df[f'{prefix}_xg_roll'] = (
            df.groupby(team_col)[f'{prefix}_xg_proxy']
            .transform(lambda x: x.shift().rolling(6, min_periods=1).mean())
        )
        
        df[f'{prefix}_xg_overperf_roll'] = (
            df.groupby(team_col)[f'{prefix}_xg_overperformance']
            .transform(lambda x: x.shift().rolling(6, min_periods=1).mean())
        )
    
    # shots quality index (quanto efficaci sono i tiri)
    df['home_shot_quality'] = df['HST'] / (df['HS'] + 1)
    df['away_shot_quality'] = df['AST'] / (df['AS'] + 1)

    # offensive pressure
    df['home_offensive_pressure'] = (
        df['AS'] * 0.4 +
        df['AC'] * 0.3 +
        df['HST'] * 0.3
    )

    df['away_offensive_pressure'] = (
        df['AS'] * 0.4 +
        df['AC'] * 0.3 +
        df['AST'] * 0.3
    )

    # defensive resistance score
    # basato su tiri concessi vs media storica
    home_shots_conceded_avg = df['AS'].rolling(10, min_periods=3).mean()
    away_shots_conceded_avg = df['HS'].rolling(10, min_periods=3).mean()

    df['home_defensive_score'] = 1 - (df['AS'] / (home_shots_conceded_avg + 1))
    df['away_defensive_score'] = 1 - (df['HS'] / (away_shots_conceded_avg + 1))

    # match tempo (ritmo della partita)
    df['match_tempo'] = df['HF'] + df['AF'] + df['HC'] + df['AC']
    df['match_intensity'] = df['HS'] + df['AS'] + df['match_tempo'] / 10

    # home / away dominance score
    df['home_dominance'] = (
        (df['HS'] / (df['HS'] + df['AS'] + 1)) * 0.35 +
        (df['HC'] / (df['HC'] + df['AC'] + 1)) * 0.25 +
        (df['HST'] / (df['HST'] + df['AST'] + 1)) * 0.40
    )

    df['away_dominance'] = 1 - df['home_dominance']

    # expected points basato su dominanza (proxy)
    df['home_xpoints'] = (
        df['home_dominance'] * 2.5 +
        (df['home_shot_quality'] > 0.3) * 0.5
    )

    df['away_xpoints'] = (
        df['away_dominance'] * 2.5 +
        (df['away_shot_quality'] > 0.3) * 0.5
    )

    # momentum shift indicator
    # confronta prima e seconda metà dei tiri
    goals_home_col, goals_away_col = _get_goals_columns(df)
    df['home_momentum'] = df[goals_home_col] - df['HTHG']
    df['away_momentum'] = df[goals_away_col] - df['HTAG']

    # danger zone activity (azioni pericolose)
    df['home_danger_index'] = (
        df['HST'] * 0.5 +
        df['HC'] * 0.3 +
        (df['HF'] > 15) * 0.2
    )

    df['away_danger_index'] = (
        df['AST'] * 0.5 +
        df['AC'] * 0.3 +
        (df['AF'] > 15) * 0.2
    )

    # rolling stats per team
    for team_col, prefix in [('home_team', 'home'), ('away_team', 'away')]:
        #rolling offensive pressure
        df[f'{prefix}_off_pressure_roll'] = (
            df.groupby(team_col)[f'{prefix}_offensive_pressure']
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )

        # rolling shots quality
        df[f'{prefix}_shot_quality_roll'] = (
            df.groupby(team_col)[f'{prefix}_shot_quality']
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )

        # rolling danger index
        df[f'{prefix}_danger_roll'] = (
            df.groupby(team_col)[f'{prefix}_danger_index']
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )

    return df

def add_advanced_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features proxy avanzate per migliorare predizioni senza xG reali
    """
    if not all(col in df.columns for col in ['HS', 'AS', 'HST', 'AST', 'HC', 'AC']):
        logger.warning("⚠️ Missing required columns for advanced proxy features")
        return df
    
    df = df.copy()
    logger.info("🚀 Adding advanced proxy features")
    
    # 1. Shot Quality Index (quanto sono efficaci i tiri)
    goals_home_col, goals_away_col = _get_goals_columns(df)
    df['home_shot_quality'] = df['HST'] / (df['HS'] + 1)  # +1 evita divisione per zero
    df['away_shot_quality'] = df['AST'] / (df['AS'] + 1)
    
    # 2. Offensive Pressure (pressione offensiva combinata)
    df['home_offensive_pressure'] = (
        df['HS'] * 0.4 +           # Tiri pesano 40%
        df['HC'] * 0.3 +           # Corner 30%
        df['HST'] * 0.3            # Tiri in porta 30%
    )
    df['away_offensive_pressure'] = (
        df['AS'] * 0.4 + 
        df['AC'] * 0.3 + 
        df['AST'] * 0.3
    )
    
    # 3. Dominance Score
    df['home_dominance'] = (
        (df['HS'] / (df['HS'] + df['AS'] + 1)) * 0.35 +      # Dominio tiri
        (df['HC'] / (df['HC'] + df['AC'] + 1)) * 0.25 +      # Dominio corner
        (df['HST'] / (df['HST'] + df['AST'] + 1)) * 0.40     # Dominio tiri in porta
    )
    df['away_dominance'] = 1 - df['home_dominance']
    
    # 4. Danger Index (azioni pericolose)
    df['home_danger_index'] = (
        df['HST'] * 0.5 +           # Tiri in porta
        df['HC'] * 0.3 +            # Corner
        (df['HF'] > 15) * 0.2       # Molti falli subiti = pressione
    )
    df['away_danger_index'] = (
        df['AST'] * 0.5 + 
        df['AC'] * 0.3 + 
        (df['AF'] > 15) * 0.2
    )
    
    # Rolling stats per team
    for team_col, prefix in [('home_team', 'home'), ('away_team', 'away')]:
        # Rolling shot quality
        df[f'{prefix}_shot_quality_roll'] = (
            df.groupby(team_col)[f'{prefix}_shot_quality']
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )
        
        # Rolling offensive pressure
        df[f'{prefix}_off_pressure_roll'] = (
            df.groupby(team_col)[f'{prefix}_offensive_pressure']
            .transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
        )
    
    return df

def add_team_strength_features(df: pd.DataFrame, ewm_alpha: float = 0.4) -> pd.DataFrame:
    """
    Stima semplici strength di attacco/difesa per squadra (split casa/trasferta) con EWM su gol.
    """
    df = df.copy()
    df = df.sort_values("date")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Home perspective
    df['att_home_strength'] = (
        df.groupby('home_team')['home_goals']
        .transform(lambda s: s.shift().ewm(alpha=ewm_alpha, min_periods=1).mean())
    )
    df['def_home_strength'] = (
        df.groupby('home_team')['away_goals']
        .transform(lambda s: s.shift().ewm(alpha=ewm_alpha, min_periods=1).mean())
    )

    # Away perspective
    df['att_away_strength'] = (
        df.groupby('away_team')['away_goals']
        .transform(lambda s: s.shift().ewm(alpha=ewm_alpha, min_periods=1).mean())
    )
    df['def_away_strength'] = (
        df.groupby('away_team')['home_goals']
        .transform(lambda s: s.shift().ewm(alpha=ewm_alpha, min_periods=1).mean())
    )

    # Differenze utili
    df['att_strength_diff'] = df['att_home_strength'] - df['att_away_strength']
    df['def_strength_diff'] = df['def_home_strength'] - df['def_away_strength']

    return df
