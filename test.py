import sys
sys.path.append('.')

from fchamp.data.loader import load_matches
from fchamp.features.engineering import add_elo, add_rolling_form
from fchamp.features.advanced_stats import add_shots_and_corners_features, add_xg_proxy_features, add_advanced_proxy_features
from fchamp.features.engineering import create_composite_features
from fc.config import load_config

# Test
cfg = load_config("config_ita.yaml")
df = load_matches(cfg.data.paths)

print(f"Colonne disponibili: {df.columns.tolist()[:10]}...")
print(f"Colonna goals: {'home_goals' if 'home_goals' in df.columns else 'FTHG'}")

# Aggiungi features base prima
# CORREZIONE: cfg.elo, non cfg.features.elo
df = add_elo(df, **cfg.elo.model_dump())
df = add_rolling_form(df, cfg.features.rolling_n, cfg.features.ewm_alpha)

# Ora test features avanzate
print("\nTest features pipeline...")
df = add_shots_and_corners_features(df)
print(f"✓ Shots features: {[c for c in df.columns if 'shot' in c]}")

df = add_xg_proxy_features(df)
print(f"✓ xG features: {[c for c in df.columns if 'xg' in c]}")

df = add_advanced_proxy_features(df)
print(f"✓ Proxy features: {[c for c in df.columns if 'quality' in c or 'pressure' in c]}")

df = create_composite_features(df)
print(f"✓ Composite features: {[c for c in df.columns if 'momentum' in c or 'balance' in c]}")

print("\n✅ Tutte le features funzionano!")
print(f"Totale features create: {len(df.columns)}")