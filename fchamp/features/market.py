import pandas as pd
import numpy as np

PREF = ("PSCH", "PSCD", "PSCA")
FALL = ("B365H", "B365D", "B365A")

def _implied_probs(oh, od, oa):
    vals = [oh, od, oa]

    if any(v is None or pd.isna(v) or v <= 1e-9 for v in vals):
        return None, None, None
    
    inv = np.array([1 / oh, 1 / od, 1 / oa], dtype=float)

    return tuple((inv / inv.sum()).tolist())

def _pick_odds(row, prefer=PREF, fallback=FALL):
    ph = row.get(prefer[0], np.nan)
    pd_ = row.get(prefer[1], np.nan)
    pa = row.get(prefer[2], np.nan)

    if pd.isna(ph) or pd.isna(pd_) or pd.isna(pa):
        ph = row.get(fallback[0], np.nan); pd_ = row.get(fallback[1], np.nan); pa = row.get(fallback[2], np.nan)

    return ph, pd_, pa

def _standardize_date(df):
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce", format="mixed").dt.normalize()
    return df

def _normalize_team_cols(df):
    df = df.rename(columns={c: c.strip() for c in df.columns})
    return df

def load_market_subset(path: str, delimiter: str=",") -> pd.DataFrame:
    usecols = [
        "Date","Time",
        "HomeTeam","AwayTeam","Home","Away",
        "PSCH","PSCD","PSCA","B365H","B365D","B365A",
    ]

    df = pd.read_csv(
        path,
        delimiter=delimiter,
        engine="python",
        on_bad_lines="skip",
        usecols=lambda c: str(c) in usecols,
    )

    rename = {
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "Home": "home_team",
        "Away": "away_team"
    }
    
    df = df.rename(columns=rename)
    df = _standardize_date(df)

    def _row_probs(r):
        row = { **{k: r.get(k, np.nan) for k in ["PSCH","PSCD","PSCA","B365H","B365D","B365A"]} }
        # probs close (Pinnacle) e open (B365)
        ph_open, px_open, pa_open = _implied_probs(row.get("B365H"), row.get("B365D"), row.get("B365A"))
        ph_close, px_close, pa_close = _implied_probs(row.get("PSCH"), row.get("PSCD"), row.get("PSCA"))

        # preferisci close -> fallback open per "book_p_*" principale
        oh, od, oa = _pick_odds(row)
        ph_main, px_main, pa_main = _implied_probs(oh, od, oa)

        d = {
            "book_p_home": ph_main, "book_p_draw": px_main, "book_p_away": pa_main,
            "book_p_home_open": ph_open, "book_p_draw_open": px_open, "book_p_away_open": pa_open,
            "book_p_home_close": ph_close, "book_p_draw_close": px_close, "book_p_away_close": pa_close
        }

        # margini
        for suff in ["", "_open", "_close"]:
            keyh, keyd, keya = f"book_p_home{suff}", f"book_p_draw{suff}", f"book_p_away{suff}"
            if d.get(keyh) is not None:
                d[f"market_margin{suff}"] = float(d[keyh] + (d.get(keyd) or 0) + (d.get(keya) or 0) - 1.0)

        # drift (close - open)
        if ph_open is not None and ph_close is not None:
            d["prob_drift_home"] = float(ph_close - ph_open)
            d["prob_drift_draw"] = float((px_close or 0) - (px_open or 0))
            d["prob_drift_away"] = float(pa_close - pa_open)
            d["prob_drift_abs_sum"] = float(abs(d["prob_drift_home"]) + abs(d["prob_drift_draw"]) + abs(d["prob_drift_away"]))
        else:
            d["prob_drift_home"] = d["prob_drift_draw"] = d["prob_drift_away"] = d["prob_drift_abs_sum"] = 0.0

        return pd.Series(d)

    probs = df.apply(_row_probs, axis=1)
    out = pd.concat([df[["date","home_team","away_team"]], probs], axis=1)

    # cast numerico + fallback
    for c in [col for col in out.columns if col.startswith("book_p_")]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["book_p_home"] = out.get("book_p_home", 1/3).fillna(1/3)
    out["book_p_draw"] = out.get("book_p_draw", 1/3).fillna(1/3)
    out["book_p_away"] = out.get("book_p_away", 1/3).fillna(1/3)

    eps = 1e-9
    out["book_logit_diff"] = np.log((out["book_p_home"] + eps) / (out["book_p_away"] + eps))

    return out.dropna(subset=["date","home_team","away_team"]).reset_index(drop=True)

def add_market_features(df_matches: pd.DataFrame, paths: list[str], delimiter: str=",") -> pd.DataFrame:
    parts = []
    for p in paths:
        try:
            parts.append(load_market_subset(p, delimiter=delimiter))
        except Exception:
            continue
    if not parts:
        return df_matches

    mk = pd.concat(parts, axis=0, ignore_index=True)
    mk = mk.drop_duplicates(subset=["date","home_team","away_team"], keep="last")

    base = df_matches.copy()
    base["date"] = pd.to_datetime(base["date"], dayfirst=True, errors="coerce").dt.normalize()

    out = base.merge(mk, on=["date","home_team","away_team"], how="left")
    
    return out

def attach_market_to_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    df = fixtures.rename(columns={c: c.strip().lower() for c in fixtures.columns}).copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce", format="mixed").dt.normalize()
    
    have_psc = all(c in df.columns for c in ["psch","pscd","psca"])
    have_b365 = all(c in df.columns for c in ["b365h","b365d","b365a"])
    
    if have_psc or have_b365:
        def _r(r):
            row = {k.upper(): r.get(k, np.nan) for k in df.columns}
            # main probs
            oh, od, oa = _pick_odds(row)
            ph, px, pa = _implied_probs(oh, od, oa)
            # open/close
            ph_open, px_open, pa_open = _implied_probs(row.get("B365H"), row.get("B365D"), row.get("B365A"))
            ph_close, px_close, pa_close = _implied_probs(row.get("PSCH"), row.get("PSCD"), row.get("PSCA"))
            d = {
                "book_p_home": ph, "book_p_draw": px, "book_p_away": pa,
                "book_p_home_open": ph_open, "book_p_draw_open": px_open, "book_p_away_open": pa_open,
                "book_p_home_close": ph_close, "book_p_draw_close": px_close, "book_p_away_close": pa_close,
            }
            # margini
            for suff in ["", "_open", "_close"]:
                keyh, keyd, keya = f"book_p_home{suff}", f"book_p_draw{suff}", f"book_p_away{suff}"
                if d.get(keyh) is not None:
                    d[f"market_margin{suff}"] = float(d[keyh] + (d.get(keyd) or 0) + (d.get(keya) or 0) - 1.0)
            # drift
            if ph_open is not None and ph_close is not None:
                d["prob_drift_home"] = float(ph_close - ph_open)
                d["prob_drift_draw"] = float((px_close or 0) - (px_open or 0))
                d["prob_drift_away"] = float(pa_close - pa_open)
                d["prob_drift_abs_sum"] = float(abs(d["prob_drift_home"]) + abs(d["prob_drift_draw"]) + abs(d["prob_drift_away"]))
            else:
                d["prob_drift_home"] = d["prob_drift_draw"] = d["prob_drift_away"] = d["prob_drift_abs_sum"] = 0.0
            return pd.Series(d)
        probs = df.apply(_r, axis=1)
        df = pd.concat([df, probs], axis=1)
        
        for c in [col for col in ["book_p_home","book_p_draw","book_p_away"] if col in df.columns]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1/3)
    
    return df