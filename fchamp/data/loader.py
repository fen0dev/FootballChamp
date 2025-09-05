import csv
import re
import time
import requests
import json
from datetime import datetime
from typing import List
import pandas as pd

REQUIRED = ["date", "home_team", "away_team", "home_goals", "away_goals"]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    m = {}

    if "date" in cols:
        m["date"] = cols["date"]
    if "hometeam" in cols:
        m["home_team"] = cols["hometeam"]
    if "awayteam" in cols:
        m["away_team"] = cols["awayteam"]
    if "fthg" in cols:
        m["home_goals"] = cols["fthg"]
    if "ftag" in cols:
        m["away_goals"] = cols["ftag"]
    # supporto Superliga Danese con headers diversi
    if "home" in cols:
        m["home_team"] = cols["home"]
    if "away" in cols:
        m["away_team"] = cols["away"]
    if "hg" in cols:
        m["home_goals"] = cols["hg"]
    if "ag" in cols:
        m["away_goals"] = cols["ag"]

    for k in REQUIRED:
        if k not in m and k in df.columns:
            m[k] = k

    out = df.rename(columns={v: k for k, v in m.items()})
    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce", format="mixed")

    return out.sort_values("date").reset_index(drop=True)

_time_re = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$")

def _is_time_token(s: str) -> bool:
    return bool(_time_re.fullmatch(str(s).strip()))

def _to_int(s):
    try:
        return int(str(s).strip())
    except Exception:
        return None

def _extract_record(row: list[str]):
    # Aspettativa base: Div, Date, [Time?], HomeTeam, AwayTeam, FTHG, FTAG, ...
    if len(row) < 6:
        return None
    date_str = row[1].strip()
    pos = 2
    if len(row) > 2 and _is_time_token(row[2]):
        pos = 3  # salta il Time se presente
    if len(row) <= pos + 1:
        return None
    home = row[pos].strip()
    away = row[pos + 1].strip()

    # Cerca i primi due interi dopo home/away per FTHG/FTAG
    fthg = _to_int(row[pos + 2]) if len(row) > pos + 2 else None
    ftag = _to_int(row[pos + 3]) if len(row) > pos + 3 else None
    if fthg is None or ftag is None:
        fthg = ftag = None
        for j in range(pos + 2, len(row) - 1):
            a = _to_int(row[j])
            b = _to_int(row[j + 1])
            if a is not None and b is not None:
                fthg, ftag = a, b
                break
    if fthg is None or ftag is None:
        return None
    return {
        "date": date_str,
        "home_team": home,
        "away_team": away,
        "home_goals": fthg,
        "away_goals": ftag,
    }

def load_matches(paths: List[str], delimiter: str = ",") -> pd.DataFrame:
    def _header_known(h: list[str]) -> bool:
        h = [c.strip().lower() for c in h]

        return {"hometeam", "awayteam", "fthg", "ftag"}.issubset(h) or {"home", "away", "hg", "ag"}.issubset(h)

    dfs = []
    records = []
    
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader, None)  # ignora header
            
            if header and _header_known(header):
                dfp = pd.read_csv(p, delimiter=delimiter, engine="python", on_bad_lines="skip")
                dfs.append(normalize_columns(dfp))
            else:
                for row in reader:
                    if not row:
                        continue
                    rec = _extract_record(row)
                
                    if rec:
                        records.append(rec)
    
    if records:
        df_rec = pd.DataFrame.from_records(records, columns=REQUIRED)
        df_rec["date"] = pd.to_datetime(df_rec["date"], dayfirst=True, errors="coerce", format="mixed")
        
        dfs.append(df_rec)

    out = pd.concat(dfs, axis=0, ignore_index=True) if dfs else pd.DataFrame(columns=REQUIRED)
    out = out.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    return out

# --------- Web ingestion (Understat) opzionale ---------

def fetch_understat_results(league: str = "Serie A", seasons: List[int] = None) -> pd.DataFrame:
    """
    Scarica risultati e xG da Understat per stagioni specificate.
    Restituisce DataFrame normalizzato con colonne: date, home_team, away_team, xG, goals.
    """
    seasons = seasons or [datetime.now().year - 1]
    # Semplice adapter HTTP: usa l'API pubblica JSON non ufficiale di Understat (mirror noto)
    # Nota: endpoint soggetto a cambio; gestiamo fallback silenzioso.
    def _fetch(league: str, year: int):
        url = f"https://understatapi.superagent.one/league/{league.replace(' ', '%20')}/{year}"
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
        except Exception:
            return []
        return []

    raw = [_fetch(league, y) for y in seasons]
    matches = []
    for season_data in raw:
        for m in season_data:
            try:
                d = {
                    'date': pd.to_datetime(m['date']).normalize(),
                    'home_team': m['h']['title'],
                    'away_team': m['a']['title'],
                    'home_goals': int(m['goals']['h']),
                    'away_goals': int(m['goals']['a']),
                    'home_xg': float(m['xG']['h']),
                    'away_xg': float(m['xG']['a']),
                }
                matches.append(d)
            except Exception:
                continue

    df = pd.DataFrame(matches)
    if not df.empty:
        df = df.sort_values('date').reset_index(drop=True)
    return df

def _collect_match_headers_from_obj(obj) -> list[dict]:
    rows = []

    def pick(k, d, fallback=None):
        return d.get(k) if isinstance(d, dict) else fallback

    if isinstance(obj, dict):
        # oggetti 'match' con id + nomi squadre + data
        if 'id' in obj and (('h' in obj and isinstance(obj['h'], dict)) or 'home_team' in obj or 'h_title' in obj):
            mid = obj.get('id')
            home = (pick('title', obj.get('h', {})) or obj.get('home_team') or obj.get('h_title') or obj.get('team_h'))
            away = (pick('title', obj.get('a', {})) or obj.get('away_team') or obj.get('a_title') or obj.get('team_a') or obj.get('a_team'))
            date_raw = obj.get('datetime') or obj.get('date') or obj.get('kickOff') or obj.get('time')
            date = str(date_raw)[:10] if date_raw else None

            if mid and home and away and date:
                try:
                    rows.append({'id': int(mid), 'date': date, 'home_team': str(home), 'away_team': str(away)})
                except Exception:
                    pass

        for v in obj.values():
            rows.extend(_collect_match_headers_from_obj(v))
    elif isinstance(obj, list):
        for it in obj:
            rows.extend(_collect_match_headers_from_obj(it))

    return rows

def _extract_match_shots_from_html(html: str) -> tuple[int, int, int, int] | None:
    # 1 - prova a prendere direttamente shotsData = JSON.parse('...')
    rx_shots = re.compile(r"shotsData\s*=\s*JSON.parse\((\"|')(.*?)(\1)\)", re.S)
    target = {'Goal', 'SavedShot'}  # definizione SOT
    m = rx_shots.search(html)
    candidates = []

    if m:
        try:
            s = m.group(2).encode('utf-8').decode('unicode_escape')
            data = json.loads(s)
            candidates.append(data)
        except Exception:
            pass
    
    # 2 - Fallback: prova tutti i JSON.parse('...')
    if not candidates:
        rx_all = re.compile(r"JSON.parse\((\"|')(.*?)(\1)\)", re.S)

        for mm in rx_all.finditer(html):
            try:
                s = mm.group(2).encode('utf-8').decode('unicode_escape')
                data = json.loads(s)
                candidates.append(data)
            except Exception:
                pass

    # 3 - estrai conteggi tiri e tiri in porta
    def agg_from_data(d) -> tuple[int, int, int, int] | None:
        try:
            # caso A: dict con chiavi 'h'  e 'a' che sono liste di eventi
            if isinstance(d, dict) and isinstance(d.get('h'), list) and isinstance(d.get('a'), list):
                H, A = d['h'], d['a']
                hs = len(H)
                as_ = len(A)
                hst = sum(1 for ev in H if isinstance(ev, dict) and ev.get('result') in target)
                ast = sum(1 for ev in A if isinstance(ev, dict) and ev.get('result') in target)
                return hs, hst, as_, ast
            
            # caso B: lista di eventi con 'h_a'
            if isinstance(d, list) and all(isinstance(ev, dict) for ev in d) and any('h_a' in ev for ev in d):
                H = [ev for ev in d if ev.get('h_a') == 'h']
                A = [ev for ev in d if ev.get('h_a') == 'a']
                hs = len(H)
                as_ = len(A)
                hst = sum(1 for ev in H if ev.get('result') in target)
                ast = sum(1 for ev in A if ev.get('result') in target)
                return hs, hst, as_, ast

        except Exception:
            return None

        return None

    for can in candidates:
        res = agg_from_data(can)
        if res:
            return res
    return None

def fetch_understat_league_shots_scrape(league: str = "Serie A", seasons: List[int] | None = None, *, progress: bool = False, limit: int | None = None) -> pd.DataFrame:
    """Scrape understat.com per ricavare tiri (HS/HST) match-level:
       ritorna: date, home_team, away_team, home_shots, away_shots, home_sot, away_sot
    """
    seasons = seasons or [datetime.now().year - 1]
    slug = _league_to_slug(league)

    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://understat.com/"
    }

    match_rows = []
    # 1 - raccogli tutti i match id dal page-level
    rx_all = re.compile(r"JSON.parse\((\"|')(.*?)(\1)\)", re.S)
    rx_next = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)

    for y in seasons:
        url = f"https://understat.com/league/{slug}/{y}"
        
        try:
            html = requests.get(url, timeout=30, headers=headers).text
        except Exception:
            continue
        
        found_any = False
        for m in rx_all.finditer(html):
            enc = m.group(2)
            try:
                s = enc.encode('utf-8').decode('unicode_escape')
                data = json.loads(s)
            except Exception:
                continue

            part = _collect_match_headers_from_obj(data)
            if part:
                if isinstance(part, list):
                    match_rows.extend(part)
                elif isinstance(part, dict):
                    match_rows.append(part)
                found_any = True

        if not found_any:
            m = rx_next.search(html)
            if m:
                try:
                    s = m.group(1)
                    data = json.loads(s)
                    part = _collect_match_headers_from_obj(data)
                    if part:
                        if isinstance(part, list):
                            match_rows.extend(part)
                        elif isinstance(part, dict):
                            match_rows.append(part)
                except Exception:
                    pass

    # dedup per id
    seen = set()
    headers_rows = []

    for r in match_rows:
        if not isinstance(r, dict):
            continue
        mid = r.get('id')
        if mid and mid not in seen:
            seen.add(mid)
            headers_rows.append(r)

    # 2 - visita ogni match per estrarre shots
    out = []
    total = len(headers_rows)
    if isinstance(limit, int) and limit > 0:
        headers_rows = headers_rows[:limit]
    n_fetch = len(headers_rows)
    if progress:
        try:
            print(f"[shots] Trovati {total} match, scarico {n_fetch} pagine...")
        except Exception:
            pass
    for i, r in enumerate(headers_rows, start=1):
        mid = r['id']
        m_url = f"https://understat.com/match/{mid}"

        try:
            m_html = requests.get(m_url, timeout=30, headers=headers).text
            stats = _extract_match_shots_from_html(m_html)
            if stats:
                hs, hst, as_, ast = stats
                out.append({
                    'date': r['date'],
                    'home_team': r['home_team'],
                    'away_team': r['away_team'],
                    'home_shots': hs,
                    'away_shots': as_,
                    'home_sot': hst,
                    'away_sot': ast,
                })
            if progress and (i == 1 or i == n_fetch or i % 25 == 0):
                try:
                    print(f"[shots] {i}/{n_fetch} completati (match id {mid})")
                except Exception:
                    pass
            time.sleep(0.25) # throttle gentile per rate-limit
            
        except Exception:
            continue

    df = pd.DataFrame(out).drop_duplicates()
    if not df.empty:
        df = df.sort_values('date').reset_index(drop=True)
    
    return df

def merge_xg_into_history(history: pd.DataFrame, xg_df: pd.DataFrame) -> pd.DataFrame:
    """Effettua il merge robusto degli xG reali (Understat adapter) sullo storico.
    Join su data normalizzata e nomi squadra normalizzati; fallback fuzzy.
    """
    if xg_df is None or xg_df.empty:
        return history

    df = history.copy()
    xg = xg_df.copy()
    # normalizza date
    df['date_norm'] = pd.to_datetime(df['date']).dt.date
    xg['date_norm'] = pd.to_datetime(xg['date']).dt.date

    # normalizza nomi squadra (semplice)
    def _norm(s: str) -> str:
        s = str(s)
        s = s.lower().replace('.', ' ').replace('-', ' ').replace("'", ' ')
        for suf in [' fc',' afc',' cf',' bk',' if',' fk',' sk',' sc']:
            if s.endswith(suf): s = s[:-len(suf)]
        return ' '.join(s.split())

    for col in ['home_team','away_team']:
        df[col + '_key'] = df[col].map(_norm)
        xg[col + '_key'] = xg[col].map(_norm)

    # merge diretto
    merged = df.merge(
        xg[['date_norm','home_team_key','away_team_key','home_xg','away_xg']],
        on=['date_norm','home_team_key','away_team_key'], how='left'
    )

    # se ancora mancanti, tentativo swap venue
    mask_missing = merged['home_xg'].isna() | merged['away_xg'].isna()
    if mask_missing.any():
        alt = df.merge(
            xg[['date_norm','home_team_key','away_team_key','home_xg','away_xg']],
            left_on=['date_norm','home_team_key','away_team_key'],
            right_on=['date_norm','away_team_key','home_team_key'], how='left', suffixes=('','_alt')
        )
        merged.loc[mask_missing, 'home_xg'] = alt.loc[mask_missing, 'away_xg']
        merged.loc[mask_missing, 'away_xg'] = alt.loc[mask_missing, 'home_xg']

    # drop helper
    merged = merged.drop(columns=['date_norm','home_team_key','away_team_key'], errors='ignore')
    return merged

def merge_shots_into_history(history: pd.DataFrame, shots_df: pd.DataFrame) -> pd.DataFrame:
    if shots_df is None or shots_df.empty:
        return history

    df = history.copy()
    s = shots_df.copy()
    s.columns = [c.lower() for c in s.columns]

    df['date_norm'] = pd.to_datetime(df['date']).dt.date
    s['date_norm'] = pd.to_datetime(s['date']).dt.date

    def _norm(t: str) -> str:
        t = str(t).lower().replace('.', ' ').replace('-', ' ').replace("'", ' ')
        for suf in [' fc',' afc',' cf',' bk',' if',' fk',' sk',' sc']:
            if t.endswith(suf):
                t = t[:-len(suf)]
        return ' '.join(t.split())

    for col in ['home_team','away_team']:
        df[col + '_key'] = df[col].map(_norm)
        s[col + '_key'] = s[col].map(_norm)

    keep_cols = ['date_norm','home_team_key','away_team_key']
    vals = []
    if 'home_shots' in s.columns and 'away_shots' in s.columns:
        vals += ['home_shots', 'away_shots']
    if 'home_sot' in s.columns and 'away_sot' in s.columns:
        vals += ['home_sot', 'away_sot']

    merged = df.merge(s[keep_cols + vals], on=['date_norm','home_team_key','away_team_key'], how='left')
    miss = merged[vals[0]].isna() if vals else pd.Series(False, index=merged.index)

    if miss.any():
        alt = df.merge(
            s[keep_cols + vals],
            left_on=['date_norm','home_team_key','away_team_key'],
            right_on=['date_norm','away_team_key','home_team_key'],
            how='left', suffixes=('','_alt')
        )
        if 'home_shots' in vals:
            merged.loc[miss, 'home_shots'] = alt.loc[miss, 'away_shots']
            merged.loc[miss, 'away_shots'] = alt.loc[miss, 'home_shots']
        if 'home_sot' in vals:
            merged.loc[miss, 'home_sot'] = alt.loc[miss, 'away_sot']
            merged.loc[miss, 'away_sot'] = alt.loc[miss, 'home_sot']

    merged = merged.drop(columns=['date_norm','home_team_key','away_team_key'], errors='ignore')
    return merged

def _league_to_slug(name: str) -> str:
    n = name.strip().lower()
    mapping = {
        'serie a': 'Serie_A',
        'seria a': 'Serie_A',
        'premier league': 'EPL',
        'epl': 'EPL',
        'bundesliga': 'Bundesliga',
        'germany bundesliga': 'Bundesliga',
    }
    return mapping.get(n, name.replace(' ', '_'))


def _collect_matches_from_obj(obj) -> list[dict]:
    rows = []
    def as_float(x):
        try:
            return float(x)
        except Exception:
            return None
    if isinstance(obj, dict):
        # pattern 1: h/a come dict con title; xG come dict con h/a
        if (('h' in obj and isinstance(obj.get('h'), dict) and 'title' in obj['h']) or 'h_title' in obj or 'home_team' in obj) \
           and (('a' in obj and isinstance(obj.get('a'), dict) and 'title' in obj['a']) or 'a_title' in obj or 'away_team' in obj):
            # estrai campi possibili
            date_raw = obj.get('datetime') or obj.get('date') or obj.get('kickOff') or obj.get('time')
            if date_raw:
                date = str(date_raw)[:10]
            else:
                date = None
            home = (obj.get('h_title') or obj.get('home_team') or (obj.get('h') or {}).get('title') or obj.get('team_h'))
            away = (obj.get('a_title') or obj.get('away_team') or (obj.get('a') or {}).get('title') or obj.get('team_a'))
            # xG possibili
            hxg = obj.get('xg_h') or obj.get('xG_h') or (obj.get('xG') or {}).get('h') or obj.get('xg_home')
            axg = obj.get('xg_a') or obj.get('xG_a') or (obj.get('xG') or {}).get('a') or obj.get('xg_away')
            if hxg is None and 'xg' in obj and isinstance(obj['xg'], dict):
                hxg = obj['xg'].get('h')
            if axg is None and 'xg' in obj and isinstance(obj['xg'], dict):
                axg = obj['xg'].get('a')
            hxg = as_float(hxg)
            axg = as_float(axg)
            if date and home and away and hxg is not None and axg is not None:
                rows.append({
                    'date': date,
                    'home_team': home,
                    'away_team': away,
                    'home_xg': hxg,
                    'away_xg': axg,
                })
        # ricorsione su valori
        for v in obj.values():
            rows.extend(_collect_matches_from_obj(v))
    elif isinstance(obj, list):
        for it in obj:
            rows.extend(_collect_matches_from_obj(it))
    return rows

def fetch_understat_league_xg_scrape(league: str = "Serie A", seasons: List[int] | None = None) -> pd.DataFrame:
    """Scrape diretto da understat.com/league/{slug}/{season}.
    Estrae JSON embeddati e produce righe match-level con xG se trovati.
    """
    seasons = seasons or [datetime.now().year - 1]
    slug = _league_to_slug(league)
    rows: list[dict] = []

    rx_all = re.compile(r"JSON.parse\((\"|')(.*?)(\1)\)", re.S)
    rx_next = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', re.S)
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://understat.com/"
    }

    for y in seasons:
        url = f"https://understat.com/league/{slug}/{y}"
        try:
            html = requests.get(url, timeout=30, headers=headers).text
        except Exception:
            continue
        # 1) raccogli tutti i JSON.parse(...)
        found_any = False
        for m in rx_all.finditer(html):
            enc = m.group(2)
            try:
                s = enc.encode('utf-8').decode('unicode_escape')
                data = json.loads(s)
            except Exception:
                continue
            rows_part = _collect_matches_from_obj(data)
            if rows_part:
                rows.extend(rows_part)
                found_any = True
        # 2) fallback su __NEXT_DATA__
        if not found_any:
            m = rx_next.search(html)
            if m:
                try:
                    next_data = json.loads(m.group(1))
                    rows_part = _collect_matches_from_obj(next_data)
                    rows.extend(rows_part)
                except Exception:
                    pass
    df = pd.DataFrame(rows).drop_duplicates()
    if not df.empty:
        df = df.sort_values('date').reset_index(drop=True)
    return df