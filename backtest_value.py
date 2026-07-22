import pandas as pd
import numpy as np
import xgboost as xgb
import re
import json
import os

URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=1556658374&single=true&output=csv"
COLONNES_CSV = ['Date', 'Réu', 'Course', 'Hippo', 'Dist', 'Disc', 'Spécialité', 'Terrain', 
                'Nb_Partants', 'Num_PMU', 'Cheval', 'Âge', 'Sexe', 'Jockey', 'Entraîneur', 
                'Poids', 'Corde', 'Musique', 'Cote', 'Classement', 'Gains_Car']

print("=" * 60)
print("🎯 BACKTESTING VALUE BETTING (Filtrage par Cote)")
print("=" * 60)

# 1. CHARGEMENT
print("\n📥 Chargement...")
df = pd.read_csv(URL_CSV, header=None, names=COLONNES_CSV, on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
df['Date'] = df['Date'].astype(str).str.strip()
df = df[df['Date'] != 'Date']
df = df[df['Date'].str.len() == 8]
df['Date_dt'] = pd.to_datetime(df['Date'], format='%d%m%Y', errors='coerce')
df = df.dropna(subset=['Date_dt']).sort_values('Date_dt').reset_index(drop=True)

for col in ['Cheval', 'Hippo', 'Jockey', 'Entraîneur', 'Terrain', 'Sexe']:
    df[col] = df[col].astype(str).str.strip()
df["Cheval_clean"] = df["Cheval"].apply(lambda x: re.sub(r'[\s\.]', '', str(x)).upper())
df["Jockey_clean"] = df["Jockey"].apply(lambda x: re.sub(r'[\s\.]', '', str(x)).upper())
df["Entraîneur_clean"] = df["Entraîneur"].apply(lambda x: re.sub(r'[\s\.]', '', str(x)).upper())
df["Terrain_clean"] = df["Terrain"].str.upper().str.strip()
df['Cote'] = pd.to_numeric(df['Cote'].str.replace(',', '.', regex=False), errors='coerce').fillna(10.0)
for col in ['Dist', 'Nb_Partants', 'Poids', 'Corde', 'Classement', 'Gains_Car', 'Âge']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# 2. MODÈLE
model_ml = xgb.XGBClassifier()
if os.path.exists('modele_galop_v3.json'):
    model_ml.load_model('modele_galop_v3.json')

# 3. FONCTION MUSIQUE
def analyser_musique(musique):
    musique = str(musique).upper().strip()
    res = {'v': 0, 'p': 0, 'm': 10.0, 'd': 10}
    if not musique or musique == 'NAN': return res
    elems = re.findall(r'(\d+|[ADTR])', musique)
    cls = [int(e) for e in elems if e.isdigit() and 1 <= int(e) <= 20]
    if cls:
        res['v'] = sum(1 for c in cls if c == 1)
        res['p'] = sum(1 for c in cls if c <= 3)
        res['m'] = round(np.mean(cls), 1)
        res['d'] = cls[0]
    return res

# 4. BACKTESTING
print("\n🔍 Analyse en cours...")
stats_c, stats_j, stats_e, stats_d, stats_h, stats_t = {}, {}, {}, {}, {}, {}

# On va tester 4 stratégies en parallèle
strategies = {
    'Tous les N°1 (8-14 partants)': {'min_cote': 0, 'max_cote': 999, 'mise': 0, 'gain': 0, 'n': 0, 'win': 0},
    'Value Cote > 3.0': {'min_cote': 3.0, 'max_cote': 999, 'mise': 0, 'gain': 0, 'n': 0, 'win': 0},
    'Value Cote > 5.0': {'min_cote': 5.0, 'max_cote': 999, 'mise': 0, 'gain': 0, 'n': 0, 'win': 0},
    'Value Cote > 8.0': {'min_cote': 8.0, 'max_cote': 999, 'mise': 0, 'gain': 0, 'n': 0, 'win': 0}
}

total_courses = 0

for idx, row in df.iterrows():
    date, reu, num = row['Date_dt'], row['Réu'], row['Course']
    nb_part = int(row['Nb_Partants'])
    
    if nb_part < 8 or nb_part > 14: continue
    
    parts = df[(df['Date_dt'] == date) & (df['Réu'] == reu) & (df['Course'] == num)]
    if len(parts) < 2: continue
    total_courses += 1
    
    preds = []
    for _, cr in parts.iterrows():
        cn, j, e = cr['Cheval_clean'], cr['Jockey_clean'], cr['Entraîneur_clean']
        dg, h, t = int((cr['Dist'] // 200) * 200), cr['Hippo'], cr['Terrain_clean']
        p = float(cr['Poids'])
        
        sc = stats_c.get(cn, {'c':0, 'v':0, 'p':0})
        sj = stats_j.get((cn, j), {'c':0, 'v':0, 'p':0})
        se = stats_e.get((cn, e), {'c':0, 'v':0, 'p':0})
        sd = stats_d.get((cn, dg), {'c':0, 'v':0, 'p':0})
        sh = stats_h.get((cn, h), {'c':0, 'v':0, 'p':0})
        st = stats_t.get((cn, t), {'c':0, 'v':0, 'p':0})
        
        score = 0
        sm = analyser_musique(cr['Musique'])
        score += min(5, sm['v'] * 1.5) + min(4, sm['p'] * 0.8)
        
        if sj['c'] >= 3: score += (sj['p']/sj['c']*100)/100 * 10 * min(1.0, sj['c']/10)
        else: score += 3
        if se['c'] >= 3: score += (se['p']/se['c']*100)/100 * 10 * min(1.0, se['c']/10)
        else: score += 3
        if sh['c'] >= 2: score += (sh['p']/sh['c']*100)/100 * 8 * min(1.0, sh['c']/5)
        else: score += 4
        if sd['c'] >= 2: score += (sd['p']/sd['c']*100)/100 * 8 * min(1.0, sd['c']/8)
        else: score += 4
        if st['c'] >= 2: score += (st['p']/st['c']*100)/100 * 6 * min(1.0, st['c']/5)
        else: score += 3
        
        hist = df[(df['Cheval_clean'] == cn) & (df['Date_dt'] < date)]
        if not hist.empty:
            ec = p - hist['Poids'].mean()
            if ec <= 0: score += 8
            elif ec <= 2: score += 6
            elif ec <= 4: score += 3
        else: score += 4
        
        cote = float(cr['Cote'])
        if cote > 0:
            sc_val = 100 if cote <= 3 else (80 if cote <= 6 else (60 if cote <= 10 else (40 if cote <= 20 else 20)))
            score += (sc_val / 100) * 12
        else: score += 6
        
        proba = 0.0
        if model_ml:
            try:
                feats = {
                    'Poids': p, 'Poids_kg': p/10, 'Corde': float(cr['Corde']), 'Nb_Partants': float(nb_part), 'Age': float(cr['Âge']),
                    'Courses_cheval': sc['c'], 'Taux_victoire_cheval': (sc['v']/sc['c']*100) if sc['c']>0 else 0, 'Taux_podium_cheval': (sc['p']/sc['c']*100) if sc['c']>0 else 0,
                    'Courses_jockey': sj['c'], 'Taux_victoire_jockey': (sj['v']/sj['c']*100) if sj['c']>0 else 0, 'Taux_podium_jockey': (sj['p']/sj['c']*100) if sj['c']>0 else 0,
                    'Courses_entraineur': se['c'], 'Taux_victoire_entraineur': (se['v']/se['c']*100) if se['c']>0 else 0, 'Taux_podium_entraineur': (se['p']/se['c']*100) if se['c']>0 else 0,
                    'Courses_distance': sd['c'], 'Taux_victoire_distance': (sd['v']/sd['c']*100) if sd['c']>0 else 0, 'Taux_podium_distance': (sd['p']/sd['c']*100) if sd['c']>0 else 0,
                    'Courses_hippo': sh['c'], 'Taux_victoire_hippo': (sh['v']/sh['c']*100) if sh['c']>0 else 0, 'Taux_podium_hippo': (sh['p']/sh['c']*100) if sh['c']>0 else 0,
                    'Courses_terrain': st['c'], 'Taux_victoire_terrain': (st['v']/st['c']*100) if st['c']>0 else 0, 'Taux_podium_terrain': (st['p']/st['c']*100) if st['c']>0 else 0,
                    'Musique_victoires': sm['v'], 'Musique_podiums': sm['p'], 'Musique_moyenne': sm['m'], 'Musique_dernier': sm['d']
                }
                proba = model_ml.predict_proba(pd.DataFrame([feats]))[0][1] * 100
            except: pass
            
        score_combine = (score * 0.6) + (proba * 0.4)
        preds.append({'cote': cote, 'cls': int(cr['Classement']), 'sc': score_combine})
        
        # MAJ Stats
        cl = int(cr['Classement'])
        for d, s in [(cn, stats_c), ((cn, j), stats_j), ((cn, e), stats_e), ((cn, dg), stats_d), ((cn, h), stats_h), ((cn, t), stats_t)]:
            if d not in s: s[d] = {'c':0, 'v':0, 'p':0}
            s[d]['c'] += 1
            s[d]['v'] += (1 if cl == 1 else 0)
            s[d]['p'] += (1 if cl <= 3 else 0)

    preds.sort(key=lambda x: x['sc'], reverse=True)
    favori = preds[0]
    
    # Application des stratégies
    for nom, strat in strategies.items():
        if strat['min_cote'] <= favori['cote'] <= strat['max_cote']:
            strat['mise'] += 1
            strat['n'] += 1
            if favori['cls'] == 1:
                strat['gain'] += favori['cote']
                strat['win'] += 1

    if total_courses % 2000 == 0:
        print(f"⏳ {total_courses} courses analysées...")

# 5. RÉSULTATS
print("\n" + "=" * 60)
print("📊 RÉSULTATS VALUE BETTING (8-14 partants)")
print("=" * 60)

for nom, s in strategies.items():
    roi = ((s['gain'] - s['mise']) / s['mise'] * 100) if s['mise'] > 0 else 0
    taux = (s['win'] / s['n'] * 100) if s['n'] > 0 else 0
    print(f"\n🎯 {nom}")
    print(f"   Paris pris : {s['n']}")
    print(f"   Taux de réussite : {taux:.1f}%")
    print(f"   ROI : {roi:.2f}%")
    print(f"   Profit sur 1000€ misés : {roi * 10:.2f}€")

print("\n" + "=" * 60)