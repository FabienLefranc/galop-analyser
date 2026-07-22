import pandas as pd
import numpy as np
import xgboost as xgb
import re
from datetime import datetime
import os

# ==========================================
# CONFIGURATION
# ==========================================
URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=1556658374&single=true&output=csv"

COLONNES_CSV = [
    'Date', 'Réu', 'Course', 'Hippo', 'Dist', 'Disc', 'Spécialité', 'Terrain', 
    'Nb_Partants', 'Num_PMU', 'Cheval', 'Âge', 'Sexe', 'Jockey', 'Entraîneur', 
    'Poids', 'Corde', 'Musique', 'Cote', 'Classement', 'Gains_Car'
]

MISE_PAR_COURSE = 1  # 1€ par course
TOP_N = 3  # On mise sur le top 3 des prédictions

print("=" * 60)
print("🎯 BACKTESTING HISTORIQUE")
print("=" * 60)

# ==========================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================
print("\n Chargement de l'historique...")
df = pd.read_csv(URL_CSV, header=None, names=COLONNES_CSV, on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
df['Date'] = df['Date'].astype(str).str.strip()
df = df[df['Date'] != 'Date']
df = df[df['Date'].str.len() == 8]
df['Date_dt'] = pd.to_datetime(df['Date'], format='%d%m%Y', errors='coerce')
df = df.dropna(subset=['Date_dt'])
df = df.sort_values('Date_dt').reset_index(drop=True)

# Nettoyage
for col in ['Cheval', 'Hippo', 'Jockey', 'Entraîneur', 'Terrain', 'Sexe']:
    df[col] = df[col].astype(str).str.strip()

df["Cheval_clean"] = df["Cheval"].apply(lambda x: re.sub(r'[\s\.]', '', str(x)).upper())
df["Jockey_clean"] = df["Jockey"].apply(lambda x: re.sub(r'[\s\.]', '', str(x)).upper())
df["Entraîneur_clean"] = df["Entraîneur"].apply(lambda x: re.sub(r'[\s\.]', '', str(x)).upper())
df["Terrain_clean"] = df["Terrain"].str.upper().str.strip()

df['Cote'] = pd.to_numeric(df['Cote'].str.replace(',', '.', regex=False), errors='coerce').fillna(10.0)
for col in ['Dist', 'Nb_Partants', 'Poids', 'Corde', 'Classement', 'Gains_Car', 'Âge']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"✅ {len(df)} courses valides chargées")

# ==========================================
# 2. CHARGEMENT DU MODÈLE
# ==========================================
print("\n🧠 Chargement du modèle V3...")
model_ml = None
if os.path.exists('modele_galop_v3.json'):
    model_ml = xgb.XGBClassifier()
    model_ml.load_model('modele_galop_v3.json')
    print("✅ Modèle chargé")
else:
    print("⚠️ Modèle non trouvé, on utilise uniquement le Score Classique")

# ==========================================
# 3. FONCTIONS DE CALCUL
# ==========================================
def nettoyer_nom(nom):
    if pd.isna(nom): return ""
    return re.sub(r'[\s\.]', '', str(nom)).upper()

def analyser_musique(musique):
    musique = str(musique).upper().strip()
    resultats = {
        'Nb_victoires': 0, 'Nb_podiums': 0, 'Moyenne': 10.0,
        'Dernier_classement': 10, 'Avant_dernier_classement': 10,
        'Progression': 0.0, 'Regularite': 5.0,
        'Abandons': 0, 'Disqualifications': 0, 'Courses_recentes': 0
    }
    
    if not musique or musique == 'NAN':
        return resultats
    
    elements = re.findall(r'(\d+|[ADTR])', musique)
    classements = []
    for elem in elements:
        if elem.isdigit():
            val = int(elem)
            if 1 <= val <= 20:
                classements.append(val)
        elif elem == 'A':
            resultats['Abandons'] += 1
        elif elem == 'D':
            resultats['Disqualifications'] += 1
    
    resultats['Courses_recentes'] = len(classements)
    
    if classements:
        resultats['Nb_victoires'] = sum(1 for c in classements if c == 1)
        resultats['Nb_podiums'] = sum(1 for c in classements if c <= 3)
        resultats['Moyenne'] = round(np.mean(classements), 1)
        resultats['Dernier_classement'] = classements[0]
        
        if len(classements) >= 2:
            resultats['Avant_dernier_classement'] = classements[1]
        
        if len(classements) >= 3:
            trois_derniers = classements[:3]
            resultats['Progression'] = round((trois_derniers[0] - trois_derniers[-1]) / 3, 1)
        
        if len(classements) >= 2:
            ecart_type = np.std(classements)
            resultats['Regularite'] = round(max(0, 10 - ecart_type * 2), 1)
    
    return resultats

# ==========================================
# 4. BACKTESTING
# ==========================================
print(f"\n🔍 Backtesting sur {len(df)} courses...")
print(f"💰 Mise par course : {MISE_PAR_COURSE}€")
print(f"🎯 On mise sur le Top {TOP_N} des prédictions")
print("-" * 60)

# Stats cumulatives (comme dans generer_stats_temporelles.py)
stats_cheval = {}
stats_jockey = {}
stats_entraineur = {}
stats_distance = {}
stats_hippo = {}
stats_terrain = {}

# Tracking des performances
total_mise = 0
total_gain = 0
courses_gagnees = 0
total_courses = 0
hits_top3 = 0

# Pour chaque course
for idx, course_row in df.iterrows():
    date_course = course_row['Date_dt']
    reu = course_row['Réu']
    num_course = course_row['Course']
    
    # Récupérer tous les partants de cette course
    course_parts = df[(df['Date_dt'] == date_course) & 
                      (df['Réu'] == reu) & 
                      (df['Course'] == num_course)]
    
    if len(course_parts) < 2:
        continue
    
    total_courses += 1
    
    # Calculer les prédictions pour chaque cheval
    predictions = []
    
    for _, cheval_row in course_parts.iterrows():
        cheval_nom = cheval_row['Cheval_clean']
        jockey = cheval_row['Jockey_clean']
        entraineur = cheval_row['Entraîneur_clean']
        dist_groupe = int((cheval_row['Dist'] // 200) * 200)
        hippo = cheval_row['Hippo']
        terrain = cheval_row['Terrain_clean'].upper()
        corde = int(cheval_row['Corde'])
        poids = float(cheval_row['Poids'])
        
        # Récupérer stats AVANT cette course
        s_cheval = stats_cheval.get(cheval_nom, {'courses': 0, 'victoires': 0, 'podiums': 0})
        s_jockey = stats_jockey.get((cheval_nom, jockey), {'courses': 0, 'victoires': 0, 'podiums': 0})
        s_entraineur = stats_entraineur.get((cheval_nom, entraineur), {'courses': 0, 'victoires': 0, 'podiums': 0})
        s_distance = stats_distance.get((cheval_nom, dist_groupe), {'courses': 0, 'victoires': 0, 'podiums': 0})
        s_hippo = stats_hippo.get((cheval_nom, hippo), {'courses': 0, 'victoires': 0, 'podiums': 0})
        s_terrain = stats_terrain.get((cheval_nom, terrain), {'courses': 0, 'victoires': 0, 'podiums': 0})
        
        # Calcul Score Classique simplifié
        score = 0
        
        # Musique
        stats_musique = analyser_musique(cheval_row['Musique'])
        score += min(5, stats_musique['Nb_victoires'] * 1.5)
        score += min(4, stats_musique['Nb_podiums'] * 0.8)
        
        # Jockey
        if s_jockey['courses'] >= 3:
            taux_p = (s_jockey['podiums'] / s_jockey['courses'] * 100)
            score += (taux_p / 100) * 10 * min(1.0, s_jockey['courses'] / 10)
        else:
            score += 3
        
        # Entraîneur
        if s_entraineur['courses'] >= 3:
            taux_p = (s_entraineur['podiums'] / s_entraineur['courses'] * 100)
            score += (taux_p / 100) * 10 * min(1.0, s_entraineur['courses'] / 10)
        else:
            score += 3
        
        # Hippodrome
        if s_hippo['courses'] >= 2:
            taux_p = (s_hippo['podiums'] / s_hippo['courses'] * 100)
            score += (taux_p / 100) * 8 * min(1.0, s_hippo['courses'] / 5)
        else:
            score += 4
        
        # Distance
        if s_distance['courses'] >= 2:
            taux_p = (s_distance['podiums'] / s_distance['courses'] * 100)
            score += (taux_p / 100) * 8 * min(1.0, s_distance['courses'] / 8)
        else:
            score += 4
        
        # Terrain
        if s_terrain['courses'] >= 2:
            taux_p = (s_terrain['podiums'] / s_terrain['courses'] * 100)
            score += (taux_p / 100) * 6 * min(1.0, s_terrain['courses'] / 5)
        else:
            score += 3
        
        # Poids (écart avec moyenne)
        historique_cheval = df[(df['Cheval_clean'] == cheval_nom) & (df['Date_dt'] < date_course)]
        if not historique_cheval.empty:
            poids_habituel = historique_cheval['Poids'].mean()
            ecart = poids - poids_habituel
            if ecart <= 0:
                score += 8
            elif ecart <= 2:
                score += 6
            elif ecart <= 4:
                score += 3
            else:
                score += 0
        else:
            score += 4
        
        # Gains par course
        gains = float(cheval_row['Gains_Car'])
        nb_c = s_cheval['courses'] if s_cheval['courses'] > 0 else 1
        gains_pc = gains / nb_c
        
        # Cote
        cote = float(cheval_row['Cote'])
        if cote > 0:
            score_cote = 100 if cote <= 3 else (80 if cote <= 6 else (60 if cote <= 10 else (40 if cote <= 20 else 20)))
            score += (score_cote / 100) * 12
        else:
            score += 6
        
        score = round(min(100, max(0, score)), 1)
        
        # Prédiction IA (si modèle disponible)
        proba_ia = 0.0
        if model_ml:
            try:
                features = {
                    'Poids': poids,
                    'Poids_kg': poids / 10,
                    'Corde': corde,
                    'Nb_Partants': float(cheval_row['Nb_Partants']),
                    'Age': float(cheval_row['Âge']),
                    'Courses_cheval': s_cheval['courses'],
                    'Taux_victoire_cheval': (s_cheval['victoires'] / s_cheval['courses'] * 100) if s_cheval['courses'] > 0 else 0,
                    'Taux_podium_cheval': (s_cheval['podiums'] / s_cheval['courses'] * 100) if s_cheval['courses'] > 0 else 0,
                    'Courses_jockey': s_jockey['courses'],
                    'Taux_victoire_jockey': (s_jockey['victoires'] / s_jockey['courses'] * 100) if s_jockey['courses'] > 0 else 0,
                    'Taux_podium_jockey': (s_jockey['podiums'] / s_jockey['courses'] * 100) if s_jockey['courses'] > 0 else 0,
                    'Courses_entraineur': s_entraineur['courses'],
                    'Taux_victoire_entraineur': (s_entraineur['victoires'] / s_entraineur['courses'] * 100) if s_entraineur['courses'] > 0 else 0,
                    'Taux_podium_entraineur': (s_entraineur['podiums'] / s_entraineur['courses'] * 100) if s_entraineur['courses'] > 0 else 0,
                    'Courses_distance': s_distance['courses'],
                    'Taux_victoire_distance': (s_distance['victoires'] / s_distance['courses'] * 100) if s_distance['courses'] > 0 else 0,
                    'Taux_podium_distance': (s_distance['podiums'] / s_distance['courses'] * 100) if s_distance['courses'] > 0 else 0,
                    'Courses_hippo': s_hippo['courses'],
                    'Taux_victoire_hippo': (s_hippo['victoires'] / s_hippo['courses'] * 100) if s_hippo['courses'] > 0 else 0,
                    'Taux_podium_hippo': (s_hippo['podiums'] / s_hippo['courses'] * 100) if s_hippo['courses'] > 0 else 0,
                    'Courses_terrain': s_terrain['courses'],
                    'Taux_victoire_terrain': (s_terrain['victoires'] / s_terrain['courses'] * 100) if s_terrain['courses'] > 0 else 0,
                    'Taux_podium_terrain': (s_terrain['podiums'] / s_terrain['courses'] * 100) if s_terrain['courses'] > 0 else 0,
                    'Musique_victoires': stats_musique['Nb_victoires'],
                    'Musique_podiums': stats_musique['Nb_podiums'],
                    'Musique_moyenne': stats_musique['Moyenne'],
                    'Musique_dernier': stats_musique['Dernier_classement']
                }
                df_input = pd.DataFrame([features])
                proba_ia = model_ml.predict_proba(df_input)[0][1] * 100
            except:
                proba_ia = 0.0
        
        # Score Combiné (60% classique + 40% IA)
        score_combine = (score * 0.6) + (proba_ia * 0.4)
        
        predictions.append({
            'Num_PMU': cheval_row['Num_PMU'],
            'Cheval': cheval_row['Cheval'],
            'Score': score,
            'Proba_IA': proba_ia,
            'Score_Combine': score_combine,
            'Cote': cote,
            'Classement_reel': int(cheval_row['Classement'])
        })
        
        # MAJ stats APRES cette course
        classement = int(cheval_row['Classement'])
        
        if cheval_nom not in stats_cheval:
            stats_cheval[cheval_nom] = {'courses': 0, 'victoires': 0, 'podiums': 0}
        stats_cheval[cheval_nom]['courses'] += 1
        stats_cheval[cheval_nom]['victoires'] += (1 if classement == 1 else 0)
        stats_cheval[cheval_nom]['podiums'] += (1 if classement <= 3 else 0)
        
        key_jockey = (cheval_nom, jockey)
        if key_jockey not in stats_jockey:
            stats_jockey[key_jockey] = {'courses': 0, 'victoires': 0, 'podiums': 0}
        stats_jockey[key_jockey]['courses'] += 1
        stats_jockey[key_jockey]['victoires'] += (1 if classement == 1 else 0)
        stats_jockey[key_jockey]['podiums'] += (1 if classement <= 3 else 0)
        
        key_entraineur = (cheval_nom, entraineur)
        if key_entraineur not in stats_entraineur:
            stats_entraineur[key_entraineur] = {'courses': 0, 'victoires': 0, 'podiums': 0}
        stats_entraineur[key_entraineur]['courses'] += 1
        stats_entraineur[key_entraineur]['victoires'] += (1 if classement == 1 else 0)
        stats_entraineur[key_entraineur]['podiums'] += (1 if classement <= 3 else 0)
        
        key_dist = (cheval_nom, dist_groupe)
        if key_dist not in stats_distance:
            stats_distance[key_dist] = {'courses': 0, 'victoires': 0, 'podiums': 0}
        stats_distance[key_dist]['courses'] += 1
        stats_distance[key_dist]['victoires'] += (1 if classement == 1 else 0)
        stats_distance[key_dist]['podiums'] += (1 if classement <= 3 else 0)
        
        key_hippo = (cheval_nom, hippo)
        if key_hippo not in stats_hippo:
            stats_hippo[key_hippo] = {'courses': 0, 'victoires': 0, 'podiums': 0}
        stats_hippo[key_hippo]['courses'] += 1
        stats_hippo[key_hippo]['victoires'] += (1 if classement == 1 else 0)
        stats_hippo[key_hippo]['podiums'] += (1 if classement <= 3 else 0)
        
        key_terrain = (cheval_nom, terrain)
        if key_terrain not in stats_terrain:
            stats_terrain[key_terrain] = {'courses': 0, 'victoires': 0, 'podiums': 0}
        stats_terrain[key_terrain]['courses'] += 1
        stats_terrain[key_terrain]['victoires'] += (1 if classement == 1 else 0)
        stats_terrain[key_terrain]['podiums'] += (1 if classement <= 3 else 0)
    
    # Trier par Score Combiné et prendre le Top N
    predictions.sort(key=lambda x: x['Score_Combine'], reverse=True)
    top_n = predictions[:TOP_N]
    
    # Simuler la mise
    mise_course = MISE_PAR_COURSE * len(top_n)
    total_mise += mise_course
    
    gain_course = 0
    for pred in top_n:
        if pred['Classement_reel'] == 1:
            gain_course += pred['Cote'] * MISE_PAR_COURSE
            courses_gagnees += 1
    
    total_gain += gain_course
    
    # Vérifier si le vrai gagnant est dans le Top 3
    gagnant_reel = next((p for p in predictions if p['Classement_reel'] == 1), None)
    if gagnant_reel and gagnant_reel in top_n:
        hits_top3 += 1
    
    # Affichage progression
    if total_courses % 1000 == 0:
        print(f"⏳ {total_courses} courses testées... ROI: {((total_gain - total_mise) / total_mise * 100):.1f}%")

# ==========================================
# 5. RÉSULTATS FINAUX
# ==========================================
print("\n" + "=" * 60)
print("📊 RÉSULTATS DU BACKTEST")
print("=" * 60)
print(f" Total courses testées : {total_courses}")
print(f"💰 Total misé : {total_mise:.2f}€")
print(f"💵 Total gagné : {total_gain:.2f}€")
print(f"📈 Profit/Perte : {total_gain - total_mise:.2f}€")
print(f"🎯 ROI : {((total_gain - total_mise) / total_mise * 100):.2f}%")
print(f"🏆 Courses gagnées : {courses_gagnees}")
print(f"🎲 Taux de réussite (Top {TOP_N}) : {(courses_gagnees / total_courses * 100):.1f}%")
print(f"✅ Gagnant dans Top {TOP_N} : {(hits_top3 / total_courses * 100):.1f}%")
print("=" * 60)

# Sauvegarde des résultats
resultats = {
    'total_courses': total_courses,
    'total_mise': total_mise,
    'total_gain': total_gain,
    'profit': total_gain - total_mise,
    'roi': (total_gain - total_mise) / total_mise * 100,
    'courses_gagnees': courses_gagnees,
    'taux_reussite': courses_gagnees / total_courses * 100,
    'hits_top3': hits_top3,
    'taux_hits_top3': hits_top3 / total_courses * 100
}

with open('backtest_resultats.json', 'w') as f:
    import json
    json.dump(resultats, f, indent=2)

print("\n💾 Résultats sauvegardés dans 'backtest_resultats.json'")