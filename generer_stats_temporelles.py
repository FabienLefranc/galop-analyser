import pandas as pd
import numpy as np
import re
from datetime import datetime

URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=1556658374&single=true&output=csv"

COLONNES_CSV = [
    'Date', 'Réu', 'Course', 'Hippo', 'Dist', 'Disc', 'Spécialité', 'Terrain', 
    'Nb_Partants', 'Num_PMU', 'Cheval', 'Âge', 'Sexe', 'Jockey', 'Entraîneur', 
    'Poids', 'Corde', 'Musique', 'Cote', 'Classement', 'Gains_Car'
]

def nettoyer_nom(nom):
    if pd.isna(nom): return ""
    return re.sub(r'[\s\.]', '', str(nom)).upper()

print(" GÉNÉRATION DES STATS TEMPORELLES V3 (40+ FEATURES)...")

# ==========================================
# 1. CHARGEMENT ET TRI CHRONOLOGIQUE
# ==========================================
print("📥 Chargement du CSV historique...")
df = pd.read_csv(URL_CSV, header=None, names=COLONNES_CSV, on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
df['Date'] = df['Date'].astype(str).str.strip()
df = df[df['Date'] != 'Date']
df['Date'] = df['Date'].apply(lambda x: x if len(x) == 8 else None)
df = df.dropna(subset=['Date'])
df['Date_dt'] = pd.to_datetime(df['Date'], format='%d%m%Y', errors='coerce')
df = df.dropna(subset=['Date_dt'])
df = df.sort_values('Date_dt').reset_index(drop=True)

# Nettoyage
for col in ['Cheval', 'Hippo', 'Jockey', 'Entraîneur', 'Terrain', 'Sexe']:
    df[col] = df[col].astype(str).str.strip()

df["Cheval_clean"] = df["Cheval"].apply(nettoyer_nom)
df["Jockey_clean"] = df["Jockey"].apply(nettoyer_nom)
df["Entraîneur_clean"] = df["Entraîneur"].apply(nettoyer_nom)
df["Terrain_clean"] = df["Terrain"].str.upper()

df['Cote'] = pd.to_numeric(df['Cote'].str.replace(',', '.', regex=False), errors='coerce').fillna(10.0)
for col in ['Dist', 'Nb_Partants', 'Poids', 'Corde', 'Classement', 'Gains_Car', 'Âge']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"✅ {len(df)} courses valides chargées")

# ==========================================
# 2. CALCUL DES STATS CUMULATIVES ENRICHIES
# ==========================================
print("🔧 Calcul des stats cumulatives (peut prendre 10-15 minutes)...")

# Dictionnaires complexes
stats_cheval = {}
stats_jockey = {}
stats_entraineur = {}
stats_distance = {}
stats_hippo = {}
stats_terrain = {}
stats_corde = {}

resultats = []

for idx, row in df.iterrows():
    cheval = row['Cheval_clean']
    jockey = row['Jockey_clean']
    entraineur = row['Entraîneur_clean']
    dist_groupe = int((row['Dist'] // 200) * 200)
    hippo = row['Hippo']
    terrain = row['Terrain_clean']
    corde = int(row['Corde'])
    classement = int(row['Classement'])
    sexe = row['Sexe']
    age = int(row['Âge'])
    poids = float(row['Poids'])
    gains = float(row['Gains_Car'])
    nb_partants = int(row['Nb_Partants'])
    date_course = row['Date_dt']
    
    # Récupérer stats AVANT cette course
    s = stats_cheval.get(cheval, {
        'courses': 0, 'victoires': 0, 'podiums': 0, 'places': 0,
        'classements': [], 'gains_total': 0, 'poids_list': [],
        'distances': [], 'hippodromes': [], 'terrains': [],
        'derniere_date': None, 'distance_favorite': 0,
        'hippo_favorite': '', 'terrain_favorite': ''
    })
    
    s_jockey = stats_jockey.get((cheval, jockey), {'courses': 0, 'victoires': 0, 'podiums': 0})
    s_entraineur = stats_entraineur.get((cheval, entraineur), {'courses': 0, 'victoires': 0, 'podiums': 0})
    s_distance = stats_distance.get((cheval, dist_groupe), {'courses': 0, 'victoires': 0, 'podiums': 0})
    s_hippo = stats_hippo.get((cheval, hippo), {'courses': 0, 'victoires': 0, 'podiums': 0})
    s_terrain = stats_terrain.get((cheval, terrain), {'courses': 0, 'victoires': 0, 'podiums': 0})
    s_corde = stats_corde.get((cheval, hippo, dist_groupe, corde), {'courses': 0, 'victoires': 0, 'podiums': 0})
    
    # Calcul jours depuis dernière course
    jours_depuis = 0
    if s['derniere_date']:
        jours_depuis = (date_course - s['derniere_date']).days
    
    # Calcul progression et régularité
    progression = 0
    regularite = 0
    if len(s['classements']) >= 3:
        recent = s['classements'][-3:]
        progression = (recent[0] - recent[-1]) / len(recent) if len(recent) > 0 else 0
        regularite = 10 - min(10, np.std(recent) * 2)
    
    # Calcul distance/hippo/terrain favorite
    distance_favorite = max(set(s['distances']), key=s['distances'].count) if s['distances'] else dist_groupe
    hippo_favorite = max(set(s['hippodromes']), key=s['hippodromes'].count) if s['hippodromes'] else hippo
    terrain_favorite = max(set(s['terrains']), key=s['terrains'].count) if s['terrains'] else terrain
    
    # Calcul gains/course
    gains_per_course = s['gains_total'] / s['courses'] if s['courses'] > 0 else 0
    
    # Calcul poids moyen et écart
    poids_moyen = np.mean(s['poids_list']) if s['poids_list'] else poids
    poids_ecart = poids - poids_moyen
    
    # Taux
    taux_v_cheval = (s['victoires'] / s['courses'] * 100) if s['courses'] > 0 else 0
    taux_p_cheval = (s['podiums'] / s['courses'] * 100) if s['courses'] > 0 else 0
    moyenne_classement = np.mean(s['classements']) if s['classements'] else 0
    
    # Stocker résultats
    resultats.append({
        'Date': row['Date'],
        'Cheval_clean': cheval,
        
        # Features de base
        'Poids': poids,
        'Poids_kg': poids / 10,
        'Corde': corde,
        'Nb_Partants': nb_partants,
        'Age': age,
        'Sexe_Male': 1 if sexe == 'M' else 0,
        'Sexe_Femelle': 1 if sexe == 'F' else 0,
        
        # Terrain
        'Terrain_Bon': 1 if 'BON' in terrain else 0,
        'Terrain_Souple': 1 if 'SOUPLE' in terrain else 0,
        'Terrain_Collant': 1 if 'COLLANT' in terrain else 0,
        'Terrain_Lourd': 1 if 'LOURD' in terrain else 0,
        'Terrain_PSF': 1 if 'PSF' in terrain or 'FIBRE' in terrain else 0,
        
        # Stats Cheval
        'Courses_cheval': s['courses'],
        'Taux_victoire_cheval': round(taux_v_cheval, 1),
        'Taux_podium_cheval': round(taux_p_cheval, 1),
        'Moyenne_classement': round(moyenne_classement, 1),
        'Jours_depuis_derniere': jours_depuis,
        'Progression': round(progression, 2),
        'Regularite': round(regularite, 1),
        'Gains_per_course': round(gains_per_course, 0),
        'Poids_ecart': round(poids_ecart, 1),
        'Distance_favorite_match': 1 if dist_groupe == distance_favorite else 0,
        'Hippo_favorite_match': 1 if hippo == hippo_favorite else 0,
        'Terrain_favorite_match': 1 if terrain == terrain_favorite else 0,
        
        # Stats Jockey
        'Courses_jockey': s_jockey['courses'],
        'Taux_victoire_jockey': round((s_jockey['victoires'] / s_jockey['courses'] * 100) if s_jockey['courses'] > 0 else 0, 1),
        'Taux_podium_jockey': round((s_jockey['podiums'] / s_jockey['courses'] * 100) if s_jockey['courses'] > 0 else 0, 1),
        
        # Stats Entraîneur
        'Courses_entraineur': s_entraineur['courses'],
        'Taux_victoire_entraineur': round((s_entraineur['victoires'] / s_entraineur['courses'] * 100) if s_entraineur['courses'] > 0 else 0, 1),
        'Taux_podium_entraineur': round((s_entraineur['podiums'] / s_entraineur['courses'] * 100) if s_entraineur['courses'] > 0 else 0, 1),
        
        # Stats Distance
        'Courses_distance': s_distance['courses'],
        'Taux_victoire_distance': round((s_distance['victoires'] / s_distance['courses'] * 100) if s_distance['courses'] > 0 else 0, 1),
        'Taux_podium_distance': round((s_distance['podiums'] / s_distance['courses'] * 100) if s_distance['courses'] > 0 else 0, 1),
        
        # Stats Hippodrome
        'Courses_hippo': s_hippo['courses'],
        'Taux_victoire_hippo': round((s_hippo['victoires'] / s_hippo['courses'] * 100) if s_hippo['courses'] > 0 else 0, 1),
        'Taux_podium_hippo': round((s_hippo['podiums'] / s_hippo['courses'] * 100) if s_hippo['courses'] > 0 else 0, 1),
        
        # Stats Terrain
        'Courses_terrain': s_terrain['courses'],
        'Taux_victoire_terrain': round((s_terrain['victoires'] / s_terrain['courses'] * 100) if s_terrain['courses'] > 0 else 0, 1),
        'Taux_podium_terrain': round((s_terrain['podiums'] / s_terrain['courses'] * 100) if s_terrain['courses'] > 0 else 0, 1),
        
        # Stats Corde (par hippo + distance)
        'Courses_corde': s_corde['courses'],
        'Taux_victoire_corde': round((s_corde['victoires'] / s_corde['courses'] * 100) if s_corde['courses'] > 0 else 0, 1),
        'Taux_podium_corde': round((s_corde['podiums'] / s_corde['courses'] * 100) if s_corde['courses'] > 0 else 0, 1),
        
        # Musique convertie
        'Musique_victoires': sum(1 for c in re.findall(r'\d+', str(row['Musique']))[:5] if int(c) == 1),
        'Musique_podiums': sum(1 for c in re.findall(r'\d+', str(row['Musique']))[:5] if int(c) <= 3),
        'Musique_moyenne': np.mean([int(c) for c in re.findall(r'\d+', str(row['Musique']))[:5]]) if re.findall(r'\d+', str(row['Musique'])) else 10,
        'Musique_dernier': int(re.findall(r'\d+', str(row['Musique']))[0]) if re.findall(r'\d+', str(row['Musique'])) else 10,
        
        # Cible
        'Classement': classement,
        'A_gagne': 1 if classement == 1 else 0
    })
    
    # MAJ stats APRES cette course
    stats_cheval[cheval] = {
        'courses': s['courses'] + 1,
        'victoires': s['victoires'] + (1 if classement == 1 else 0),
        'podiums': s['podiums'] + (1 if classement <= 3 else 0),
        'places': s['places'] + (1 if classement <= 5 else 0),
        'classements': s['classements'] + [classement],
        'gains_total': s['gains_total'] + gains,
        'poids_list': s['poids_list'] + [poids],
        'distances': s['distances'] + [dist_groupe],
        'hippodromes': s['hippodromes'] + [hippo],
        'terrains': s['terrains'] + [terrain],
        'derniere_date': date_course,
        'distance_favorite': distance_favorite,
        'hippo_favorite': hippo_favorite,
        'terrain_favorite': terrain_favorite
    }
    
    stats_jockey[(cheval, jockey)] = {
        'courses': s_jockey['courses'] + 1,
        'victoires': s_jockey['victoires'] + (1 if classement == 1 else 0),
        'podiums': s_jockey['podiums'] + (1 if classement <= 3 else 0)
    }
    
    stats_entraineur[(cheval, entraineur)] = {
        'courses': s_entraineur['courses'] + 1,
        'victoires': s_entraineur['victoires'] + (1 if classement == 1 else 0),
        'podiums': s_entraineur['podiums'] + (1 if classement <= 3 else 0)
    }
    
    stats_distance[(cheval, dist_groupe)] = {
        'courses': s_distance['courses'] + 1,
        'victoires': s_distance['victoires'] + (1 if classement == 1 else 0),
        'podiums': s_distance['podiums'] + (1 if classement <= 3 else 0)
    }
    
    stats_hippo[(cheval, hippo)] = {
        'courses': s_hippo['courses'] + 1,
        'victoires': s_hippo['victoires'] + (1 if classement == 1 else 0),
        'podiums': s_hippo['podiums'] + (1 if classement <= 3 else 0)
    }
    
    stats_terrain[(cheval, terrain)] = {
        'courses': s_terrain['courses'] + 1,
        'victoires': s_terrain['victoires'] + (1 if classement == 1 else 0),
        'podiums': s_terrain['podiums'] + (1 if classement <= 3 else 0)
    }
    
    stats_corde[(cheval, hippo, dist_groupe, corde)] = {
        'courses': s_corde['courses'] + 1,
        'victoires': s_corde['victoires'] + (1 if classement == 1 else 0),
        'podiums': s_corde['podiums'] + (1 if classement <= 3 else 0)
    }
    
    if idx % 10000 == 0:
        print(f"  ⏳ {idx}/{len(df)} courses traitées...")

# ==========================================
# 3. SAUVEGARDE
# ==========================================
df_resultats = pd.DataFrame(resultats)
df_resultats.to_csv('stats_temporelles_v3.csv', index=False, sep=';')

print(f"\n✅ Fichier 'stats_temporelles_v3.csv' créé avec {len(df_resultats)} lignes")
print(f" Nombre de features : {len(df_resultats.columns) - 3} (hors Date, Cheval, Classement)")