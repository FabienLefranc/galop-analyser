import pandas as pd
import numpy as np
import re
from datetime import datetime

# URLs des CSV
URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=1556658374&single=true&output=csv"

COLONNES_CSV = [
    'Date', 'Réu', 'Course', 'Hippo', 'Dist', 'Disc', 'Spécialité', 'Terrain', 
    'Nb_Partants', 'Num_PMU', 'Cheval', 'Âge', 'Sexe', 'Jockey', 'Entraîneur', 
    'Poids', 'Corde', 'Musique', 'Cote', 'Classement', 'Gains_Car'
]

def nettoyer_nom(nom):
    if pd.isna(nom): return ""
    return re.sub(r'[\s\.]', '', str(nom)).upper()

print("🚀 GÉNÉRATION DES STATISTIQUES V2...")

# ==========================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================
print("📥 Chargement du CSV historique...")
df = pd.read_csv(URL_CSV, header=None, names=COLONNES_CSV, on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
df['Date'] = df['Date'].astype(str).str.strip()
df = df[df['Date'] != 'Date']

# Conversion Date en datetime pour calculs temporels
df['Date_dt'] = pd.to_datetime(df['Date'], format='%d%m%Y', errors='coerce')

# Nettoyage des noms
for col in ['Cheval', 'Hippo', 'Jockey', 'Entraîneur', 'Terrain']:
    df[col] = df[col].astype(str).str.strip()

df["Cheval_clean"] = df["Cheval"].apply(nettoyer_nom)
df["Jockey_clean"] = df["Jockey"].apply(nettoyer_nom)
df["Entraîneur_clean"] = df["Entraîneur"].apply(nettoyer_nom)
df["Terrain_clean"] = df["Terrain"].str.upper().str.strip()

# Conversion numérique
df['Cote'] = pd.to_numeric(df['Cote'].str.replace(',', '.', regex=False), errors='coerce').fillna(10.0)
for col in ['Dist', 'Nb_Partants', 'Num_PMU', 'Âge', 'Poids', 'Corde', 'Classement', 'Gains_Car']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Tranche de poids (par groupes de 2kg)
df['Poids_groupe'] = (df['Poids'] // 2) * 2

print(f"✅ {len(df)} courses chargées")

# ==========================================
# 2. TABLE 1 : Stats globales par cheval (ENRICHIE)
# ==========================================
print(" Génération stats_chevaux.csv...")
stats_chevaux = df.groupby('Cheval_clean').agg(
    Cheval=('Cheval', 'first'),
    Courses=('Date', 'count'),
    Victoires=('Classement', lambda x: (x == 1).sum()),
    Podiums=('Classement', lambda x: ((x >= 1) & (x <= 3)).sum()),
    Places=('Classement', lambda x: ((x >= 1) & (x <= 5)).sum()),
    Gains_max=('Gains_Car', 'max'),
    Gains_total=('Gains_Car', 'sum'),
    Moyenne_classement=('Classement', 'mean'),
    Mediane_classement=('Classement', 'median'),
    Ecart_type_classement=('Classement', 'std'),
    Distance_min=('Dist', 'min'),
    Distance_max=('Dist', 'max'),
    Poids_moyen=('Poids', 'mean'),
    Poids_min=('Poids', 'min'),
    Poids_max=('Poids', 'max'),
    Derniere_course=('Date_dt', 'max'),
    Premiere_course=('Date_dt', 'min')
).reset_index()

# Calcul de la distance favorite (celle avec le plus de courses)
dist_fav = df.groupby('Cheval_clean')['Dist'].agg(lambda x: x.value_counts().idxmax() if len(x) > 0 else 0).reset_index()
dist_fav.columns = ['Cheval_clean', 'Distance_favorite']
stats_chevaux = stats_chevaux.merge(dist_fav, on='Cheval_clean', how='left')

# Calcul de l'hippodrome préféré
hippo_fav = df.groupby('Cheval_clean')['Hippo'].agg(lambda x: x.value_counts().idxmax() if len(x) > 0 else '').reset_index()
hippo_fav.columns = ['Cheval_clean', 'Hippodrome_prefere']
stats_chevaux = stats_chevaux.merge(hippo_fav, on='Cheval_clean', how='left')

# Calcul des jours depuis la dernière course
stats_chevaux['Jours_depuis_derniere'] = (datetime.now() - stats_chevaux['Derniere_course']).dt.days

# Taux
stats_chevaux['Taux_victoire'] = (stats_chevaux['Victoires'] / stats_chevaux['Courses'] * 100).round(1)
stats_chevaux['Taux_podium'] = (stats_chevaux['Podiums'] / stats_chevaux['Courses'] * 100).round(1)
stats_chevaux['Taux_place'] = (stats_chevaux['Places'] / stats_chevaux['Courses'] * 100).round(1)

stats_chevaux.to_csv('stats_chevaux.csv', index=False, sep=';')
print(f"✅ {len(stats_chevaux)} chevaux")

# ==========================================
# 3. TABLE 2 : Stats par couple Cheval/Jockey
# ==========================================
print(" Génération stats_jockey.csv...")
stats_jockey = df.groupby(['Cheval_clean', 'Jockey_clean']).agg(
    Cheval=('Cheval', 'first'),
    Jockey=('Jockey', 'first'),
    Courses=('Date', 'count'),
    Victoires=('Classement', lambda x: (x == 1).sum()),
    Podiums=('Classement', lambda x: ((x >= 1) & (x <= 3)).sum()),
    Moyenne_classement=('Classement', 'mean'),
    Derniere_course=('Date_dt', 'max')
).reset_index()

stats_jockey['Taux_victoire'] = (stats_jockey['Victoires'] / stats_jockey['Courses'] * 100).round(1)
stats_jockey['Taux_podium'] = (stats_jockey['Podiums'] / stats_jockey['Courses'] * 100).round(1)
stats_jockey['Jours_depuis'] = (datetime.now() - stats_jockey['Derniere_course']).dt.days

stats_jockey.to_csv('stats_jockey.csv', index=False, sep=';')
print(f"✅ {len(stats_jockey)} couples cheval/jockey")

# ==========================================
# 4. TABLE 3 : Stats par couple Cheval/Entraîneur
# ==========================================
print(" Génération stats_entraineur.csv...")
stats_entraineur = df.groupby(['Cheval_clean', 'Entraîneur_clean']).agg(
    Cheval=('Cheval', 'first'),
    Entraîneur=('Entraîneur', 'first'),
    Courses=('Date', 'count'),
    Victoires=('Classement', lambda x: (x == 1).sum()),
    Podiums=('Classement', lambda x: ((x >= 1) & (x <= 3)).sum()),
    Moyenne_classement=('Classement', 'mean'),
    Derniere_course=('Date_dt', 'max')
).reset_index()

stats_entraineur['Taux_victoire'] = (stats_entraineur['Victoires'] / stats_entraineur['Courses'] * 100).round(1)
stats_entraineur['Taux_podium'] = (stats_entraineur['Podiums'] / stats_entraineur['Courses'] * 100).round(1)
stats_entraineur['Jours_depuis'] = (datetime.now() - stats_entraineur['Derniere_course']).dt.days

stats_entraineur.to_csv('stats_entraineur.csv', index=False, sep=';')
print(f"✅ {len(stats_entraineur)} couples cheval/entraîneur")

# ==========================================
# 5. TABLE 4 : Stats par cheval/distance (±200m)
# ==========================================
print("🔧 Génération stats_distance.csv...")
df['Dist_groupe'] = (df['Dist'] // 200) * 200

stats_distance = df.groupby(['Cheval_clean', 'Dist_groupe']).agg(
    Cheval=('Cheval', 'first'),
    Distance=('Dist_groupe', 'first'),
    Courses=('Date', 'count'),
    Victoires=('Classement', lambda x: (x == 1).sum()),
    Podiums=('Classement', lambda x: ((x >= 1) & (x <= 3)).sum()),
    Moyenne_classement=('Classement', 'mean'),
    Derniere_course=('Date_dt', 'max')
).reset_index()

stats_distance['Taux_victoire'] = (stats_distance['Victoires'] / stats_distance['Courses'] * 100).round(1)
stats_distance['Taux_podium'] = (stats_distance['Podiums'] / stats_distance['Courses'] * 100).round(1)
stats_distance['Jours_depuis'] = (datetime.now() - stats_distance['Derniere_course']).dt.days

stats_distance.to_csv('stats_distance.csv', index=False, sep=';')
print(f"✅ {len(stats_distance)} couples cheval/distance")

# ==========================================
# 6. TABLE 5 : Stats par cheval/hippodrome (NOUVEAU)
# ==========================================
print("🔧 Génération stats_hippodrome.csv...")
stats_hippodrome = df.groupby(['Cheval_clean', 'Hippo']).agg(
    Cheval=('Cheval', 'first'),
    Hippodrome=('Hippo', 'first'),
    Courses=('Date', 'count'),
    Victoires=('Classement', lambda x: (x == 1).sum()),
    Podiums=('Classement', lambda x: ((x >= 1) & (x <= 3)).sum()),
    Moyenne_classement=('Classement', 'mean'),
    Derniere_course=('Date_dt', 'max')
).reset_index()

stats_hippodrome['Taux_victoire'] = (stats_hippodrome['Victoires'] / stats_hippodrome['Courses'] * 100).round(1)
stats_hippodrome['Taux_podium'] = (stats_hippodrome['Podiums'] / stats_hippodrome['Courses'] * 100).round(1)
stats_hippodrome['Jours_depuis'] = (datetime.now() - stats_hippodrome['Derniere_course']).dt.days

stats_hippodrome.to_csv('stats_hippodrome.csv', index=False, sep=';')
print(f"✅ {len(stats_hippodrome)} couples cheval/hippodrome")

# ==========================================
# 7. TABLE 6 : Stats par cheval/terrain (NOUVEAU)
# ==========================================
print("🔧 Génération stats_terrain.csv...")
stats_terrain = df.groupby(['Cheval_clean', 'Terrain_clean']).agg(
    Cheval=('Cheval', 'first'),
    Terrain=('Terrain', 'first'),
    Courses=('Date', 'count'),
    Victoires=('Classement', lambda x: (x == 1).sum()),
    Podiums=('Classement', lambda x: ((x >= 1) & (x <= 3)).sum()),
    Moyenne_classement=('Classement', 'mean')
).reset_index()

stats_terrain['Taux_victoire'] = (stats_terrain['Victoires'] / stats_terrain['Courses'] * 100).round(1)
stats_terrain['Taux_podium'] = (stats_terrain['Podiums'] / stats_terrain['Courses'] * 100).round(1)

stats_terrain.to_csv('stats_terrain.csv', index=False, sep=';')
print(f"✅ {len(stats_terrain)} couples cheval/terrain")

# ==========================================
# 8. TABLE 7 : Stats par hippodrome/distance/corde (NOUVEAU)
# ==========================================
print("🔧 Génération stats_corde.csv...")
stats_corde = df.groupby(['Hippo', 'Dist_groupe', 'Corde']).agg(
    Hippodrome=('Hippo', 'first'),
    Distance=('Dist_groupe', 'first'),
    Valeur_corde=('Corde', 'first'),
    Courses=('Date', 'count'),
    Victoires=('Classement', lambda x: (x == 1).sum()),
    Podiums=('Classement', lambda x: ((x >= 1) & (x <= 3)).sum())
).reset_index()

stats_corde['Taux_victoire'] = (stats_corde['Victoires'] / stats_corde['Courses'] * 100).round(1)
stats_corde['Taux_podium'] = (stats_corde['Podiums'] / stats_corde['Courses'] * 100).round(1)

stats_corde.to_csv('stats_corde.csv', index=False, sep=';')
print(f"✅ {len(stats_corde)} combinaisons hippodrome/distance/corde")

# ==========================================
# 9. TABLE 8 : Stats par cheval/surface (NOUVEAU)
# ==========================================
print("🔧 Génération stats_surface.csv...")
# Surface = GAZON ou PSF (on simplifie)
df['Surface'] = df['Terrain_clean'].apply(lambda x: 'PSF' if 'FIBRE' in x or 'PSF' in x else 'GAZON')

stats_surface = df.groupby(['Cheval_clean', 'Surface']).agg(
    Cheval=('Cheval', 'first'),
    Valeur_surface=('Surface', 'first'),  # Corrigé ici
    Courses=('Date', 'count'),
    Victoires=('Classement', lambda x: (x == 1).sum()),
    Podiums=('Classement', lambda x: ((x >= 1) & (x <= 3)).sum()),
    Moyenne_classement=('Classement', 'mean')
).reset_index()

stats_surface['Taux_victoire'] = (stats_surface['Victoires'] / stats_surface['Courses'] * 100).round(1)
stats_surface['Taux_podium'] = (stats_surface['Podiums'] / stats_surface['Courses'] * 100).round(1)

stats_surface.to_csv('stats_surface.csv', index=False, sep=';')
print(f"✅ {len(stats_surface)} couples cheval/surface")

# ==========================================
# 10. TABLE 9 : Stats par cheval/tranche de poids (NOUVEAU)
# ==========================================
print("🔧 Génération stats_poids.csv...")
stats_poids = df.groupby(['Cheval_clean', 'Poids_groupe']).agg(
    Cheval=('Cheval', 'first'),
    Valeur_poids=('Poids_groupe', 'first'),  # Corrigé ici
    Courses=('Date', 'count'),
    Victoires=('Classement', lambda x: (x == 1).sum()),
    Podiums=('Classement', lambda x: ((x >= 1) & (x <= 3)).sum()),
    Moyenne_classement=('Classement', 'mean')
).reset_index()

stats_poids['Taux_victoire'] = (stats_poids['Victoires'] / stats_poids['Courses'] * 100).round(1)
stats_poids['Taux_podium'] = (stats_poids['Podiums'] / stats_poids['Courses'] * 100).round(1)

stats_poids.to_csv('stats_poids.csv', index=False, sep=';')
print(f"✅ {len(stats_poids)} couples cheval/poids")

print("\n TOUTES LES TABLES SONT GÉNÉRÉES !")
print("📁 Fichiers créés :")
print("   - stats_chevaux.csv (enrichi)")
print("   - stats_jockey.csv (enrichi)")
print("   - stats_entraineur.csv (enrichi)")
print("   - stats_distance.csv (enrichi)")
print("   - stats_hippodrome.csv (NOUVEAU)")
print("   - stats_terrain.csv (NOUVEAU)")
print("   - stats_corde.csv (NOUVEAU)")
print("   - stats_surface.csv (NOUVEAU)")
print("   - stats_poids.csv (NOUVEAU)")