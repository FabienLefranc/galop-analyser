import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# --- 1. CONFIGURATION ---
URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=1556658374&single=true&output=csv"

COLONNES_CSV = [
    'Date', 'Réu', 'Course', 'Hippo', 'Dist', 'Disc', 'Spécialité', 'Terrain', 
    'Nb_Partants', 'Num_PMU', 'Cheval', 'Âge', 'Sexe', 'Jockey', 'Entraîneur', 
    'Poids', 'Corde', 'Musique', 'Cote', 'Classement', 'Gains_Car'
]

def nettoyer_nom(nom):
    if pd.isna(nom): return ""
    return re.sub(r'[\s\.]', '', str(nom)).upper()

def calculer_score_forme(musique):
    """Calcule un score de forme brut (0-20)"""
    chiffres = re.findall(r'\d+', str(musique))
    if not chiffres: return 10
    scores = []
    for c in chiffres[:5]:
        val = int(c)
        if val == 1: scores.append(20)
        elif val == 2: scores.append(16)
        elif val == 3: scores.append(13)
        elif val == 4: scores.append(10)
        else: scores.append(6)
    return np.mean(scores)

print("🚀 DÉBUT DE L'ENTRAÎNEMENT...")

# --- 2. CHARGEMENT DES DONNÉES ---
print("📥 Chargement du CSV historique...")
df = pd.read_csv(URL_CSV, header=None, names=COLONNES_CSV, on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
df['Date'] = df['Date'].astype(str).str.strip()
df = df[df['Date'] != 'Date'] # Retire l'en-tête fantôme

# Nettoyage des noms pour le merge
df["Cheval_clean"] = df["Cheval"].apply(nettoyer_nom)
df["Jockey_clean"] = df["Jockey"].apply(nettoyer_nom)
df["Entraîneur_clean"] = df["Entraîneur"].apply(nettoyer_nom)

# Conversion numérique
df['Cote'] = pd.to_numeric(df['Cote'].str.replace(',', '.', regex=False), errors='coerce').fillna(10.0)
for col in ['Dist', 'Nb_Partants', 'Poids', 'Corde', 'Classement', 'Gains_Car']:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"✅ {len(df)} courses chargées.")

# --- 3. CHARGEMENT DES STATS PRÉ-CALCULÉES ---
print("📊 Chargement des tables de stats...")
try:
    stats_jockey = pd.read_csv('stats_jockey.csv', sep=';')
    stats_entraineur = pd.read_csv('stats_entraineur.csv', sep=';')
    stats_distance = pd.read_csv('stats_distance.csv', sep=';')
    
        # Merge des stats en renommant les colonnes AVANT pour éviter les conflits de noms
    stats_j_merge = stats_jockey[['Cheval_clean', 'Jockey_clean', 'Taux_victoire']].rename(columns={'Taux_victoire': 'Taux_victoire_jockey'})
    df = df.merge(stats_j_merge, on=['Cheval_clean', 'Jockey_clean'], how='left')
    
    stats_e_merge = stats_entraineur[['Cheval_clean', 'Entraîneur_clean', 'Taux_victoire']].rename(columns={'Taux_victoire': 'Taux_victoire_entraineur'})
    df = df.merge(stats_e_merge, on=['Cheval_clean', 'Entraîneur_clean'], how='left')
    
    # Pour la distance, on prend la tranche la plus proche
    df['Dist_groupe'] = (df['Dist'] // 200) * 200
    stats_d_merge = stats_distance[['Cheval_clean', 'Dist_groupe', 'Taux_victoire']].rename(columns={'Taux_victoire': 'Taux_victoire_dist'})
    df = df.merge(stats_d_merge, on=['Cheval_clean', 'Dist_groupe'], how='left')

except Exception as e:
    print(f"️ Erreur chargement stats: {e}")
    print("Le modèle sera moins précis.")

# Remplir les NaN par 0
df['Taux_victoire_jockey'] = df['Taux_victoire_jockey'].fillna(0)
df['Taux_victoire_entraineur'] = df['Taux_victoire_entraineur'].fillna(0)
df['Taux_victoire_dist'] = df['Taux_victoire_dist'].fillna(0)

# --- 4. CRÉATION DES FEATURES (X) ET CIBLE (y) ---
print("🔧 Construction des features...")

# Calcul du score de forme
df['Score_Forme'] = df['Musique'].apply(calculer_score_forme)

# La CIBLE : 1 si le cheval a gagné (Classement == 1), 0 sinon
df['Gagne'] = (df['Classement'] == 1).astype(int)

# Sélection des colonnes pour le modèle
features = [
    'Cote', 'Poids', 'Corde', 'Nb_Partants', 'Score_Forme',
    'Taux_victoire_jockey', 'Taux_victoire_entraineur', 'Taux_victoire_dist'
]

X = df[features]
y = df['Gagne']

# --- 5. ENTRAÎNEMENT ---
print("🧠 Entraînement du modèle XGBoost...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Configuration XGBoost (optimisé pour des données déséquilibrées car peu de gagnants)
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    scale_pos_weight=len(y[y==0])/len(y[y==1]), # Gère le déséquilibre Victoire/Défaite
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# --- 6. ÉVALUATION ---
predictions = model.predict(X_test)
print(f"🎯 Précision globale : {accuracy_score(y_test, predictions):.2%}")
print("\n📊 Rapport de classification :")
print(classification_report(y_test, predictions, target_names=['Perdu (0)', 'Gagné (1)']))

# --- 7. SAUVEGARDE ---
model.save_model('modele_galop.json')
print("💾 Modèle sauvegardé dans 'modele_galop.json'")
print("✅ FINI ! Tu peux maintenant utiliser ce modèle dans Streamlit.")