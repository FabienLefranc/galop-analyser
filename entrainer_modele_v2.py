import pandas as pd
import numpy as np
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("🚀 ENTRAÎNEMENT DU MODÈLE V2 (SANS DATA LEAKAGE)...")

# ==========================================
# 1. CHARGEMENT DES STATS TEMPORELLES
# ==========================================
print("📥 Chargement de stats_temporelles.csv...")
df_temp = pd.read_csv('stats_temporelles.csv', sep=';', dtype={'Date': str, 'Cheval_clean': str})
print(f"✅ {len(df_temp)} lignes chargées")

# ==========================================
# 2. CHARGEMENT DU CSV HISTORIQUE (pour les autres features)
# ==========================================
print("📥 Chargement du CSV historique pour les features complémentaires...")

URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=1556658374&single=true&output=csv"

COLONNES_CSV = [
    'Date', 'Réu', 'Course', 'Hippo', 'Dist', 'Disc', 'Spécialité', 'Terrain', 
    'Nb_Partants', 'Num_PMU', 'Cheval', 'Âge', 'Sexe', 'Jockey', 'Entraîneur', 
    'Poids', 'Corde', 'Musique', 'Cote', 'Classement', 'Gains_Car'
]

df_hist = pd.read_csv(URL_CSV, header=None, names=COLONNES_CSV, on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
df_hist['Date'] = df_hist['Date'].astype(str).str.strip()
df_hist = df_hist[df_hist['Date'] != 'Date']

def nettoyer_nom(nom):
    if pd.isna(nom): return ""
    return re.sub(r'[\s\.]', '', str(nom)).upper()

df_hist["Cheval_clean"] = df_hist["Cheval"].apply(nettoyer_nom)
df_hist['Cote'] = pd.to_numeric(df_hist['Cote'].str.replace(',', '.', regex=False), errors='coerce').fillna(10.0)
for col in ['Dist', 'Nb_Partants', 'Poids', 'Corde', 'Classement', 'Gains_Car', 'Âge']:
    df_hist[col] = pd.to_numeric(df_hist[col], errors='coerce').fillna(0)

# ==========================================
# 3. FUSION DES DEUX TABLES
# ==========================================
print("🔗 Fusion des tables...")
df = df_temp.merge(
    df_hist[['Date', 'Cheval_clean', 'Cote', 'Poids', 'Corde', 'Nb_Partants', 'Musique', 'Gains_Car', 'Âge', 'Terrain']],
    on=['Date', 'Cheval_clean'],
    how='left'
)

print(f"✅ {len(df)} lignes après fusion")

# ==========================================
# 4. CALCUL DU SCORE DE FORME
# ==========================================
def calculer_score_forme(musique):
    chiffres = re.findall(r'\d+', str(musique))
    if not chiffres:
        return 10
    scores = []
    for c in chiffres[:5]:
        val = int(c)
        if val == 1: scores.append(20)
        elif val == 2: scores.append(16)
        elif val == 3: scores.append(13)
        elif val == 4: scores.append(10)
        else: scores.append(6)
    return np.mean(scores)

df['Score_Forme'] = df['Musique'].apply(calculer_score_forme)

# Conversion poids en kg
df['Poids_kg'] = df['Poids'] / 10

# ==========================================
# 5. SÉLECTION DES FEATURES
# ==========================================
print(" Sélection des features...")

features = [
    'Cote',
    'Poids_kg',
    'Corde',
    'Nb_Partants',
    'Âge',
    'Score_Forme',
    'Courses_cheval',
    'Taux_victoire_cheval',
    'Taux_podium_cheval',
    'Courses_jockey',
    'Taux_victoire_jockey',
    'Taux_podium_jockey',
    'Courses_entraineur',
    'Taux_victoire_entraineur',
    'Taux_podium_entraineur',
    'Courses_distance',
    'Taux_victoire_distance',
    'Taux_podium_distance',
    'Courses_hippo',
    'Taux_victoire_hippo',
    'Taux_podium_hippo'
]

X = df[features]
y = df['A_gagne']

# Remplir les NaN
X = X.fillna(0)

print(f"✅ {len(features)} features sélectionnées")

# ==========================================
# 6. ENTRAÎNEMENT
# ==========================================
print("🧠 Entraînement du modèle XGBoost...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    scale_pos_weight=len(y[y==0])/len(y[y==1]),
    use_label_encoder=False,
    eval_metric='logloss',
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8
)

model.fit(X_train, y_train)

# ==========================================
# 7. ÉVALUATION
# ==========================================
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print(f"\n🎯 Précision globale : {accuracy_score(y_test, predictions):.2%}")
print("\n📊 Rapport de classification :")
print(classification_report(y_test, predictions, target_names=['Perdu (0)', 'Gagné (1)']))

# Importance des features
print("\n Importance des features (top 10) :")
importances = model.feature_importances_
feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
for feat, imp in feature_importance[:10]:
    print(f"  {feat}: {imp:.3f}")

# ==========================================
# 8. SAUVEGARDE
# ==========================================
model.save_model('modele_galop_v2.json')
print("\n💾 Modèle sauvegardé dans 'modele_galop_v2.json'")
print("✅ FINI ! Le modèle est maintenant entraîné SANS data leakage.")