import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("🚀 ENTRAÎNEMENT DU MODÈLE V3 (40+ FEATURES)...")

# ==========================================
# 1. CHARGEMENT DES DONNÉES
# ==========================================
print("📥 Chargement de stats_temporelles_v3.csv...")
df = pd.read_csv('stats_temporelles_v3.csv', sep=';')
print(f"✅ {len(df)} lignes chargées")

# ==========================================
# 2. SÉLECTION DES FEATURES
# ==========================================
print("🔧 Sélection des features...")

# On exclut les colonnes qui ne sont pas des features (Date, Nom, Cible)
drop_cols = ['Date', 'Cheval_clean', 'Classement', 'A_gagne']
features = [col for col in df.columns if col not in drop_cols]

X = df[features].fillna(0) # Remplacer les NaN par 0
y = df['A_gagne']

print(f"✅ {len(features)} features sélectionnées")

# ==========================================
# 3. ENTRAÎNEMENT
# ==========================================
print("🧠 Entraînement du modèle XGBoost...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
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
# 4. ÉVALUATION
# ==========================================
predictions = model.predict(X_test)

print(f"\n🎯 Précision globale : {accuracy_score(y_test, predictions):.2%}")
print("\n📊 Rapport de classification :")
print(classification_report(y_test, predictions, target_names=['Perdu (0)', 'Gagné (1)']))

# Importance des features
print("\n📊 Importance des features (Top 15) :")
importances = model.feature_importances_
feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
for feat, imp in feature_importance[:15]:
    print(f"  {feat}: {imp:.3f}")

# ==========================================
# 5. SAUVEGARDE
# ==========================================
model.save_model('modele_galop_v3.json')
print("\n💾 Modèle sauvegardé dans 'modele_galop_v3.json'")
print("✅ FINI !")