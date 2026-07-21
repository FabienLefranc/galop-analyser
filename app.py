import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import xgboost as xgb
import re
from datetime import datetime
import os

st.set_page_config(page_title="Galop Analyzer", page_icon="🏇", layout="wide", initial_sidebar_state="expanded")

URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=1556658374&single=true&output=csv"
URL_COURSES_JOUR = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=365561583&single=true&output=csv"

# Liste officielle des colonnes
COLONNES_CSV = [
    'Date', 'Réu', 'Course', 'Hippo', 'Dist', 'Disc', 'Spécialité', 'Terrain', 
    'Nb_Partants', 'Num_PMU', 'Cheval', 'Âge', 'Sexe', 'Jockey', 'Entraîneur', 
    'Poids', 'Corde', 'Musique', 'Cote', 'Classement', 'Gains_Car'
]

# ==========================================
# FONCTIONS UTILITAIRES (Définies en premier)
# ==========================================
def nettoyer_nom(nom):
    if pd.isna(nom): return ""
    return re.sub(r'[\s\.]', '', str(nom)).upper()

def calculer_regularite(musique):
    chiffres = re.findall(r'\d+', str(musique))
    if len(chiffres) < 3:
        return 5
    positions = [int(c) for c in chiffres[:5]]
    ecart_type = np.std(positions)
    score = max(0, 10 - (ecart_type * 2))
    return round(score, 1)

def calculer_progression(musique):
    chiffres = re.findall(r'\d+', str(musique))
    if len(chiffres) < 3:
        return 0
    dernieres_3 = [int(c) for c in chiffres[:3]]
    if len(dernieres_3) >= 2:
        x = range(len(dernieres_3))
        tendance = np.polyfit(x, dernieres_3, 1)[0]
        score = max(-10, min(10, -tendance * 5))
        return round(score, 1)
    return 0

# ==========================================
# CHARGEMENT DES DONNÉES
# ==========================================
@st.cache_data(ttl=60)
def load_data():
    try:
        df_historique = pd.read_csv(URL_CSV, header=None, names=COLONNES_CSV, 
                                    on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
        df_historique = df_historique[df_historique['Date'] != 'Date']
        st.sidebar.success(f"✅ Historique: {len(df_historique)} lignes")
        
        try:
            df_today = pd.read_csv(URL_COURSES_JOUR, header=None, names=COLONNES_CSV, 
                                   on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
            df_today = df_today[df_today['Date'] != 'Date']
            st.sidebar.success(f"✅ Courses du jour: {len(df_today)} lignes")
            df = pd.concat([df_historique, df_today], ignore_index=True)
        except Exception as e:
            st.sidebar.error(f"⚠️ Erreur courses du jour: {e}")
            df = df_historique
        
        if 'Date' in df.columns:
            df['Date'] = df['Date'].astype(str).str.strip()
        
        for col in ['Cheval', 'Hippo', 'Jockey', 'Entraîneur']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace('\u00a0', ' ', regex=False)
                df[col] = df[col].str.replace('\u200b', '', regex=False)
                df[col] = df[col].str.replace('\ufeff', '', regex=False)
                df[col] = df[col].str.strip()
        
        # 🛠️ CORRECTION AGRESSIVE DES COTES
        if 'Cote' in df.columns:
            # Nettoyage et conversion robuste
            df['Cote'] = df['Cote'].astype(str).str.strip()
            df['Cote'] = df['Cote'].str.replace(',', '.', regex=False)
            df['Cote'] = df['Cote'].str.replace('"', '', regex=False)
            df['Cote'] = df['Cote'].str.replace("'", '', regex=False)
            
            # Conversion en float avec gestion des erreurs
            df['Cote'] = pd.to_numeric(df['Cote'], errors='coerce')
            
            # Correction des valeurs aberrantes (>100 = erreur de saisie)
            df.loc[df['Cote'] > 100, 'Cote'] = 10.0
            
            # Remplissage des NaN (cotes manquantes) par 10.0 (valeur moyenne réaliste)
            df['Cote'] = df['Cote'].fillna(10.0)
            
            # Affichage stats
            cotes_valides = df[df['Cote'] > 0]['Cote']
            st.sidebar.success(f"✅ Cotes corrigées (Moy: {cotes_valides.mean():.2f}, Min: {cotes_valides.min():.1f}, Max: {cotes_valides.max():.1f})")
        
        for col in ['Dist', 'Nb_Partants', 'Num_PMU', 'Âge', 'Poids', 'Corde', 'Classement', 'Gains_Car', 'Réu', 'Course']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        df["Cheval_clean"] = df["Cheval"].apply(nettoyer_nom)
        df["Jockey_clean"] = df["Jockey"].apply(nettoyer_nom) if "Jockey" in df.columns else ""
        df["Entraîneur_clean"] = df["Entraîneur"].apply(nettoyer_nom) if "Entraîneur" in df.columns else ""
        
        return df
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
        return None

# ==========================================
# CHARGEMENT DES STATS ET DU MODÈLE ML
# ==========================================
df = load_data()

def charger_stats(nom_fichier):
    try:
        return pd.read_csv(nom_fichier, sep=';', dtype={'Cheval_clean': str, 'Jockey_clean': str, 'Entraîneur_clean': str})
    except Exception:
        return pd.DataFrame()

stats_chevaux = charger_stats('stats_chevaux.csv')
stats_jockey = charger_stats('stats_jockey.csv')
stats_entraineur = charger_stats('stats_entraineur.csv')
stats_distance = charger_stats('stats_distance.csv')

dict_chevaux, dict_jockey, dict_entraineur, dict_distance = {}, {}, {}, {}
# ==========================================
# CHARGEMENT DES NOUVELLES TABLES DE STATS
# ==========================================
dict_hippodrome = {}
dict_terrain = {}
dict_corde = {}
dict_surface = {}
dict_poids = {}

# Hippodrome
try:
    df_hippo = pd.read_csv('stats_hippodrome.csv', sep=';', dtype={'Cheval_clean': str})
    for _, row in df_hippo.iterrows():
        dict_hippodrome[(row['Cheval_clean'], str(row['Hippo']))] = {
            'courses': int(row['Courses']), 'taux_victoire': float(row['Taux_victoire']), 'taux_podium': float(row['Taux_podium'])
        }
    st.sidebar.success(f"✅ {len(dict_hippodrome)} couples cheval/hippodrome")
except Exception: pass

# Terrain
try:
    df_terr = pd.read_csv('stats_terrain.csv', sep=';', dtype={'Cheval_clean': str})
    for _, row in df_terr.iterrows():
        dict_terrain[(row['Cheval_clean'], str(row['Terrain']))] = {
            'courses': int(row['Courses']), 'taux_victoire': float(row['Taux_victoire']), 'taux_podium': float(row['Taux_podium'])
        }
    st.sidebar.success(f"✅ {len(dict_terrain)} couples cheval/terrain")
except Exception: pass

# Corde (Hippo + Distance + Corde)
try:
    df_corde = pd.read_csv('stats_corde.csv', sep=';')
    for _, row in df_corde.iterrows():
        dict_corde[(str(row['Hippo']), int(row['Dist_groupe']), int(row['Corde']))] = {
            'courses': int(row['Courses']), 'taux_victoire': float(row['Taux_victoire']), 'taux_podium': float(row['Taux_podium'])
        }
    st.sidebar.success(f"✅ {len(dict_corde)} combos hippo/dist/corde")
except Exception: pass

# Surface (GAZON / PSF)
try:
    df_surf = pd.read_csv('stats_surface.csv', sep=';', dtype={'Cheval_clean': str})
    for _, row in df_surf.iterrows():
        dict_surface[(row['Cheval_clean'], str(row['Surface']))] = {
            'courses': int(row['Courses']), 'taux_victoire': float(row['Taux_victoire']), 'taux_podium': float(row['Taux_podium'])
        }
    st.sidebar.success(f"✅ {len(dict_surface)} couples cheval/surface")
except Exception: pass

# Poids (par tranche de 2kg)
try:
    df_poids = pd.read_csv('stats_poids.csv', sep=';', dtype={'Cheval_clean': str})
    for _, row in df_poids.iterrows():
        dict_poids[(row['Cheval_clean'], int(row['Poids_groupe']))] = {
            'courses': int(row['Courses']), 'taux_victoire': float(row['Taux_victoire']), 'taux_podium': float(row['Taux_podium'])
        }
    st.sidebar.success(f"✅ {len(dict_poids)} couples cheval/poids")
except Exception: pass

if not stats_chevaux.empty:
    for _, row in stats_chevaux.iterrows():
        dict_chevaux[row['Cheval_clean']] = {
            'courses': int(row['Courses']), 'victoires': int(row['Victoires']),
            'podiums': int(row['Podiums']), 'taux_victoire': float(row['Taux_victoire']), 'taux_podium': float(row['Taux_podium'])
        }

if not stats_jockey.empty:
    for _, row in stats_jockey.iterrows():
        dict_jockey[(row['Cheval_clean'], row['Jockey_clean'])] = {
            'courses': int(row['Courses']), 'victoires': int(row['Victoires']),
            'podiums': int(row['Podiums']), 'taux_victoire': float(row['Taux_victoire']), 'taux_podium': float(row['Taux_podium'])
        }

if not stats_entraineur.empty:
    for _, row in stats_entraineur.iterrows():
        dict_entraineur[(row['Cheval_clean'], row['Entraîneur_clean'])] = {
            'courses': int(row['Courses']), 'victoires': int(row['Victoires']),
            'podiums': int(row['Podiums']), 'taux_victoire': float(row['Taux_victoire']), 'taux_podium': float(row['Taux_podium'])
        }

if not stats_distance.empty:
    for _, row in stats_distance.iterrows():
        dict_distance[(row['Cheval_clean'], int(row['Distance']))] = {
            'courses': int(row['Courses']), 'victoires': int(row['Victoires']),
            'podiums': int(row['Podiums']), 'taux_victoire': float(row['Taux_victoire']), 'taux_podium': float(row['Taux_podium'])
        }

st.sidebar.success(f"✅ {len(dict_chevaux)} chevaux en mémoire")
st.sidebar.success(f"✅ {len(dict_jockey)} couples jockey")

# Chargement du modèle Machine Learning
model_ml = None
try:
    if os.path.exists('modele_galop.json'):
        model_ml = xgb.XGBClassifier()
        model_ml.load_model('modele_galop_v2.json')
        st.sidebar.success("✅ Modèle IA chargé avec succès")
    else:
        st.sidebar.warning("⚠️ Fichier 'modele_galop.json' introuvable")
except Exception as e:
    st.sidebar.warning(f"⚠️ Erreur chargement modèle IA: {e}")

# ==========================================
# FONCTIONS DE CALCUL DE SCORE
# ==========================================
def predire_proba_ml(row):
    """Calcule la probabilité de victoire avec le modèle IA V2 (21 features)"""
    if model_ml is None: 
        return 0.0
    
    try:
        cheval_nom = nettoyer_nom(row.get('Cheval', ''))
        jockey_actuel = nettoyer_nom(row.get('Jockey', ''))
        entraineur_actuel = nettoyer_nom(row.get('Entraîneur', ''))
        hippo_actuel = str(row.get('Hippo', '')).strip()
        dist_actuelle = float(row.get('Dist', 0))
        dist_groupe = int((dist_actuelle // 200) * 200)
        
        # 1. Calcul du Score de Forme (d'après la musique)
        musique = str(row.get('Musique', ''))
        chiffres = re.findall(r'\d+', musique)
        score_forme = 10
        if chiffres:
            scores = []
            for c in chiffres[:5]:
                val = int(c)
                if val == 1: scores.append(20)
                elif val == 2: scores.append(16)
                elif val == 3: scores.append(13)
                elif val == 4: scores.append(10)
                else: scores.append(6)
            score_forme = np.mean(scores)

        # 2. Construction des 21 features pour le modèle V2
        features = {
            'Cote': float(row.get('Cote', 10)),
            'Poids_kg': float(row.get('Poids', 0)) / 10,
            'Corde': float(row.get('Corde', 0)),
            'Nb_Partants': float(row.get('Nb_Partants', 16)),
            'Âge': float(row.get('Âge', 0)),
            'Score_Forme': score_forme,
            
            # Stats Cheval
            'Courses_cheval': dict_chevaux.get(cheval_nom, {}).get('courses', 0),
            'Taux_victoire_cheval': dict_chevaux.get(cheval_nom, {}).get('taux_victoire', 0),
            'Taux_podium_cheval': dict_chevaux.get(cheval_nom, {}).get('taux_podium', 0),
            
            # Stats Jockey
            'Courses_jockey': dict_jockey.get((cheval_nom, jockey_actuel), {}).get('courses', 0),
            'Taux_victoire_jockey': dict_jockey.get((cheval_nom, jockey_actuel), {}).get('taux_victoire', 0),
            'Taux_podium_jockey': dict_jockey.get((cheval_nom, jockey_actuel), {}).get('taux_podium', 0),
            
            # Stats Entraîneur
            'Courses_entraineur': dict_entraineur.get((cheval_nom, entraineur_actuel), {}).get('courses', 0),
            'Taux_victoire_entraineur': dict_entraineur.get((cheval_nom, entraineur_actuel), {}).get('taux_victoire', 0),
            'Taux_podium_entraineur': dict_entraineur.get((cheval_nom, entraineur_actuel), {}).get('taux_podium', 0),
            
            # Stats Distance
            'Courses_distance': dict_distance.get((cheval_nom, dist_groupe), {}).get('courses', 0),
            'Taux_victoire_distance': dict_distance.get((cheval_nom, dist_groupe), {}).get('taux_victoire', 0),
            'Taux_podium_distance': dict_distance.get((cheval_nom, dist_groupe), {}).get('taux_podium', 0),
            
            # Stats Hippodrome
            'Courses_hippo': dict_hippodrome.get((cheval_nom, hippo_actuel), {}).get('courses', 0),
            'Taux_victoire_hippo': dict_hippodrome.get((cheval_nom, hippo_actuel), {}).get('taux_victoire', 0),
            'Taux_podium_hippo': dict_hippodrome.get((cheval_nom, hippo_actuel), {}).get('taux_podium', 0)
        }

        # 3. Prédiction
        df_input = pd.DataFrame([features])
        proba = model_ml.predict_proba(df_input)[0][1]
        return round(proba * 100, 1)
        
    except Exception as e:
        return 0.0

def normaliser_probas_course(parts_df):
    """
    Normalise les Proba_IA brutes pour qu'elles somment à ~100% dans la course.
    C'est la méthode la plus fiable pour obtenir des probabilités réalistes.
    """
    total_probas = parts_df['Proba_IA'].sum()
    if total_probas == 0:
        return parts_df['Proba_IA']
    
    # On normalise et on multiplie par 100 pour avoir un pourcentage
    # On ajoute un petit bonus de base (5%) pour éviter les 0% absolus
    probas_norm = ((parts_df['Proba_IA'] / total_probas) * 100)
    
    # Lissage : on mélange 70% de la proba normalisée avec 30% de répartition égale
    nb_chevaux = len(parts_df)
    proba_equitable = 100 / nb_chevaux
    probas_lissees = (probas_norm * 0.7) + (proba_equitable * 0.3)
    
    return probas_lissees.round(1)

def calculer_score_combine(row):
    """Combine Score Classique (60%) et Proba normalisée (40%)"""
    score_classique = float(row.get('Score', 0))
    proba_norm = float(row.get('Proba_Norm', 0))
    
    # Les deux sont déjà sur une échelle 0-100, on peut les combiner directement
    score_combine = (score_classique * 0.6) + (proba_norm * 0.4)
    return round(score_combine, 1)

def calculer_score_ameliore(row, df_global, df_course):
    score = 0
    cheval_nom = nettoyer_nom(row.get('Cheval', ''))
    hippo_actuel = str(row.get('Hippo', '')).strip()
    terrain_actuel = str(row.get('Terrain', '')).strip().upper()
    dist_actuelle = float(row.get('Dist', 0))
    dist_groupe = int((dist_actuelle // 200) * 200)
    corde_actuelle = float(row.get('Corde', 0))
    poids_actuel = float(row.get('Poids', 0))
    poids_groupe = int((poids_actuel // 2) * 2)
    
    # Déterminer la surface (GAZON ou PSF)
    surface_actuelle = 'PSF' if 'FIBRE' in terrain_actuel or 'PSF' in terrain_actuel else 'GAZON'
    
    jockey_actuel = nettoyer_nom(row.get('Jockey', ''))
    entraineur_actuel = nettoyer_nom(row.get('Entraîneur', ''))
    musique = str(row.get('Musique', ''))
    
    # 1. FORME RÉCENTE (12 pts)
    chiffres = re.findall(r'\d+', musique)
    if chiffres:
        scores_forme = [100 if int(c)==1 else 80 if int(c)==2 else 65 if int(c)==3 else 50 if int(c)==4 else 30 for c in chiffres[:5]]
        score += (np.mean(scores_forme) / 100) * 12
    else:
        score += 6 

    # 2. RÉGULARITÉ (4 pts)
    score += (calculer_regularite(musique) / 10) * 4

    # 3. PROGRESSION (4 pts)
    progression = calculer_progression(musique)
    score += (max(0, min(10, progression + 10)) / 10) * 4

    # 4. JOCKEY (10 pts)
    stats = dict_jockey.get((cheval_nom, jockey_actuel), {})
    if stats.get('courses', 0) >= 3:
        s = (stats['taux_podium'] * 0.6 + stats['taux_victoire'] * 0.4) / 100 * 10
        score += s * min(1.0, stats['courses'] / 10)
    else:
        score += 3

    # 5. ENTRAÎNEUR (10 pts)
    stats = dict_entraineur.get((cheval_nom, entraineur_actuel), {})
    if stats.get('courses', 0) >= 3:
        s = (stats['taux_podium'] * 0.6 + stats['taux_victoire'] * 0.4) / 100 * 10
        score += s * min(1.0, stats['courses'] / 10)
    else:
        score += 3

    # 6. DISTANCE (8 pts)
    meilleure_stats = None
    for dist_test in [dist_groupe - 200, dist_groupe, dist_groupe + 200]:
        stats = dict_distance.get((cheval_nom, dist_test))
        if stats and (meilleure_stats is None or stats['courses'] > meilleure_stats['courses']):
            meilleure_stats = stats
    if meilleure_stats and meilleure_stats['courses'] >= 2:
        s = (meilleure_stats['taux_podium'] * 0.6 + meilleure_stats['taux_victoire'] * 0.4) / 100 * 8
        score += s * min(1.0, meilleure_stats['courses'] / 8)
    else:
        score += 4

    # 7. HIPPODROME (8 pts) - NOUVEAU
    stats = dict_hippodrome.get((cheval_nom, hippo_actuel), {})
    if stats.get('courses', 0) >= 2:
        s = (stats['taux_podium'] * 0.6 + stats['taux_victoire'] * 0.4) / 100 * 8
        score += s * min(1.0, stats['courses'] / 5)
    else:
        score += 4

    # 8. SURFACE (6 pts) - NOUVEAU
    stats = dict_surface.get((cheval_nom, surface_actuelle), {})
    if stats.get('courses', 0) >= 2:
        s = (stats['taux_podium'] * 0.6 + stats['taux_victoire'] * 0.4) / 100 * 6
        score += s * min(1.0, stats['courses'] / 5)
    else:
        score += 3

    # 9. CORDE (8 pts) - NOUVEAU (basé sur Hippo + Distance + Corde)
    stats = dict_corde.get((hippo_actuel, dist_groupe, int(corde_actuelle)), {})
    if stats.get('courses', 0) >= 5:
        s = (stats['taux_podium'] * 0.6 + stats['taux_victoire'] * 0.4) / 100 * 8
        score += s * min(1.0, stats['courses'] / 10)
    else:
        score += 4

    # 10. POIDS (8 pts) - NOUVEAU (basé sur tranche de poids)
    stats = dict_poids.get((cheval_nom, poids_groupe), {})
    if stats.get('courses', 0) >= 2:
        s = (stats['taux_podium'] * 0.6 + stats['taux_victoire'] * 0.4) / 100 * 8
        score += s * min(1.0, stats['courses'] / 5)
    else:
        score += 4

    # 11. GAINS (12 pts)
    gains = float(row.get('Gains_Car', 0))
    gains_max = df_course['Gains_Car'].max() if 'Gains_Car' in df_course.columns else 0
    score += (min(100, (gains / gains_max) * 100) / 100) * 12 if gains > 0 and gains_max > 0 else 6

    # 12. COTE (10 pts)
    cote = float(row.get('Cote', 0.0))
    if cote > 0:
        score_cote = 100 if cote <= 3 else (80 if cote <= 6 else (60 if cote <= 10 else (40 if cote <= 20 else 20)))
        score += (score_cote / 100) * 10
    else:
        score += 5
    
    return round(min(100, max(0, score)), 1)

# ==========================================
# INITIALISATION STREAMLIT
# ==========================================
if 'selected_date' not in st.session_state: st.session_state.selected_date = None
if 'selected_reu' not in st.session_state: st.session_state.selected_reu = None
if 'selected_course' not in st.session_state: st.session_state.selected_course = None

st.title("🏇 Galop Analyzer")
st.markdown("---")

if df is None or df.empty:
    st.error("❌ Impossible de charger les données.")
    st.stop()

st.sidebar.header("🎯 Navigation")
page = st.sidebar.radio("Choisir une vue :", ["📊 Tableau de bord", "📋 Résumé du jour", "🏆 Analyse d'une course", "🐎 Statistiques chevaux", "🎯 Score prédictif", "🔍 Recherche cheval"])

if st.session_state.selected_date and st.sidebar.button("🗑️ Effacer la sélection"):
    st.session_state.selected_date = st.session_state.selected_reu = st.session_state.selected_course = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(f"💡 **{len(df)} partants** analysés")

# ==========================================
# PAGES
# ==========================================
if page == "📊 Tableau de bord":
    st.header("📊 Vue d'ensemble")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("🏇 Total courses", df["Date"].nunique())
    with col2: st.metric("🐎 Total partants", len(df))
    with col3: st.metric("🏟️ Hippodromes", df["Hippo"].nunique())
    st.markdown("---")
    hippo_counts = df["Hippo"].value_counts().head(10)
    fig = px.bar(x=hippo_counts.values, y=hippo_counts.index, orientation='h', color=hippo_counts.values, color_continuous_scale='Blues')
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

elif page == "📋 Résumé du jour":
    st.header("📋 Résumé de toutes les courses du jour")
    date_du_jour = datetime.now().strftime("%d%m%Y")
    courses_du_jour = df[df["Date"] == date_du_jour]
    
    if courses_du_jour.empty:
        st.warning(f"⚠️ Aucune course trouvée pour la date {date_du_jour}")
        dates_valides = [d for d in df["Date"].unique() if isinstance(d, str) and len(d) == 8]
        date_alt = st.selectbox("Autre date :", sorted(dates_valides, reverse=True))
        courses_du_jour = df[df["Date"] == date_alt]
        st.write(f"📊 Courses trouvées pour {date_alt} : {len(courses_du_jour)}")
    else:
        st.success(f"✅ {len(courses_du_jour)} partants trouvés pour le {date_du_jour}")
        courses_list = courses_du_jour.groupby(["Réu", "Course", "Hippo", "Dist"]).size().reset_index()
        st.markdown("---")
        
        recap_data = []
        for _, course in courses_list.iterrows():
            reu = int(course["Réu"])
            num_course = int(course["Course"])
            hippo = str(course["Hippo"])
            dist = int(course["Dist"])
            
            parts = courses_du_jour[(courses_du_jour["Réu"] == reu) & (courses_du_jour["Course"] == num_course)].copy()
            
            if not parts.empty:
                parts["Score"] = parts.apply(lambda row: calculer_score_ameliore(row, df, parts), axis=1)
                parts["Proba_IA"] = parts.apply(lambda row: predire_proba_ml(row), axis=1)
                parts["Proba_Norm"] = normaliser_probas_course(parts)
                parts["Score_Combine"] = parts.apply(calculer_score_combine, axis=1)
                
                top3_score = parts.sort_values("Score", ascending=False).head(3)
                top3_ia = parts.sort_values("Proba_Norm", ascending=False).head(3)
                top3_combine = parts.sort_values("Score_Combine", ascending=False).head(3)
                
                for i, (_, cheval) in enumerate(top3_score.iterrows()):
                    recap_data.append({
                        "Hippodrome": hippo,
                        "Course": f"R{reu}C{num_course}",
                        "Distance": f"{dist}m",
                        "Rang": i + 1,
                        "Num": int(cheval["Num_PMU"]),
                        "Cheval": cheval["Cheval"],
                        "Score": float(cheval["Score"]),
                        "Proba_Norm": float(cheval["Proba_Norm"]),
                        "Score_Combine": float(cheval["Score_Combine"]),
                        "Type": "Score"
                    })
                
                for i, (_, cheval) in enumerate(top3_ia.iterrows()):
                    recap_data.append({
                        "Hippodrome": hippo,
                        "Course": f"R{reu}C{num_course}",
                        "Distance": f"{dist}m",
                        "Rang": i + 1,
                        "Num": int(cheval["Num_PMU"]),
                        "Cheval": cheval["Cheval"],
                        "Score": float(cheval["Score"]),
                        "Proba_Norm": float(cheval["Proba_Norm"]),
                        "Score_Combine": float(cheval["Score_Combine"]),
                        "Type": "IA"
                    })
                
                for i, (_, cheval) in enumerate(top3_combine.iterrows()):
                    recap_data.append({
                        "Hippodrome": hippo,
                        "Course": f"R{reu}C{num_course}",
                        "Distance": f"{dist}m",
                        "Rang": i + 1,
                        "Num": int(cheval["Num_PMU"]),
                        "Cheval": cheval["Cheval"],
                        "Score": float(cheval["Score"]),
                        "Proba_Norm": float(cheval["Proba_Norm"]),
                        "Score_Combine": float(cheval["Score_Combine"]),
                        "Type": "Combine"
                    })
        
        if recap_data:
            recap_df = pd.DataFrame(recap_data)
            
            st.subheader("🏆 Top 3 de chaque course (tous classements)")
            st.dataframe(recap_df, use_container_width=True, hide_index=True)
            
            csv = recap_df.to_csv(index=False, sep=';')
            st.download_button(
                label="📥 Télécharger le résumé en CSV",
                data=csv,
                file_name=f"resume_{date_du_jour}.csv",
                mime="text/csv"
            )
            
            st.markdown("---")
            st.subheader(" Détail par hippodrome")
            
            for hippo_name in recap_df["Hippodrome"].unique():
                courses_hippo = recap_df[recap_df["Hippodrome"] == hippo_name]
                with st.expander(f"🏟️ **{hippo_name}** ({len(courses_hippo['Course'].unique())} courses)", expanded=False):
                    for course_num in courses_hippo["Course"].unique():
                        course_data = courses_hippo[courses_hippo["Course"] == course_num]
                        distance_affichee = course_data.iloc[0]['Distance']
                        st.markdown(f"**{distance_affichee}** - {course_num}")
                        
                        # === TOP 3 SCORE CLASSIQUE ===
                        st.markdown("#### 📊 Top 3 Score Classique")
                        col1, col2, col3 = st.columns(3)
                        data_score = course_data[course_data["Type"] == "Score"].sort_values("Rang")
                        for i, (_, row) in enumerate(data_score.head(3).iterrows()):
                            with [col1, col2, col3][i]:
                                medal = ["🥇", "", "🥉"][i]
                                st.metric(f"{medal} N°{int(row['Num'])} {row['Cheval']}", f"Score: {row['Score']}")
                        
                        # === TOP 3 IA ===
                        st.markdown("#### 🤖 Top 3 IA")
                        col1, col2, col3 = st.columns(3)
                        data_ia = course_data[course_data["Type"] == "IA"].sort_values("Rang")
                        for i, (_, row) in enumerate(data_ia.head(3).iterrows()):
                            with [col1, col2, col3][i]:
                                medal = ["", "🥈", "🥉"][i]
                                st.metric(f"{medal} N°{int(row['Num'])} {row['Cheval']}", f"Proba: {row['Proba_Norm']:.1f}%")
                        
                        # === TOP 3 COMBINÉ ===
                        st.markdown("#### 🎯 Top 3 Score Combiné")
                        col1, col2, col3 = st.columns(3)
                        data_combine = course_data[course_data["Type"] == "Combine"].sort_values("Rang")
                        for i, (_, row) in enumerate(data_combine.head(3).iterrows()):
                            with [col1, col2, col3][i]:
                                medal = ["🥇", "🥈", ""][i]
                                st.metric(f"{medal} N°{int(row['Num'])} {row['Cheval']}", f"Combine: {row['Score_Combine']}")
                        
                        st.markdown("---")

elif page == "🏆 Analyse d'une course":
    st.header("🏆 Analyse détaillée d'une course")
    dates_valides = sorted([d for d in df["Date"].unique() if isinstance(d, str) and len(d) == 8], reverse=True)
    
    # Date du jour au format JJMMYYYY (ex: 21072026 pour le 21/07/2026)
    date_aujourdhui = datetime.now().strftime("%d%m%Y")
    
    # Si la date du jour existe dans la base, on la sélectionne par défaut
    if date_aujourdhui in dates_valides:
        index_defaut = dates_valides.index(date_aujourdhui)
    else:
        index_defaut = 0
    
    date_sel = st.selectbox(" Date :", dates_valides, index=index_defaut)
    courses_df = df[df["Date"] == date_sel].groupby(["Réu", "Course", "Hippo", "Dist"]).size().reset_index(name='count')
    courses_df["label"] = courses_df.apply(lambda x: f"{x['Hippo']} - R{x['Réu']}C{x['Course']} ({x['Dist']}m)", axis=1)
    course_label = st.selectbox(" Course :", courses_df["label"])
    
    if course_label:
        info = courses_df[courses_df["label"] == course_label].iloc[0]
        st.session_state.selected_date, st.session_state.selected_reu, st.session_state.selected_course = str(date_sel), int(info["Réu"]), int(info["Course"])
        parts = df[(df["Date"] == str(date_sel)) & (df["Réu"] == int(info["Réu"])) & (df["Course"] == int(info["Course"]))]
        st.markdown("---")
        st.dataframe(parts[["Num_PMU", "Cheval", "Poids", "Corde", "Cote", "Gains_Car"]], use_container_width=True)

elif page == "🐎 Statistiques chevaux":
    st.header("🐎 Statistiques des chevaux")
    if st.session_state.selected_date is None:
        st.warning("⚠️ Va d'abord dans **🏆 Analyse d'une course** !")
    else:
        parts = df[(df["Date"] == str(st.session_state.selected_date)) & (df["Réu"] == int(st.session_state.selected_reu)) & (df["Course"] == int(st.session_state.selected_course))]
        if not parts.empty:
            cheval_choisi = st.selectbox("Sélectionne un cheval :", parts["Cheval"].unique())
            if cheval_choisi:
                historique = df[df["Cheval"] == cheval_choisi].sort_values("Date", ascending=False)
                st.dataframe(historique[["Date", "Hippo", "Dist", "Classement", "Cote"]], use_container_width=True)

elif page == "🎯 Score prédictif":
    st.header(" Score prédictif")
    if st.session_state.selected_date is None:
        st.warning("️ Va d'abord dans **🏆 Analyse d'une course** !")
    else:
        parts = df[(df["Date"] == str(st.session_state.selected_date)) & 
                   (df["Réu"] == int(st.session_state.selected_reu)) & 
                   (df["Course"] == int(st.session_state.selected_course))].copy()
        
        if not parts.empty:
            st.success(f"✅ Course du **{st.session_state.selected_date}**")
            
            # Calcul des scores
            parts["Score"] = parts.apply(lambda row: calculer_score_ameliore(row, df, parts), axis=1)
            parts["Proba_IA"] = parts.apply(lambda row: predire_proba_ml(row), axis=1)
            parts["Proba_Norm"] = normaliser_probas_course(parts)
            
            # Score combiné et classement
            parts["Score_Combine"] = parts.apply(calculer_score_combine, axis=1)
            parts = parts.sort_values("Score_Combine", ascending=False)
            parts["Rang"] = range(1, len(parts)+1)
            
            st.subheader("🏆 Classement")
            st.dataframe(parts[["Rang", "Num_PMU", "Cheval", "Score_Combine", "Score", "Proba_Norm", "Cote", "Musique"]], use_container_width=True)
            
            fig = px.bar(parts, x="Cheval", y="Score", color="Score", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

elif page == "🔍 Recherche cheval":
    st.header("🔍 Recherche un cheval")
    search = st.text_input("Nom du cheval :")
    if search:
        search_clean = search.upper().strip()
        results = df[df["Cheval"].str.upper().str.contains(search_clean, na=False)]
        st.write(f"🔍 Résultats trouvés: {len(results)}")
        if not results.empty:
            st.dataframe(results[["Date", "Hippo", "Cheval", "Classement", "Cote"]].head(20), use_container_width=True)
        else:
            st.warning("❌ Aucun résultat trouvé.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;font-size:12px'>🏇 Galop Analyzer</div>", unsafe_allow_html=True)