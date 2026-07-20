import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import xgboost as xgb
import re
from datetime import datetime

st.set_page_config(page_title="Galop Analyzer", page_icon="🏇", layout="wide", initial_sidebar_state="expanded")

URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=1556658374&single=true&output=csv"
URL_COURSES_JOUR = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=365561583&single=true&output=csv"

# Liste officielle des colonnes (puisque le CSV n'a plus d'en-tête)
COLONNES_CSV = [
    'Date', 'Réu', 'Course', 'Hippo', 'Dist', 'Disc', 'Spécialité', 'Terrain', 
    'Nb_Partants', 'Num_PMU', 'Cheval', 'Âge', 'Sexe', 'Jockey', 'Entraîneur', 
    'Poids', 'Corde', 'Musique', 'Cote', 'Classement', 'Gains_Car'
]

@st.cache_data(ttl=60)
def load_data():
    try:
        # header=None + names=COLONNES_CSV force Pandas à utiliser nos noms
        df_historique = pd.read_csv(URL_CSV, header=None, names=COLONNES_CSV, 
                                    on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
        
        # Si jamais l'en-tête réapparaît dans le CSV, on supprime la ligne "Date"
        df_historique = df_historique[df_historique['Date'] != 'Date']
        
        st.sidebar.success(f"✅ Historique: {len(df_historique)} lignes")
        
        try:
            df_today = pd.read_csv(URL_COURSES_JOUR, header=None, names=COLONNES_CSV, 
                                   on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
            df_today = df_today[df_today['Date'] != 'Date']
            st.sidebar.success(f"✅ Courses du jour: {len(df_today)} lignes")
            df = pd.concat([df_historique, df_today], ignore_index=True)
        except Exception as e:
            st.sidebar.error(f" Erreur courses du jour: {e}")
            df = df_historique
        
        # Nettoyage des dates
        if 'Date' in df.columns:
            df['Date'] = df['Date'].astype(str).str.strip()
        
        # Nettoyage agressif des noms
        for col in ['Cheval', 'Hippo', 'Jockey', 'Entraîneur']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace('\u00a0', ' ', regex=False)
                df[col] = df[col].str.replace('\u200b', '', regex=False)
                df[col] = df[col].str.replace('\u200c', '', regex=False)
                df[col] = df[col].str.replace('\u200d', '', regex=False)
                df[col] = df[col].str.replace('\ufeff', '', regex=False)
                df[col] = df[col].str.strip()
        
        # ️ CORRECTION AGRESSIVE DES COTES
        if 'Cote' in df.columns:
            df['Cote'] = df['Cote'].astype(str)
            df['Cote'] = df['Cote'].str.replace('"', '', regex=False).str.strip()
            df['Cote'] = df['Cote'].str.replace(',', '.', regex=False)
            df['Cote'] = pd.to_numeric(df['Cote'], errors='coerce').fillna(0.0)
            st.sidebar.success(f"✅ Cotes corrigées (Moyenne: {df['Cote'].mean():.2f})")
        
        # Conversion des autres nombres entiers
        for col in ['Dist', 'Nb_Partants', 'Num_PMU', 'Âge', 'Poids', 'Corde', 'Classement', 'Gains_Car', 'Réu', 'Course']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        #  OPTIMISATION : Nettoyer les noms UNE SEULE FOIS au chargement
        df["Cheval_clean"] = df["Cheval"].apply(nettoyer_nom)
        df["Jockey_clean"] = df["Jockey"].apply(nettoyer_nom) if "Jockey" in df.columns else ""
        df["Entraîneur_clean"] = df["Entraîneur"].apply(nettoyer_nom) if "Entraîneur" in df.columns else ""
        
        return df
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
        return None

def nettoyer_nom(nom):
    if pd.isna(nom): return ""
    return re.sub(r'[\s\.]', '', str(nom)).upper()

def calculer_regularite(musique):
    """
    Calcule un score de régularité de 0 à 10
    10 = très régulier (ex: 2-3-2-3-2)
    0 = très irrégulier (ex: 1-12-1-15-2)
    """
    chiffres = re.findall(r'\d+', musique)
    if len(chiffres) < 3:
        return 5  # Pas assez de données → score neutre
    
    positions = [int(c) for c in chiffres[:5]]
    ecart_type = np.std(positions)
    
    # Si écart_type = 0 (parfaitement régulier) → 10 pts
    # Si écart_type = 5 (très irrégulier) → 0 pts
    score = max(0, 10 - (ecart_type * 2))
    return round(score, 1)

def calculer_progression(musique):
    """
    Calcule un score de progression de -10 à +10
    +10 = en nette progression (ex: 8-6-4-2-1)
    -10 = en déclin (ex: 1-2-4-6-8)
    0 = stable
    """
    chiffres = re.findall(r'\d+', musique)
    if len(chiffres) < 3:
        return 0  # Pas assez de données
    
    # On prend les 3 dernières courses (les plus récentes)
    dernieres_3 = [int(c) for c in chiffres[:3]]
    
    # Régression linéaire simple
    if len(dernieres_3) >= 2:
        x = range(len(dernieres_3))
        tendance = np.polyfit(x, dernieres_3, 1)[0]
        
        # tendance < 0 = progression (positions qui diminuent)
        # tendance > 0 = déclin (positions qui augmentent)
        score = max(-10, min(10, -tendance * 5))
        return round(score, 1)
    
    return 0

def calculer_score_ameliore(row, df_global, df_course):
    score = 0
    cheval_nom = nettoyer_nom(row.get('Cheval', ''))
    dist_actuelle = float(row.get('Dist', 0))
    jockey_actuel = nettoyer_nom(row.get('Jockey', ''))
    entraineur_actuel = nettoyer_nom(row.get('Entraîneur', ''))
    
    # ==========================================
    # 1. FORME RÉCENTE (15 pts) - Réduit de 20 à 15
    # ==========================================
    musique = str(row.get('Musique', ''))
    chiffres = re.findall(r'\d+', musique)
    if chiffres:
        scores_forme = []
        for c in chiffres[:5]:
            val = int(c)
            if val == 1: scores_forme.append(100)
            elif val == 2: scores_forme.append(80)
            elif val == 3: scores_forme.append(65)
            elif val == 4: scores_forme.append(50)
            else: scores_forme.append(30)
        score += (np.mean(scores_forme) / 100) * 15
    else:
        score += 7.5 

    # ==========================================
    # 2. RÉGULARITÉ (5 pts) - NOUVEAU
    # ==========================================
    regularite = calculer_regularite(musique)
    score += (regularite / 10) * 5

    # ==========================================
    # 3. PROGRESSION (5 pts) - NOUVEAU
    # ==========================================
    progression = calculer_progression(musique)
    # On convertit -10/+10 en 0-10 puis en points
    score_progression = max(0, min(10, progression + 10)) / 10 * 5
    score += score_progression

    # ==========================================
    # 4. COUPLE JOCKEY / ENTRAÎNEUR (25 pts)
    # ==========================================
    if jockey_actuel != "" and cheval_nom != "":
        key_jockey = (cheval_nom, jockey_actuel)
        if key_jockey in dict_jockey:
            stats = dict_jockey[key_jockey]
            if stats['courses'] >= 3:
                score_jockey = (stats['taux_podium'] * 0.6 + stats['taux_victoire'] * 0.4) / 100 * 12.5
                fiabilite = min(1.0, stats['courses'] / 10)
                score += score_jockey * fiabilite
            else:
                score += 3
    
    if entraineur_actuel != "" and cheval_nom != "":
        key_entraineur = (cheval_nom, entraineur_actuel)
        if key_entraineur in dict_entraineur:
            stats = dict_entraineur[key_entraineur]
            if stats['courses'] >= 3:
                score_entraineur = (stats['taux_podium'] * 0.6 + stats['taux_victoire'] * 0.4) / 100 * 12.5
                fiabilite = min(1.0, stats['courses'] / 10)
                score += score_entraineur * fiabilite
            else:
                score += 3

    # ==========================================
    # 5. AFFINITÉ DISTANCE (15 pts)
    # ==========================================
    dist_groupe = int((dist_actuelle // 200) * 200)
    
    meilleure_stats = None
    for dist_test in [dist_groupe - 200, dist_groupe, dist_groupe + 200]:
        key_dist = (cheval_nom, dist_test)
        if key_dist in dict_distance:
            stats = dict_distance[key_dist]
            if meilleure_stats is None or stats['courses'] > meilleure_stats['courses']:
                meilleure_stats = stats
    
    if meilleure_stats and meilleure_stats['courses'] >= 2:
        score_distance = (meilleure_stats['taux_podium'] * 0.6 + meilleure_stats['taux_victoire'] * 0.4) / 100 * 15
        fiabilite = min(1.0, meilleure_stats['courses'] / 8)
        score += score_distance * fiabilite
    else:
        score += 5

    # ==========================================
    # 6. GAINS CARRIÈRE (15 pts)
    # ==========================================
    gains = float(row.get('Gains_Car', 0))
    gains_max = df_course['Gains_Car'].max() if 'Gains_Car' in df_course.columns else 0
    if gains > 0 and gains_max > 0:
        score_gains = min(100, (gains / gains_max) * 100)
        score += (score_gains / 100) * 15
    else:
        score += 7.5

    # ==========================================
    # 7. POIDS (10 pts)
    # ==========================================
    poids = float(row.get('Poids', 0))
    poids_moyen = df_course['Poids'].mean() if 'Poids' in df_course.columns else 0
    if poids > 0 and poids_moyen > 0:
        ecart = poids_moyen - poids
        score_poids = max(0, min(100, 50 + (ecart * 5)))
        score += (score_poids / 100) * 10
    else:
        score += 5

    # ==========================================
    # 8. CORDE (10 pts)
    # ==========================================
    corde = float(row.get('Corde', 0))
    nb_partants = float(row.get('Nb_Partants', 16))
    if corde > 0 and nb_partants > 0:
        if corde <= nb_partants / 3: score += 10
        elif corde <= nb_partants * 2 / 3: score += 6
        else: score += 3
    else:
        score += 5

    # ==========================================
    # 9. COTE (5 pts)
    # ==========================================
    cote = float(row.get('Cote', 0.0))
    if cote > 0:
        if cote <= 3: score_cote = 100
        elif cote <= 6: score_cote = 80
        elif cote <= 10: score_cote = 60
        elif cote <= 20: score_cote = 40
        else: score_cote = 20
        score += (score_cote / 100) * 5
    else:
        score += 2.5
    
    return round(min(100, max(0, score)), 1)

def predire_proba_ml(row):
    """Calcule la probabilité de victoire avec le modèle IA"""
    if model_ml is None: 
        return 0.0
    
    cheval_nom = nettoyer_nom(row.get('Cheval', ''))
    jockey_actuel = nettoyer_nom(row.get('Jockey', ''))
    entraineur_actuel = nettoyer_nom(row.get('Entraîneur', ''))
    dist_actuelle = float(row.get('Dist', 0))
    dist_groupe = int((dist_actuelle // 200) * 200)
    
    # Récupération des stats
    taux_jockey = 0
    taux_entraineur = 0
    taux_dist = 0
    
    if (cheval_nom, jockey_actuel) in dict_jockey:
        taux_jockey = dict_jockey[(cheval_nom, jockey_actuel)]['taux_victoire']
        
    if (cheval_nom, entraineur_actuel) in dict_entraineur:
        taux_entraineur = dict_entraineur[(cheval_nom, entraineur_actuel)]['taux_victoire']
        
    if (cheval_nom, dist_groupe) in dict_distance:
        taux_dist = dict_distance[(cheval_nom, dist_groupe)]['taux_victoire']

    # Calcul du score forme
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

    # Construction des features
    features = {
        'Cote': float(row.get('Cote', 10)),
        'Poids': float(row.get('Poids', 0)),
        'Corde': float(row.get('Corde', 0)),
        'Nb_Partants': float(row.get('Nb_Partants', 16)),
        'Score_Forme': score_forme,
        'Taux_victoire_jockey': taux_jockey,
        'Taux_victoire_entraineur': taux_entraineur,
        'Taux_victoire_dist': taux_dist
    }

    # Prédiction
    import pandas as pd
    df_input = pd.DataFrame([features])
    proba = model_ml.predict_proba(df_input)[0][1]
    return round(proba * 100, 1)

df = load_data()

# ==========================================
# CHARGEMENT DES TABLES DE STATISTIQUES
# ==========================================
st.sidebar.info("📊 Chargement des stats pré-calculées...")

# Fonction pour charger un CSV de stats avec gestion d'erreur
def charger_stats(nom_fichier):
    try:
        return pd.read_csv(nom_fichier, sep=';', dtype={'Cheval_clean': str, 'Jockey_clean': str, 'Entraîneur_clean': str})
    except Exception as e:
        st.sidebar.warning(f"⚠️ {nom_fichier} non trouvé")
        return pd.DataFrame()

stats_chevaux = charger_stats('stats_chevaux.csv')
stats_jockey = charger_stats('stats_jockey.csv')
stats_entraineur = charger_stats('stats_entraineur.csv')
stats_distance = charger_stats('stats_distance.csv')

# Création de dictionnaires pour accès ultra-rapide (O(1))
dict_chevaux = {}
if not stats_chevaux.empty:
    for _, row in stats_chevaux.iterrows():
        dict_chevaux[row['Cheval_clean']] = {
            'courses': int(row['Courses']),
            'victoires': int(row['Victoires']),
            'podiums': int(row['Podiums']),
            'taux_victoire': float(row['Taux_victoire']),
            'taux_podium': float(row['Taux_podium'])
        }

dict_jockey = {}
if not stats_jockey.empty:
    for _, row in stats_jockey.iterrows():
        key = (row['Cheval_clean'], row['Jockey_clean'])
        dict_jockey[key] = {
            'courses': int(row['Courses']),
            'victoires': int(row['Victoires']),
            'podiums': int(row['Podiums']),
            'taux_victoire': float(row['Taux_victoire']),
            'taux_podium': float(row['Taux_podium'])
        }

dict_entraineur = {}
if not stats_entraineur.empty:
    for _, row in stats_entraineur.iterrows():
        key = (row['Cheval_clean'], row['Entraîneur_clean'])
        dict_entraineur[key] = {
            'courses': int(row['Courses']),
            'victoires': int(row['Victoires']),
            'podiums': int(row['Podiums']),
            'taux_victoire': float(row['Taux_victoire']),
            'taux_podium': float(row['Taux_podium'])
        }

dict_distance = {}
if not stats_distance.empty:
    for _, row in stats_distance.iterrows():
        key = (row['Cheval_clean'], int(row['Distance']))
        dict_distance[key] = {
            'courses': int(row['Courses']),
            'victoires': int(row['Victoires']),
            'podiums': int(row['Podiums']),
            'taux_victoire': float(row['Taux_victoire']),
            'taux_podium': float(row['Taux_podium'])
        }

st.sidebar.success(f"✅ {len(dict_chevaux)} chevaux en mémoire")
st.sidebar.success(f"✅ {len(dict_jockey)} couples jockey")
st.sidebar.success(f"✅ {len(dict_entraineur)} couples entraîneur")
st.sidebar.success(f"✅ {len(dict_distance)} couples distance")
# Chargement du modèle Machine Learning
try:
    model_ml = xgb.XGBClassifier()
    model_ml.load_model('modele_galop.json')
    st.sidebar.success("✅ Modèle IA chargé")
except Exception as e:
    st.sidebar.warning("⚠️ Modèle IA non trouvé")
    model_ml = None

if 'selected_date' not in st.session_state:
    st.session_state.selected_date = None
if 'selected_reu' not in st.session_state:
    st.session_state.selected_reu = None
if 'selected_course' not in st.session_state:
    st.session_state.selected_course = None

st.title("🏇 Galop Analyzer")
st.markdown("---")

if df is None or df.empty:
    st.error("❌ Impossible de charger les données.")
    st.stop()

st.sidebar.header("🎯 Navigation")
page = st.sidebar.radio(
    "Choisir une vue :",
    ["📊 Tableau de bord", "📋 Résumé du jour", "🏆 Analyse d'une course", "🐎 Statistiques chevaux", "🎯 Score prédictif", "🔍 Recherche cheval"]
)

if st.session_state.selected_date:
    if st.sidebar.button("🗑️ Effacer la sélection"):
        st.session_state.selected_date = None
        st.session_state.selected_reu = None
        st.session_state.selected_course = None
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.info(f"💡 **{len(df)} partants** analysés")

# ==========================================
# PAGE 1: Tableau de bord
# ==========================================
if page == "📊 Tableau de bord":
    st.header("📊 Vue d'ensemble")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("🏇 Total courses", df["Date"].nunique())
    with col2: st.metric("🐎 Total partants", len(df))
    with col3: st.metric("🏟️ Hippodromes", df["Hippo"].nunique())
    st.markdown("---")
    st.subheader("🏟️ Top 10 hippodromes")
    hippo_counts = df["Hippo"].value_counts().head(10)
    fig = px.bar(x=hippo_counts.values, y=hippo_counts.index, orientation='h', color=hippo_counts.values, color_continuous_scale='Blues')
    fig.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 2: Résumé du jour
# ==========================================
elif page == "📋 Résumé du jour":
    st.header("📋 Résumé de toutes les courses du jour")
    date_du_jour = datetime.now().strftime("%d%m%Y")
    courses_du_jour = df[df["Date"] == date_du_jour]
    
    if courses_du_jour.empty:
        st.warning(f"⚠️ Aucune course trouvée pour la date {date_du_jour}")
        dates_valides = [d for d in df["Date"].unique() if isinstance(d, str) and len(d) == 8]
        date_alt = st.selectbox("Autre date :", sorted(dates_valides, reverse=True))
        courses_du_jour = df[df["Date"] == date_alt]
        st.write(f" Courses trouvées pour {date_alt} : {len(courses_du_jour)}")
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
                parts = parts.sort_values("Score", ascending=False)
                top3 = parts.head(3)
                
                for i, (_, cheval) in enumerate(top3.iterrows()):
                    recap_data.append({
                        "Hippodrome": hippo, "Course": f"R{reu}C{num_course}", "Distance": f"{dist}m",
                        "Rang": i + 1, "Num": int(cheval["Num_PMU"]), "Cheval": cheval["Cheval"],
                        "Score": float(cheval["Score"]), "Musique": str(cheval.get("Musique", "")),
                        "Cote": float(cheval.get("Cote", 0))
                    })
        
        if recap_data:
            recap_df = pd.DataFrame(recap_data)
            st.subheader("🏆 Top 3 de chaque course")
            st.dataframe(recap_df, use_container_width=True, hide_index=True)
            
            # Téléchargement CSV
            csv = recap_df.to_csv(index=False, sep=';')
            st.download_button(label="📥 Télécharger le résumé en CSV", data=csv, file_name=f"resume_{date_du_jour}.csv", mime="text/csv")
            
            st.markdown("---")
            st.subheader("📊 Détail par hippodrome")
            
            for hippo_name in recap_df["Hippodrome"].unique():
                courses_hippo = recap_df[recap_df["Hippodrome"] == hippo_name]
                with st.expander(f"🏟️ **{hippo_name}** ({len(courses_hippo['Course'].unique())} courses)", expanded=False):
                    for course_num in courses_hippo["Course"].unique():
                        course_data = courses_hippo[courses_hippo["Course"] == course_num]
                        st.markdown(f"**{course_data.iloc[0]['Distance']}** - {course_num}")
                        
                        col1, col2, col3 = st.columns(3)
                        for i, (_, row) in enumerate(course_data.iterrows()):
                            with [col1, col2, col3][i]:
                                medal = ["🥇", "🥈", "🥉"][i]
                                st.metric(f"{medal} {row['Cheval']}", f"Score: {row['Score']}")
                                musique_str = str(row['Musique'])
                                st.caption(f"Musique: {musique_str[:20]}..." if len(musique_str) > 20 else f"Musique: {musique_str}")
                                st.caption(f"Cote: {row['Cote']}")

# ==========================================
# PAGE 3: Analyse d'une course
# ==========================================
elif page == "🏆 Analyse d'une course":
    st.header("🏆 Analyse détaillée d'une course")
    dates_valides = [d for d in df["Date"].unique() if isinstance(d, str) and len(d) == 8]
    dates = sorted(dates_valides, reverse=True)
    date_sel = st.selectbox("📅 Date :", dates, index=0)
    
    courses_df = df[df["Date"] == date_sel].groupby(["Réu", "Course", "Hippo", "Dist"]).size().reset_index(name='count')
    courses_df["label"] = courses_df.apply(lambda x: f"{x['Hippo']} - R{x['Réu']}C{x['Course']} ({x['Dist']}m)", axis=1)
    course_label = st.selectbox("🏇 Course :", courses_df["label"])
    
    if course_label:
        info = courses_df[courses_df["label"] == course_label].iloc[0]
        st.session_state.selected_date = str(date_sel)
        st.session_state.selected_reu = int(info["Réu"])
        st.session_state.selected_course = int(info["Course"])
        
        parts = df[(df["Date"] == str(date_sel)) & (df["Réu"] == int(info["Réu"])) & (df["Course"] == int(info["Course"]))]
        st.markdown("---")
        st.dataframe(parts[["Num_PMU", "Cheval", "Poids", "Corde", "Cote", "Gains_Car"]], use_container_width=True)

# ==========================================
# PAGE 4: Statistiques chevaux
# ==========================================
elif page == "🐎 Statistiques chevaux":
    st.header("🐎 Statistiques des chevaux")
    if st.session_state.selected_date is None:
        st.warning("⚠️ Va d'abord dans **🏆 Analyse d'une course** !")
    else:
        parts = df[(df["Date"] == str(st.session_state.selected_date)) & (df["Réu"] == int(st.session_state.selected_reu)) & (df["Course"] == int(st.session_state.selected_course))]
        if not parts.empty:
            st.success(f"✅ Course du **{st.session_state.selected_date}**")
            cheval_choisi = st.selectbox("Sélectionne un cheval :", parts["Cheval"].unique())
            if cheval_choisi:
                historique = df[df["Cheval"] == cheval_choisi].sort_values("Date", ascending=False)
                st.dataframe(historique[["Date", "Hippo", "Dist", "Classement", "Cote"]], use_container_width=True)

# ==========================================
# PAGE 5: Score prédictif
# ==========================================
elif page == "🎯 Score prédictif":
    st.header("🎯 Score prédictif")
    
    if st.session_state.selected_date is None:
        st.warning("⚠️ Va d'abord dans **🏆 Analyse d'une course** !")
    else:
        parts = df[(df["Date"] == str(st.session_state.selected_date)) & 
                   (df["Réu"] == int(st.session_state.selected_reu)) & 
                   (df["Course"] == int(st.session_state.selected_course))].copy()
        
        if not parts.empty:
            st.markdown("### 🔍 Vérification des données (1er cheval)")
            premier = parts.iloc[0]
            nom_propre = nettoyer_nom(premier['Cheval'])
            hist_cheval = df[df['Cheval'].apply(nettoyer_nom) == nom_propre]
            
            col1, col2, col3 = st.columns(3)
            with col1: 
                st.metric("Courses dans l'historique", len(hist_cheval))
            with col2: 
                st.metric("Cote lue", f"{float(premier['Cote']):.2f}")
            with col3: 
                jockey = nettoyer_nom(premier['Jockey'])
                victoires_jockey = len(hist_cheval[(hist_cheval['Jockey'].apply(nettoyer_nom) == jockey) & (hist_cheval['Classement'] == 1)])
                st.metric("Victoires avec ce jockey", victoires_jockey)
            
            st.markdown("---")
            st.success(f"✅ Course du **{st.session_state.selected_date}**")
            
            parts["Score"] = parts.apply(lambda row: calculer_score_ameliore(row, df, parts), axis=1)
            parts["Proba_IA"] = parts.apply(lambda row: predire_proba_ml(row), axis=1)
            parts = parts.sort_values("Score", ascending=False)
            parts["Rang"] = range(1, len(parts)+1)
            
            st.subheader("🏆 Classement")
            st.dataframe(parts[["Rang", "Num_PMU", "Cheval", "Score", "Proba_IA", "Cote", "Musique"]], use_container_width=True)
            
            fig = px.bar(parts, x="Cheval", y="Score", color="Score", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 6: Recherche
# ==========================================
elif page == "🔍 Recherche cheval":
    st.header("🔍 Recherche un cheval")
    search = st.text_input("Nom du cheval :")
    
    if search:
        st.write(f"📊 Total lignes dans la base: {len(df)}")
        search_clean = search.upper().strip()
        results = df[df["Cheval"].str.upper().str.contains(search_clean, na=False)]
        
        st.write(f"🔍 Résultats trouvés: {len(results)}")
        
        if not results.empty:
            st.success(f"✅ Trouvé {len(results)} course(s)")
            st.dataframe(results[["Date", "Hippo", "Cheval", "Classement", "Cote"]].head(20), use_container_width=True)
        else:
            st.warning("❌ Aucun résultat trouvé.")

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;font-size:12px'>🏇 Galop Analyzer</div>", unsafe_allow_html=True)