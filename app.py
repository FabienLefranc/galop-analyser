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
        
        if 'Cote' in df.columns:
            df['Cote'] = df['Cote'].astype(str).str.replace('"', '', regex=False).str.strip()
            df['Cote'] = df['Cote'].str.replace(',', '.', regex=False)
            df['Cote'] = pd.to_numeric(df['Cote'], errors='coerce').fillna(0.0)
            st.sidebar.success(f"✅ Cotes corrigées (Moy: {df['Cote'].mean():.2f})")
        
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
        model_ml.load_model('modele_galop.json')
        st.sidebar.success("✅ Modèle IA chargé avec succès")
    else:
        st.sidebar.warning("⚠️ Fichier 'modele_galop.json' introuvable")
except Exception as e:
    st.sidebar.warning(f"⚠️ Erreur chargement modèle IA: {e}")

# ==========================================
# FONCTIONS DE CALCUL DE SCORE
# ==========================================
def predire_proba_ml(row):
    if model_ml is None: 
        st.sidebar.error("❌ model_ml est None")
        return 0.0
    
    try:
        cheval_nom = nettoyer_nom(row.get('Cheval', ''))
        jockey_actuel = nettoyer_nom(row.get('Jockey', ''))
        dist_actuelle = float(row.get('Dist', 0))
        dist_groupe = int((dist_actuelle // 200) * 200)
        
        # Debug
        st.sidebar.write(f"🔍 Cheval: {cheval_nom[:20]}")
        st.sidebar.write(f"🔍 Jockey: {jockey_actuel[:20]}")
        st.sidebar.write(f"🔍 Distance: {dist_groupe}")
        
        taux_jockey = dict_jockey.get((cheval_nom, jockey_actuel), {}).get('taux_victoire', 0)
        taux_dist = dict_distance.get((cheval_nom, dist_groupe), {}).get('taux_victoire', 0)
        
        st.sidebar.write(f"📊 Taux jockey: {taux_jockey}")
        st.sidebar.write(f"📊 Taux distance: {taux_dist}")

        musique = str(row.get('Musique', ''))
        chiffres = re.findall(r'\d+', musique)
        score_forme = 10
        if chiffres:
            scores = [20 if int(c)==1 else 16 if int(c)==2 else 13 if int(c)==3 else 10 if int(c)==4 else 6 for c in chiffres[:5]]
            score_forme = np.mean(scores)

        poids_kg = float(row.get('Poids', 0)) / 10  # Conversion grammes → kg

features = {
    'Cote': float(row.get('Cote', 10)),
    'Poids': poids_kg,  # Maintenant en kg (58 au lieu de 580)
    'Corde': float(row.get('Corde', 0)),
    'Nb_Partants': float(row.get('Nb_Partants', 16)),
    'Score_Forme': score_forme,
    'Taux_victoire_jockey': taux_jockey,
    'Taux_victoire_entraineur': 0,
    'Taux_victoire_dist': taux_dist
}
        
        st.sidebar.write(f"📊 Features: {features}")

        df_input = pd.DataFrame([features])
        proba = model_ml.predict_proba(df_input)[0][1]
        st.sidebar.success(f"✅ Proba calculée: {proba*100:.1f}%")
        return round(proba * 100, 1)
    except Exception as e:
        st.sidebar.error(f"❌ Erreur prédiction: {e}")
        import traceback
        st.sidebar.code(traceback.format_exc())
        return 0.0

def calculer_score_ameliore(row, df_global, df_course):
    score = 0
    cheval_nom = nettoyer_nom(row.get('Cheval', ''))
    dist_actuelle = float(row.get('Dist', 0))
    jockey_actuel = nettoyer_nom(row.get('Jockey', ''))
    entraineur_actuel = nettoyer_nom(row.get('Entraîneur', ''))
    musique = str(row.get('Musique', ''))
    
    # 1. FORME (15 pts)
    chiffres = re.findall(r'\d+', musique)
    if chiffres:
        scores_forme = [100 if int(c)==1 else 80 if int(c)==2 else 65 if int(c)==3 else 50 if int(c)==4 else 30 for c in chiffres[:5]]
        score += (np.mean(scores_forme) / 100) * 15
    else:
        score += 7.5 

    # 2. RÉGULARITÉ (5 pts)
    score += (calculer_regularite(musique) / 10) * 5

    # 3. PROGRESSION (5 pts)
    progression = calculer_progression(musique)
    score += (max(0, min(10, progression + 10)) / 10) * 5

    # 4. JOCKEY / ENTRAÎNEUR (25 pts)
    for key_dict, acteur_actuel in [(dict_jockey, jockey_actuel), (dict_entraineur, entraineur_actuel)]:
        if acteur_actuel != "" and cheval_nom != "":
            stats = key_dict.get((cheval_nom, acteur_actuel), {})
            if stats.get('courses', 0) >= 3:
                s = (stats['taux_podium'] * 0.6 + stats['taux_victoire'] * 0.4) / 100 * 12.5
                score += s * min(1.0, stats['courses'] / 10)
            else:
                score += 3

    # 5. DISTANCE (15 pts)
    dist_groupe = int((dist_actuelle // 200) * 200)
    meilleure_stats = None
    for dist_test in [dist_groupe - 200, dist_groupe, dist_groupe + 200]:
        stats = dict_distance.get((cheval_nom, dist_test))
        if stats and (meilleure_stats is None or stats['courses'] > meilleure_stats['courses']):
            meilleure_stats = stats
    
    if meilleure_stats and meilleure_stats['courses'] >= 2:
        s = (meilleure_stats['taux_podium'] * 0.6 + meilleure_stats['taux_victoire'] * 0.4) / 100 * 15
        score += s * min(1.0, meilleure_stats['courses'] / 8)
    else:
        score += 5

    # 6. GAINS (15 pts)
    gains = float(row.get('Gains_Car', 0))
    gains_max = df_course['Gains_Car'].max() if 'Gains_Car' in df_course.columns else 0
    score += (min(100, (gains / gains_max) * 100) / 100) * 15 if gains > 0 and gains_max > 0 else 7.5

    # 7. POIDS (10 pts)
    poids = float(row.get('Poids', 0))
    poids_moyen = df_course['Poids'].mean() if 'Poids' in df_course.columns else 0
    if poids > 0 and poids_moyen > 0:
        score += (max(0, min(100, 50 + ((poids_moyen - poids) * 5))) / 100) * 10
    else:
        score += 5

    # 8. CORDE (10 pts)
    corde = float(row.get('Corde', 0))
    nb_partants = float(row.get('Nb_Partants', 16))
    if corde > 0 and nb_partants > 0:
        score += 10 if corde <= nb_partants / 3 else (6 if corde <= nb_partants * 2 / 3 else 3)
    else:
        score += 5

    # 9. COTE (5 pts)
    cote = float(row.get('Cote', 0.0))
    if cote > 0:
        score_cote = 100 if cote <= 3 else (80 if cote <= 6 else (60 if cote <= 10 else (40 if cote <= 20 else 20)))
        score += (score_cote / 100) * 5
    else:
        score += 2.5
    
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
        dates_valides = [d for d in df["Date"].unique() if isinstance(d, str) and len(d) == 8]
        date_alt = st.selectbox("Autre date :", sorted(dates_valides, reverse=True))
        courses_du_jour = df[df["Date"] == date_alt]
    
    st.success(f"✅ {len(courses_du_jour)} partants trouvés")
    courses_list = courses_du_jour.groupby(["Réu", "Course", "Hippo", "Dist"]).size().reset_index()
    st.markdown("---")
    
    recap_data = []
    for _, course in courses_list.iterrows():
        parts = courses_du_jour[(courses_du_jour["Réu"] == int(course["Réu"])) & (courses_du_jour["Course"] == int(course["Course"]))].copy()
        if not parts.empty:
            parts["Score"] = parts.apply(lambda row: calculer_score_ameliore(row, df, parts), axis=1)
            parts = parts.sort_values("Score", ascending=False)
            for i, (_, cheval) in enumerate(parts.head(3).iterrows()):
                recap_data.append({
                    "Hippodrome": str(course["Hippo"]), "Course": f"R{int(course['Réu'])}C{int(course['Course'])}", "Distance": f"{int(course['Dist'])}m",
                    "Rang": i + 1, "Num": int(cheval["Num_PMU"]), "Cheval": cheval["Cheval"],
                    "Score": float(cheval["Score"]), "Musique": str(cheval.get("Musique", "")), "Cote": float(cheval.get("Cote", 0))
                })
    
    if recap_data:
        recap_df = pd.DataFrame(recap_data)
        st.subheader("🏆 Top 3 de chaque course")
        st.dataframe(recap_df, use_container_width=True, hide_index=True)
        st.download_button(label="📥 Télécharger le résumé en CSV", data=recap_df.to_csv(index=False, sep=';'), file_name=f"resume_{date_du_jour}.csv", mime="text/csv")

elif page == "🏆 Analyse d'une course":
    st.header("🏆 Analyse détaillée d'une course")
    dates_valides = sorted([d for d in df["Date"].unique() if isinstance(d, str) and len(d) == 8], reverse=True)
    date_sel = st.selectbox("📅 Date :", dates_valides, index=0)
    courses_df = df[df["Date"] == date_sel].groupby(["Réu", "Course", "Hippo", "Dist"]).size().reset_index(name='count')
    courses_df["label"] = courses_df.apply(lambda x: f"{x['Hippo']} - R{x['Réu']}C{x['Course']} ({x['Dist']}m)", axis=1)
    course_label = st.selectbox("🏇 Course :", courses_df["label"])
    
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
    st.header("🎯 Score prédictif")
    if st.session_state.selected_date is None:
        st.warning("⚠️ Va d'abord dans **🏆 Analyse d'une course** !")
    else:
        parts = df[(df["Date"] == str(st.session_state.selected_date)) & (df["Réu"] == int(st.session_state.selected_reu)) & (df["Course"] == int(st.session_state.selected_course))].copy()
        if not parts.empty:
            st.success(f"✅ Course du **{st.session_state.selected_date}**")
            parts["Score"] = parts.apply(lambda row: calculer_score_ameliore(row, df, parts), axis=1)
            parts["Proba_IA"] = parts.apply(lambda row: predire_proba_ml(row), axis=1)
            parts = parts.sort_values("Score", ascending=False)
            parts["Rang"] = range(1, len(parts)+1)
            
            st.subheader("🏆 Classement")
            st.dataframe(parts[["Rang", "Num_PMU", "Cheval", "Score", "Proba_IA", "Cote", "Musique"]], use_container_width=True)
            
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