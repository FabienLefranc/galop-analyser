import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import re
from datetime import datetime

st.set_page_config(page_title="Galop Analyzer", page_icon="🏇", layout="wide", initial_sidebar_state="expanded")

URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=1556658374&single=true&output=csv"
URL_COURSES_JOUR = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=365561583&single=true&output=csv"

@st.cache_data(ttl=60)
def load_data():
    try:
        # On force la lecture de la Cote en TEXTE pour éviter les bugs de virgules
        df_historique = pd.read_csv(URL_CSV, on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
        st.sidebar.success(f"✅ Historique: {len(df_historique)} lignes")
        try:
            df_today = pd.read_csv(URL_COURSES_JOUR, on_bad_lines='skip', dtype={'Date': str, 'Cote': str})
            st.sidebar.success(f"✅ Courses du jour: {len(df_today)} lignes")
            df = pd.concat([df_historique, df_today], ignore_index=True)
        except Exception as e:
            st.sidebar.error(f"❌ Erreur courses du jour: {e}")
            df = df_historique
        
        # Nettoyage des dates et textes
        if 'Date' in df.columns:
            df['Date'] = df['Date'].astype(str).str.strip()
        
        for col in ['Cheval', 'Hippo']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        # 🛠️ CORRECTION AGRESSIVE DES COTES
        if 'Cote' in df.columns:
            # 1. On s'assure que c'est du texte
            df['Cote'] = df['Cote'].astype(str)
            # 2. On enlève les guillemets et les espaces
            df['Cote'] = df['Cote'].str.replace('"', '', regex=False).str.strip()
            # 3. On remplace la virgule par un point
            df['Cote'] = df['Cote'].str.replace(',', '.', regex=False)
            # 4. On convertit en nombre décimal (float). Les erreurs deviennent 0.0
            df['Cote'] = pd.to_numeric(df['Cote'], errors='coerce').fillna(0.0)
            st.sidebar.success(f"✅ Cotes corrigées (Moyenne: {df['Cote'].mean():.2f})")
        
        # Conversion des autres nombres entiers
        for col in ['Dist', 'Nb_Partants', 'Num_PMU', 'Âge', 'Poids', 'Corde', 'Classement', 'Gains_Car', 'Réu', 'Course']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
        return None

def nettoyer_nom(nom):
    if pd.isna(nom): return ""
    return re.sub(r'[\s\.]', '', str(nom)).upper()

def calculer_score_ameliore(row, df_global, df_course):
    score = 0
    cheval_nom = nettoyer_nom(row.get('Cheval', ''))
    dist_actuelle = float(row.get('Dist', 0))
    jockey_actuel = nettoyer_nom(row.get('Jockey', ''))
    entraineur_actuel = nettoyer_nom(row.get('Entraîneur', ''))
    
    hist_cheval = df_global[df_global['Cheval'].apply(nettoyer_nom) == cheval_nom]
    
    # 1. FORME (20 pts)
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
        score += (np.mean(scores_forme) / 100) * 20
    else:
        score += 10 

    # 2. JOCKEY / ENTRAÎNEUR (25 pts)
    if jockey_actuel != "":
        victoires_jockey = len(hist_cheval[
            (hist_cheval['Jockey'].apply(nettoyer_nom) == jockey_actuel) & 
            (hist_cheval['Classement'] == 1)
        ])
        score += min(12.5, victoires_jockey * 5)
    
    if entraineur_actuel != "":
        victoires_entraineur = len(hist_cheval[
            (hist_cheval['Entraîneur'].apply(nettoyer_nom) == entraineur_actuel) & 
            (hist_cheval['Classement'] == 1)
        ])
        score += min(12.5, victoires_entraineur * 5)

    # 3. DISTANCE (15 pts)
    dist_min = dist_actuelle - 200
    dist_max = dist_actuelle + 200
    hist_distance = hist_cheval[(hist_cheval['Dist'] >= dist_min) & (hist_cheval['Dist'] <= dist_max)]
    victoires_dist = len(hist_distance[hist_distance['Classement'] == 1])
    podiums_dist = len(hist_distance[(hist_distance['Classement'] >= 1) & (hist_distance['Classement'] <= 3)])
    score += min(15, (victoires_dist * 10) + (podiums_dist * 3))

    # 4. GAINS (15 pts)
    gains = float(row.get('Gains_Car', 0))
    gains_max = df_course['Gains_Car'].max() if 'Gains_Car' in df_course.columns else 0
    if gains > 0 and gains_max > 0:
        score_gains = min(100, (gains / gains_max) * 100)
        score += (score_gains / 100) * 15
    else:
        score += 7.5

    # 5. POIDS (10 pts)
    poids = float(row.get('Poids', 0))
    poids_moyen = df_course['Poids'].mean() if 'Poids' in df_course.columns else 0
    if poids > 0 and poids_moyen > 0:
        ecart = poids_moyen - poids
        score_poids = max(0, min(100, 50 + (ecart * 5)))
        score += (score_poids / 100) * 10
    else:
        score += 5

    # 6. CORDE (10 pts)
    corde = float(row.get('Corde', 0))
    nb_partants = float(row.get('Nb_Partants', 16))
    if corde > 0 and nb_partants > 0:
        if corde <= nb_partants / 3: score += 10
        elif corde <= nb_partants * 2 / 3: score += 6
        else: score += 3
    else:
        score += 5

    # 7. COTE (5 pts) - Conversion forcée en float ici aussi pour être sûr
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

df = load_data()

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
                        "Score": float(cheval["Score"]), "Cote": float(cheval.get("Cote", 0))
                    })
        
        if recap_data:
            recap_df = pd.DataFrame(recap_data)
            st.subheader("🏆 Top 3 de chaque course")
            st.dataframe(recap_df, use_container_width=True, hide_index=True)

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
# PAGE 5: Score prédictif (AVEC DEBUG)
# ==========================================
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
            # 🔍 DEBUG : Vérifions ce que le script "voit" pour le 1er cheval
            st.markdown("### 🔍 Vérification des données (1er cheval de la liste)")
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
            
            st.info(f"💡 *Si l'historique est faible (ex: 1 ou 2 courses), les points 'Jockey/Entraîneur' et 'Distance' seront à 0. Il faut plus de données pour que l'IA brille !*")
            st.markdown("---")

            st.success(f"✅ Course du **{st.session_state.selected_date}**")
            
            # Calcul des scores
            parts["Score"] = parts.apply(lambda row: calculer_score_ameliore(row, df, parts), axis=1)
            parts = parts.sort_values("Score", ascending=False)
            parts["Rang"] = range(1, len(parts)+1)
            
            st.subheader("🏆 Classement")
            st.dataframe(parts[["Rang", "Num_PMU", "Cheval", "Score", "Cote", "Musique"]], use_container_width=True)
            
            fig = px.bar(parts, x="Cheval", y="Score", color="Score", color_continuous_scale="Viridis")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================
# PAGE 6: Recherche
# ==========================================
elif page == "🔍 Recherche cheval":
    st.header("🔍 Recherche un cheval")
    search = st.text_input("Nom du cheval :")
    if search:
        results = df[df["Cheval"].str.contains(search, case=False, na=False)]
        if not results.empty:
            st.dataframe(results[["Date", "Hippo", "Cheval", "Classement", "Cote"]].head(20), use_container_width=True)

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;font-size:12px'>🏇 Galop Analyzer</div>", unsafe_allow_html=True)