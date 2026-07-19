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
        df_historique = pd.read_csv(URL_CSV, on_bad_lines='skip', dtype={'Date': str})
        st.sidebar.success(f"✅ Historique: {len(df_historique)} lignes")
        try:
            df_today = pd.read_csv(URL_COURSES_JOUR, on_bad_lines='skip', dtype={'Date': str})
            st.sidebar.success(f"✅ Courses du jour: {len(df_today)} lignes")
            df = pd.concat([df_historique, df_today], ignore_index=True)
        except Exception as e:
            st.sidebar.error(f"❌ Erreur courses du jour: {e}")
            df = df_historique
        
        # IMPORTANT : Forcer Date en texte et nettoyer
        if 'Date' in df.columns:
            df['Date'] = df['Date'].astype(str).str.strip()
        
        for col in ['Cheval', 'Hippo']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        for col in ['Dist', 'Nb_Partants', 'Num_PMU', 'Âge', 'Poids', 'Corde', 'Classement', 'Gains_Car', 'Cote', 'Réu', 'Course']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        return df
    except Exception as e:
        st.error(f"Erreur: {e}")
        return None

def calculer_score(row, df_course):
    score = 0
    musique = str(row.get('Musique', ''))
    chiffres = re.findall(r'\d+', musique)
    if chiffres:
        scores = []
        for c in chiffres[:5]:
            val = int(c)
            if val == 1: scores.append(100)
            elif val == 2: scores.append(80)
            elif val == 3: scores.append(65)
            elif val == 4: scores.append(50)
            else: scores.append(30)
        score += (np.mean(scores) / 100) * 30
    else:
        score += 15
    
    gains = row.get('Gains_Car', 0)
    gains_max = df_course['Gains_Car'].max() if 'Gains_Car' in df_course.columns else 0
    if pd.notna(gains) and gains > 0 and pd.notna(gains_max) and gains_max > 0:
        score_gains = min(100, (gains / gains_max) * 100)
        score += (score_gains / 100) * 25
    else:
        score += 12.5
    
    poids = row.get('Poids', 0)
    poids_moyen = df_course['Poids'].mean() if 'Poids' in df_course.columns else 0
    if pd.notna(poids) and poids > 0 and pd.notna(poids_moyen) and poids_moyen > 0:
        ecart = poids_moyen - poids
        score_poids = max(0, min(100, 50 + (ecart * 5)))
        score += (score_poids / 100) * 15
    else:
        score += 7.5
    
    corde = row.get('Corde', 0)
    nb_partants = row.get('Nb_Partants', 16)
    if pd.notna(corde) and corde > 0 and pd.notna(nb_partants) and nb_partants > 0:
        if corde <= nb_partants / 3: score += 10
        elif corde <= nb_partants * 2 / 3: score += 6
        else: score += 3
    else:
        score += 5
    
    cote = row.get('Cote', 0)
    if pd.notna(cote) and cote > 0:
        if cote <= 3: score_cote = 100
        elif cote <= 6: score_cote = 80
        elif cote <= 10: score_cote = 60
        elif cote <= 20: score_cote = 40
        else: score_cote = 20
        score += (score_cote / 100) * 20
    else:
        score += 10
    
    score = max(0, min(100, score))
    return round(score, 1)

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
    
    # Date automatique d'aujourd'hui
    date_du_jour = datetime.now().strftime("%d%m%Y")
    courses_du_jour = df[df["Date"] == date_du_jour]
    
    if courses_du_jour.empty:
        st.warning(f"⚠️ Aucune course trouvée pour la date {date_du_jour}")
        dates_valides = [d for d in df["Date"].unique() if isinstance(d, str) and len(d) == 8]
        date_alt = st.selectbox("Autre date :", sorted(dates_valides, reverse=True))
        courses_du_jour = df[df["Date"] == date_alt]
        st.write(f"🔍 Courses trouvées pour {date_alt} : {len(courses_du_jour)}")
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
                parts["Score"] = parts.apply(lambda row: calculer_score(row, parts), axis=1)
                parts = parts.sort_values("Score", ascending=False)
                top3 = parts.head(3)
                
                for i, (_, cheval) in enumerate(top3.iterrows()):
                    recap_data.append({
                        "Hippodrome": hippo,
                        "Course": f"R{reu}C{num_course}",
                        "Distance": f"{dist}m",
                        "Rang": i + 1,
                        "Num": int(cheval["Num_PMU"]),
                        "Cheval": cheval["Cheval"],
                        "Score": cheval["Score"],
                        "Musique": str(cheval.get("Musique", "")),
                        "Cote": float(cheval.get("Cote", 0))
                    })
        
        if recap_data:
            recap_df = pd.DataFrame(recap_data)
            st.subheader("🏆 Top 3 de chaque course")
            st.dataframe(recap_df[["Hippodrome", "Course", "Rang", "Num", "Cheval", "Score", "Musique", "Cote"]], use_container_width=True, hide_index=True)
            
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
    
    # Filtrer les dates valides pour éviter l'erreur float vs str
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
        
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.info(f"**Date:** {date_sel}")
        with col2: st.info(f"**Hippodrome:** {info['Hippo']}")
        with col3: st.info(f"**Distance:** {info['Dist']}m")
        with col4: st.info(f"**Partants:** {len(parts)}")
        
        st.markdown("---")
        st.subheader("🐎 Partants")
        cols_affichage = ["Num_PMU", "Cheval", "Âge", "Sexe", "Poids", "Corde", "Musique", "Cote", "Classement", "Gains_Car"]
        cols_affichage = [c for c in cols_affichage if c in parts.columns]
        st.dataframe(parts[cols_affichage], use_container_width=True)

# ==========================================
# PAGE 4: Statistiques chevaux
# ==========================================
elif page == "🐎 Statistiques chevaux":
    st.header("🐎 Statistiques des chevaux")
    
    if st.session_state.selected_date is None:
        st.warning("⚠️ Va d'abord dans **🏆 Analyse d'une course** pour sélectionner une course !")
    else:
        parts = df[
            (df["Date"] == str(st.session_state.selected_date)) & 
            (df["Réu"] == int(st.session_state.selected_reu)) & 
            (df["Course"] == int(st.session_state.selected_course))
        ]
        
        if parts.empty:
            st.error("❌ Aucun partant trouvé pour cette course")
        else:
            st.success(f"✅ Course du **{st.session_state.selected_date}** - R{st.session_state.selected_reu}C{st.session_state.selected_course}")
            
            st.subheader("📊 Partants")
            st.dataframe(parts[["Num_PMU", "Cheval", "Âge", "Musique", "Poids", "Corde", "Gains_Car"]], use_container_width=True)
            
            st.markdown("---")
            st.subheader("🔍 Détail d'un cheval")
            cheval_choisi = st.selectbox("Sélectionne un cheval :", parts["Cheval"].unique())
            
            if cheval_choisi:
                historique = df[df["Cheval"] == cheval_choisi].sort_values("Date", ascending=False)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1: st.metric("Courses", len(historique))
                with col2: st.metric("Victoires", int((historique["Classement"] == 1).sum()))
                with col3:
                    taux = int((historique["Classement"] == 1).sum()) / len(historique) * 100 if len(historique) > 0 else 0
                    st.metric("Taux", f"{taux:.1f}%")
                with col4: st.metric("Gains max", f"{int(historique['Gains_Car'].max()):,}€")
                
                if len(historique) > 0:
                    st.subheader("📅 Historique")
                    cols = ["Date", "Hippo", "Dist", "Classement", "Cote", "Gains_Car", "Musique"]
                    cols = [c for c in cols if c in historique.columns]
                    st.dataframe(historique[cols], use_container_width=True)

# ==========================================
# PAGE 5: Score prédictif
# ==========================================
elif page == "🎯 Score prédictif":
    st.header("🎯 Score prédictif")
    st.info("💡 Score sur 100 pts")
    
    if st.session_state.selected_date is None:
        st.warning("⚠️ Va d'abord dans **🏆 Analyse d'une course** !")
    else:
        parts = df[
            (df["Date"] == str(st.session_state.selected_date)) & 
            (df["Réu"] == int(st.session_state.selected_reu)) & 
            (df["Course"] == int(st.session_state.selected_course))
        ].copy()
        
        if not parts.empty:
            st.success(f"✅ Course du **{st.session_state.selected_date}**")
            parts["Score"] = parts.apply(lambda row: calculer_score(row, parts), axis=1)
            parts = parts.sort_values("Score", ascending=False)
            parts["Rang"] = range(1, len(parts)+1)
            
            st.subheader("🏆 Classement")
            cols = ["Rang", "Num_PMU", "Cheval", "Score", "Cote", "Musique"]
            cols = [c for c in cols if c in parts.columns]
            st.dataframe(parts[cols], use_container_width=True)
            
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
            st.success(f"✅ {len(results)} course(s) trouvée(s)")
            st.dataframe(results[["Date", "Hippo", "Dist", "Cheval", "Classement", "Cote"]].head(20), use_container_width=True)
        else:
            st.warning("Aucun résultat")

st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;font-size:12px'>🏇 Galop Analyzer</div>", unsafe_allow_html=True)