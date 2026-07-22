import pandas as pd
from datetime import datetime
import os

URL_COURSES_JOUR = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQJugx0HS5vID0MHWLRO-5GYEBtb1vmJXvZrYPLfI4x6avcitpRO7dtfRE9WxK3UwZRpzx-59MRicxV/pub?gid=365561583&single=true&output=csv"

COLONNES = ['Date', 'Réu', 'Course', 'Hippo', 'Dist', 'Disc', 'Spécialité', 'Terrain', 
            'Nb_Partants', 'Num_PMU', 'Cheval', 'Âge', 'Sexe', 'Jockey', 'Entraîneur', 
            'Poids', 'Corde', 'Musique', 'Cote', 'Classement', 'Gains_Car']

print("📋 Génération du template de cotes du jour...")

df = pd.read_csv(URL_COURSES_JOUR, header=None, names=COLONNES, on_bad_lines='skip', dtype={'Date': str})
df = df[df['Date'] != 'Date']
df['Date'] = df['Date'].astype(str).str.strip()

date_jour = datetime.now().strftime("%d%m%Y")
df_jour = df[df['Date'] == date_jour]

if df_jour.empty:
    print(f"⚠️ Aucune course trouvée pour aujourd'hui ({date_jour}) dans le flux.")
else:
    # On garde juste les infos nécessaires pour identifier le cheval
    template = df_jour[['Hippo', 'Course', 'Num_PMU', 'Cheval']].copy()
    template['Cote'] = ""  # Colonne vide à remplir
    
    nom_fichier = f"cotes_{date_jour}.csv"
    template.to_csv(nom_fichier, index=False, sep=';')
    
    print(f"✅ Template créé : {nom_fichier}")
    print(f"📊 {len(template)} chevaux à coter")
    print("\n📝 Mode d'emploi :")
    print("1. Ouvre ce fichier dans Excel")
    print("2. Remplis la colonne 'Cote' (ex: 4.5) avec les cotes du site PMU/France Galop")
    print("3. Sauvegarde (ne change pas le nom du fichier ni le séparateur ';')")
    print("4. Lance ton app Streamlit : elle utilisera automatiquement tes cotes !")