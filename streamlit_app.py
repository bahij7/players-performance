"""
Neya eSports - Streamlit : prediction du niveau d'un joueur (SVM)
Meme pipeline que le script terminal (class-knn,svm.py).
Lancer :  streamlit run streamlit_app.py
"""
import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

PRESSURE_MAP = {
    'Final': 6,
    'Semi Final': 5,
    'Rush Point/Play Off': 4,
    'Group Phase': 3,
    'Qualifications': 2,
    'Scrims': 1,
}
FEATURES = ['kills', 'assists', 'damage', 'KO', 'tournament_pressure']
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def train_model():
    df = pd.read_csv(os.path.join(BASE_DIR, 'neya_labeled.csv'))
    le = LabelEncoder()
    y = le.fit_transform(df['skill_level'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES])
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = SVC(kernel='rbf', C=1.0, gamma='scale',
                probability=True, random_state=42)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, scaler, le, acc


svm_model, scaler, le, accuracy = train_model()

st.set_page_config(page_title="Player' Performance..", page_icon="🎮")
st.title("🎮 BAHIJ – Niveau du joueur")
st.caption("Prédiction SVM à partir des stats et du type de tournoi")

col1, col2 = st.columns(2)
kills = col1.number_input("Kills", min_value=0, value=0, step=1)
assists = col2.number_input("Assists", min_value=0, value=0, step=1)
damage = col1.number_input("Damage", min_value=0, value=0, step=1)
ko = col2.number_input("KO", min_value=0, value=0, step=1)
tournament = st.selectbox("Type de tournoi", list(PRESSURE_MAP.keys()))

if st.button("Prédire le niveau", type="primary", use_container_width=True):
    pressure = PRESSURE_MAP[tournament]
    row = pd.DataFrame([{'kills': kills, 'assists': assists, 'damage': damage,
                         'KO': ko, 'tournament_pressure': pressure}])
    scaled = scaler.transform(row)
    pred = svm_model.predict(scaled)[0]
    probs = dict(zip(le.classes_, svm_model.predict_proba(scaled)[0]))
    label = le.inverse_transform([pred])[0]

    st.divider()
    msg = f"Niveau prédit : **{label}**  (pression = {pressure})"
    if label == 'Expert':
        st.success(msg)
    elif label == 'Intermediaire':
        st.warning(msg)
    else:
        st.error(msg)

    for lvl in ['Expert', 'Intermediaire', 'Debutant']:
        pct = float(probs.get(lvl, 0)) * 100
        st.write(f"{lvl} — {pct:.1f}%")
        st.progress(min(int(round(pct)), 100))

st.caption(f"Accuracy du modèle (test) : {accuracy*100:.1f}%")
