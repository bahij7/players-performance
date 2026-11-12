"""
=============================================================
 Neya eSports – Script 1 : K-Means Labeling
 Output : neya_labeled.csv  (pret pour KNN et SVM)
=============================================================
 Features KMeans      : kills, assists, damage, KO
   → qualite du jeu uniquement, pas le contexte
   → un 0 kills en finale reste Debutant

 tournament_pressure  : extrait ici, transporte dans le CSV
   → PAS utilise par KMeans
   → utilise par KNN et SVM uniquement
   → finale != scrims pour la prediction

 PCA : visualisation uniquement apres KMeans
   → 4D en 2D pour afficher les clusters
   → ne modifie rien au pipeline ML
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()  # lit le fichier .env (non envoye sur GitHub)

os.makedirs("images", exist_ok=True)


# ══════════════════════════════════════════════════════════
#  1. CONNEXION ET EXTRACTION
# ══════════════════════════════════════════════════════════

conn = mysql.connector.connect(
    host     = os.getenv("DB_HOST", "localhost"),
    user     = os.getenv("DB_USER"),
    password = os.getenv("DB_PASSWORD"),
    database = os.getenv("DB_NAME", "neya")
)

query = """
    SELECT
        ps.player_id,
        ps.kills,
        ps.assists,
        ps.damage,
        ps.KO,
        t.type AS tournament_type
    FROM players_stats ps
    JOIN matches     m ON ps.match_id     = m.id
    JOIN tournaments t ON m.tournament_id = t.id
    ORDER BY ps.player_id, ps.match_id
"""

df = pd.read_sql(query, conn)
conn.close()

print(f"[1] Extraction : {df.shape[0]} lignes x {df.shape[1]} colonnes")
print(df.head(6).to_string(index=False))


# ══════════════════════════════════════════════════════════
#  2. VERIFICATION NULL
#     0 kills / 0 damage = performance valide, on garde tout
# ══════════════════════════════════════════════════════════

nulls = df.isnull().sum()
if nulls.sum() > 0:
    print(f"\n[!] Valeurs NULL detectees :")
    print(nulls[nulls > 0].to_string())
else:
    print(f"\n[2] Aucune valeur NULL — {len(df)} lignes conservees intactes")


# ══════════════════════════════════════════════════════════
#  3. ENCODAGE ORDINAL — tournament_pressure
#     Extrait ici pour le transporter dans le CSV
#     KMeans ne l utilise PAS
#     KNN et SVM l utiliseront dans le script 2
# ══════════════════════════════════════════════════════════

pressure_map = {
    'Final'               : 6,
    'Semi Final'          : 5,
    'Rush Point/Play Off' : 4,
    'Group Phase'         : 3,
    'Qualifications'      : 2,
    'Scrims'              : 1,
}

df['tournament_pressure'] = df['tournament_type'].map(pressure_map)

unrecognized = df[df['tournament_pressure'].isna()]['tournament_type'].unique()
if len(unrecognized) > 0:
    print(f"\n[!] Types non reconnus dans pressure_map : {unrecognized}")
    print("    Ajoutez-les et relancez.")
    exit(1)

df['tournament_pressure'] = df['tournament_pressure'].astype(int)

print(f"\n[3] Encodage ordinal tournament_pressure :")
for t, v in sorted(pressure_map.items(), key=lambda x: -x[1]):
    count = (df['tournament_type'] == t).sum()
    print(f"    {v}  {t:25s}  ({count} matchs)")


# ══════════════════════════════════════════════════════════
#  4. FEATURES ET NORMALISATION (KMeans uniquement)
# ══════════════════════════════════════════════════════════

FEATURES = ['kills', 'assists', 'damage', 'KO']
X = df[FEATURES].copy()

print(f"\n[4] Features KMeans (sans tournament_pressure) : {FEATURES}")
print(X.describe().round(2).to_string())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n    StandardScaler applique")


# ══════════════════════════════════════════════════════════
#  5. ELBOW METHOD
# ══════════════════════════════════════════════════════════

inertias = []
k_range  = range(2, 9)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 4))
plt.plot(list(k_range), inertias, marker='o', color='#3266ad', linewidth=2)
plt.axvline(x=3, color='#E24B4A', linestyle='--', label='k=3 choisi')
plt.title("Elbow Method – Choix du nombre de clusters", fontsize=12)
plt.xlabel("Nombre de clusters k")
plt.ylabel("Inertie (WCSS)")
plt.legend()
plt.tight_layout()
plt.savefig("images/elbow_method.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[5] Elbow method → elbow_method.png")


# ══════════════════════════════════════════════════════════
#  6. KMEANS k=3
# ══════════════════════════════════════════════════════════

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)
df['cluster'] = kmeans.labels_

centroids_real = pd.DataFrame(
    scaler.inverse_transform(kmeans.cluster_centers_),
    columns=FEATURES
)
centroids_real['cluster'] = range(3)
centroids_sorted = centroids_real.sort_values('damage').reset_index(drop=True)

level_map = {
    int(centroids_sorted.loc[0, 'cluster']): 'Debutant',
    int(centroids_sorted.loc[1, 'cluster']): 'Intermediaire',
    int(centroids_sorted.loc[2, 'cluster']): 'Expert',
}

df['skill_level'] = df['cluster'].map(level_map)

print(f"\n[6] K-Means applique (k=3) — features : {FEATURES}")
print(f"\n    Centroides (valeurs originales) :")
print(
    centroids_sorted
    .assign(skill_level=lambda x: x['cluster'].map(level_map))
    [['skill_level'] + FEATURES]
    .to_string(index=False)
)

print(f"\n    Distribution des niveaux :")
vc = df['skill_level'].value_counts()
for lvl, n in vc.items():
    print(f"    {lvl:15s}: {n:5d}  ({n/len(df)*100:.1f}%)")


# ══════════════════════════════════════════════════════════
#  7. PCA 2D — VISUALISATION UNIQUEMENT
#     On compresse 4 features en 2 composantes
#     pour pouvoir afficher un vrai nuage de points 2D
#     qui reflète la separation dans l espace complet
#     La PCA ne modifie pas le KMeans ni le pipeline ML
# ══════════════════════════════════════════════════════════

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

var1 = pca.explained_variance_ratio_[0] * 100
var2 = pca.explained_variance_ratio_[1] * 100
total_var = var1 + var2

print(f"\n[7] PCA 2D (visualisation uniquement) :")
print(f"    PC1 explique {var1:.1f}% de la variance")
print(f"    PC2 explique {var2:.1f}% de la variance")
print(f"    Total       {total_var:.1f}% de l information conservee")

palette = {
    'Expert'        : '#27500A',
    'Intermediaire' : '#000000',
    'Debutant'      : '#E24B4A',
}
order = ['Debutant', 'Intermediaire', 'Expert']

plt.figure(figsize=(8, 6))
for lvl in order:
    mask = df['skill_level'] == lvl
    plt.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        label=lvl, color=palette[lvl], alpha=0.45, s=18
    )
plt.xlabel(f"PC1 ({var1:.1f}% variance)")
plt.ylabel(f"PC2 ({var2:.1f}% variance)")
plt.title("Clusters K-Means — Projection PCA 2D\n(4 features comprimes en 2 pour la visualisation)", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig("images/clusters_pca_2d.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"    → clusters_pca_2d.png")


# ══════════════════════════════════════════════════════════
#  8. PAIRPLOT — toutes les combinaisons 2 a 2
# ══════════════════════════════════════════════════════════

plot_df = df[FEATURES + ['skill_level']].copy()
g = sns.pairplot(
    plot_df,
    hue       = 'skill_level',
    hue_order = order,
    palette   = palette,
    plot_kws  = {'alpha': 0.4, 's': 15},
    diag_kind = 'kde',
)
g.figure.suptitle("Pairplot — Clusters K-Means", y=1.02, fontsize=12)
plt.savefig("images/pairplot_clusters.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[8] → pairplot_clusters.png")


# ══════════════════════════════════════════════════════════
#  9. EXPORT
#     player_id        : pour le profil final des joueurs
#     kills/assists/damage/KO : features ML
#     tournament_pressure     : pour KNN et SVM (pas KMeans)
#     skill_level      : label cree par KMeans
# ══════════════════════════════════════════════════════════

output_cols = ['player_id'] + FEATURES + ['tournament_pressure', 'skill_level']
df[output_cols].to_csv("neya_labeled.csv", index=False)

print(f"\n[9] Exporte → neya_labeled.csv  ({len(df)} lignes)")
print(f"    Colonnes : {output_cols}")
print(df[output_cols].head(6).to_string(index=False))

print("\n" + "=" * 60)
print("  KMeans termine. Lancer 02_knn_svm.py")
print("=" * 60)
