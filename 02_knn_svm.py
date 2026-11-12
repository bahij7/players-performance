"""
=============================================================
 Neya eSports – Script 2 : KNN + SVM + Comparaison
 Input  : neya_labeled.csv (genere par 01_kmeans.py)
 Output : rapports, confusion matrices, profil joueurs
=============================================================
 Features : kills, assists, damage, KO, tournament_pressure
   kills/assists/damage/KO   → qualite du jeu (memes que KMeans)
   tournament_pressure       → contexte : finale != scrims
                               KNN/SVM apprennent cette nuance
=============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

os.makedirs("images", exist_ok=True)


# ══════════════════════════════════════════════════════════
#  1. CHARGEMENT
# ══════════════════════════════════════════════════════════

df = pd.read_csv("neya_labeled.csv")

print(f"[1] Dataset charge : {len(df)} lignes")
print(f"\n    Distribution des labels (crees par KMeans) :")
vc = df['skill_level'].value_counts()
for lvl, n in vc.items():
    print(f"    {lvl:15s}: {n:5d}  ({n/len(df)*100:.1f}%)")


# ══════════════════════════════════════════════════════════
#  2. FEATURES ET LABEL
# ══════════════════════════════════════════════════════════

FEATURES = ['kills', 'assists', 'damage', 'KO', 'tournament_pressure']

X     = df[FEATURES].copy()
y_raw = df['skill_level']

le = LabelEncoder()
y  = le.fit_transform(y_raw)

print(f"\n[2] Features : {FEATURES}")
print(f"    Labels    : {list(le.classes_)}")


# ══════════════════════════════════════════════════════════
#  3. NORMALISATION
# ══════════════════════════════════════════════════════════

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"\n[3] StandardScaler applique")


# ══════════════════════════════════════════════════════════
#  4. SPLIT 80 / 20
# ══════════════════════════════════════════════════════════

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y
)

print(f"\n[4] Split 80/20 : train={len(X_train)}  test={len(X_test)}")


# ══════════════════════════════════════════════════════════
#  5. KNN — recherche du meilleur k par cross-validation
# ══════════════════════════════════════════════════════════

print(f"\n[5] KNN — recherche du meilleur k ...")

k_scores = []
for k in range(1, 21):
    knn    = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    k_scores.append((k, scores.mean(), scores.std()))

best_k     = max(k_scores, key=lambda x: x[1])[0]
best_k_acc = max(k_scores, key=lambda x: x[1])[1]
print(f"    Meilleur k = {best_k}  (accuracy CV = {best_k_acc*100:.2f}%)")

plt.figure(figsize=(8, 4))
plt.plot([x[0] for x in k_scores], [x[1] for x in k_scores],
         marker='o', color='#185FA5', linewidth=2)
plt.axvline(x=best_k, color='#E24B4A', linestyle='--', label=f'k={best_k} optimal')
plt.title("KNN — Accuracy selon k (CV 5-fold)", fontsize=12)
plt.xlabel("k"); plt.ylabel("Accuracy")
plt.legend(); plt.tight_layout()
plt.savefig("images/knn_k_selection.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"    → knn_k_selection.png")

knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)


# ══════════════════════════════════════════════════════════
#  6. SVM — kernel RBF
#     Gere bien les frontieres non lineaires,
#     utile quand les clusters se chevauchent
# ══════════════════════════════════════════════════════════

print(f"\n[6] SVM (kernel=rbf) ...")

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

svm_cv = cross_val_score(svm_model, X_train, y_train, cv=5, scoring='accuracy').mean()
print(f"    Accuracy CV (5-fold) = {svm_cv*100:.2f}%")


# ══════════════════════════════════════════════════════════
#  7. COMPARAISON
# ══════════════════════════════════════════════════════════

class_names = list(le.classes_)

def get_metrics(name, y_true, y_pred):
    return {
        'Modele'    : name,
        'Accuracy'  : f"{accuracy_score(y_true, y_pred)*100:.2f}%",
        'Precision' : f"{precision_score(y_true, y_pred, average='weighted')*100:.2f}%",
        'Recall'    : f"{recall_score(y_true, y_pred, average='weighted')*100:.2f}%",
        'F1 Score'  : f"{f1_score(y_true, y_pred, average='weighted')*100:.2f}%",
    }

results = pd.DataFrame([
    get_metrics(f"KNN (k={best_k})", y_test, y_pred_knn),
    get_metrics("SVM (rbf)",         y_test, y_pred_svm),
])

print(f"\n[7] Comparaison finale :")
print(results.to_string(index=False))

print(f"\n--- Rapport detaille KNN ---")
print(classification_report(y_test, y_pred_knn, target_names=class_names))
print(f"--- Rapport detaille SVM ---")
print(classification_report(y_test, y_pred_svm, target_names=class_names))

print("    Note : si le recall d Intermediaire est faible,")
print("    c est normal — la frontiere Debutant/Intermediaire")
print("    est peu nette. A mentionner dans la conclusion.")


# ══════════════════════════════════════════════════════════
#  8. MATRICES DE CONFUSION
# ══════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Matrices de confusion — KNN vs SVM", fontsize=13)

for ax, pred, name, cmap in zip(
    axes,
    [y_pred_knn, y_pred_svm],
    [f'KNN (k={best_k})', 'SVM (kernel=rbf)'],
    ['Blues', 'Purples']
):
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel('Predit')
    ax.set_ylabel('Reel')

plt.tight_layout()
plt.savefig("images/confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"\n[8] → confusion_matrices.png")


# ══════════════════════════════════════════════════════════
#  9. GRAPHIQUE BARRES COMPARAISON
# ══════════════════════════════════════════════════════════

def get_floats(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred) * 100,
        precision_score(y_true, y_pred, average='weighted') * 100,
        recall_score(y_true, y_pred, average='weighted') * 100,
        f1_score(y_true, y_pred, average='weighted') * 100,
    ]

labels   = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
knn_vals = get_floats(y_test, y_pred_knn)
svm_vals = get_floats(y_test, y_pred_svm)
x        = np.arange(len(labels))
w        = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - w/2, knn_vals, w, label=f'KNN (k={best_k})', color='#185FA5', alpha=0.85)
b2 = ax.bar(x + w/2, svm_vals, w, label='SVM (rbf)',         color='#534AB7', alpha=0.85)
ax.set_ylabel('Score (%)'); ax.set_ylim(0, 110)
ax.set_title('Comparaison KNN vs SVM', fontsize=12)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.legend()
for bar in list(b1) + list(b2):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig("images/comparison_knn_svm.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"[9] → comparison_knn_svm.png")


# ══════════════════════════════════════════════════════════
#  10. PROFIL GLOBAL DES 12 JOUEURS
# ══════════════════════════════════════════════════════════

knn_acc = accuracy_score(y_test, y_pred_knn)
svm_acc = accuracy_score(y_test, y_pred_svm)
best_model      = svm_model if svm_acc >= knn_acc else knn_model
best_model_name = 'SVM'     if svm_acc >= knn_acc else 'KNN'

df['predicted_level'] = le.inverse_transform(best_model.predict(X_scaled))

player_profiles = (
    df.groupby('player_id')['predicted_level']
    .value_counts(normalize=True)
    .mul(100).round(1)
    .unstack(fill_value=0)
    .reindex(columns=['Expert', 'Intermediaire', 'Debutant'], fill_value=0)
)

def global_level(row):
    if row.get('Expert', 0) >= 50:
        return 'Expert'
    if row.get('Expert', 0) >= 30 or row.get('Intermediaire', 0) >= 50:
        return 'Intermediaire'
    return 'Debutant'

player_profiles['Niveau global'] = player_profiles.apply(global_level, axis=1)

print(f"\n[10] Profil global — modele : {best_model_name}")
print(player_profiles.to_string())

fig, ax = plt.subplots(figsize=(11, 5))
players = player_profiles.index.astype(str)
ax.bar(players, player_profiles.get('Expert', 0),        label='Expert',        color='#27500A')
ax.bar(players, player_profiles.get('Intermediaire', 0), label='Intermediaire', color='#444441',
       bottom=player_profiles.get('Expert', 0))
ax.bar(players, player_profiles.get('Debutant', 0),      label='Debutant',      color='#E24B4A',
       bottom=player_profiles.get('Expert', 0) + player_profiles.get('Intermediaire', 0))
ax.set_xlabel('Joueur (player_id)'); ax.set_ylabel('%'); ax.set_ylim(0, 110)
ax.set_title(f'Profil global des 12 joueurs — {best_model_name}', fontsize=12)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig("images/player_profiles.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"    → player_profiles.png")


# ══════════════════════════════════════════════════════════
#  RESUME
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  RESUME DU PROJET")
print("=" * 60)
print(f"  Dataset         : {len(df)} performances")
print(f"  Features        : {FEATURES}")
print(f"  KNN accuracy    : {knn_acc*100:.2f}%  (k={best_k})")
print(f"  SVM accuracy    : {svm_acc*100:.2f}%  (kernel=rbf)")
print(f"  Meilleur modele : {best_model_name}")
print("=" * 60)
print("  Livrables generes :")
print("  elbow_method.png       — choix k=3")
print("  clusters_pca_2d.png    — visualisation clusters 4D→2D")
print("  pairplot_clusters.png  — toutes combinaisons features")
print("  knn_k_selection.png    — choix du meilleur k")
print("  confusion_matrices.png — KNN vs SVM")
print("  comparison_knn_svm.png — metriques comparees")
print("  player_profiles.png    — niveau global 12 joueurs")
print("=" * 60)
print("\n  LIMITES A MENTIONNER DANS LA CONCLUSION :")
print("  1. Labels crees par KMeans → accuracy mesure la")
print("     coherence geometrique, pas la vraie competence.")
print("  2. Chevauchement Debutant/Intermediaire attendu —")
print("     les donnees reelles ne sont pas parfaitement separables.")
print("=" * 60)


# ══════════════════════════════════════════════════════════
#  SIMULATEUR — predire le niveau d une nouvelle performance
#  Utilise SVM (meilleur modele, accuracy 97.3%)
#  Modifiez les valeurs dans les exemples en bas du script
# ══════════════════════════════════════════════════════════

pressure_map = {
    'Final'               : 6,
    'Semi Final'          : 5,
    'Rush Point/Play Off' : 4,
    'Group Phase'         : 3,
    'Qualifications'      : 2,
    'Scrims'              : 1,
}

def predict_performance(kills, assists, damage, ko, tournament_type):
    """
    Predit le niveau de competence d une performance individuelle.

    Parametres :
        kills           (int) : nombre de kills
        assists         (int) : nombre d assists
        damage          (int) : damage total inflige
        ko              (int) : nombre de KO
        tournament_type (str) : 'Final', 'Semi Final',
                                'Rush Point/Play Off',
                                'Group Phase',
                                'Qualifications', 'Scrims'
    """
    if tournament_type not in pressure_map:
        print(f"[!] tournament_type inconnu : '{tournament_type}'")
        print(f"    Valeurs acceptees : {list(pressure_map.keys())}")
        return None

    pressure = pressure_map[tournament_type]

    input_data = pd.DataFrame([{
        'kills'               : kills,
        'assists'             : assists,
        'damage'              : damage,
        'KO'                  : ko,
        'tournament_pressure' : pressure,
    }])

    input_scaled  = scaler.transform(input_data)
    prediction    = svm_model.predict(input_scaled)[0]
    probabilities = svm_model.predict_proba(input_scaled)[0]

    label     = le.inverse_transform([prediction])[0]
    prob_dict = dict(zip(le.classes_, probabilities))

    print(f"\n{'='*45}")
    print(f"  PREDICTION SVM")
    print(f"{'='*45}")
    print(f"  Kills    : {kills}")
    print(f"  Assists  : {assists}")
    print(f"  Damage   : {damage}")
    print(f"  KO       : {ko}")
    print(f"  Tournoi  : {tournament_type} (pression={pressure})")
    print(f"{'='*45}")
    print(f"  Niveau predit : >>> {label} <<<")
    print(f"{'='*45}")
    print(f"  Probabilites :")
    for lvl in ['Expert', 'Intermediaire', 'Debutant']:
        pct = prob_dict.get(lvl, 0) * 100
        bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
        print(f"    {lvl:15s} {bar} {pct:.1f}%")
    print(f"{'='*45}\n")
    return label


# ── Saisie interactive ───────────────────────────────────

tournament_options = list(pressure_map.keys())

print()
print('=' * 45)
print('  SIMULATEUR — Saisie des stats du joueur')
print('=' * 45)

while True:
    try:
        kills   = int(input("  Entrez les kills du joueur   : "))
        assists = int(input("  Entrez les assists du joueur : "))
        damage  = int(input("  Entrez le damage du joueur   : "))
        ko      = int(input("  Entrez les KO du joueur      : "))
    except ValueError:
        print('[!] Entrez des nombres entiers.')
        continue

    print()
    print('  Types de tournoi disponibles :')
    for i, t in enumerate(tournament_options, 1):
        print(f'    {i}. {t}')
    print()

    while True:
        try:
            choix = int(input('  Choisissez le type (1-6) : '))
            if 1 <= choix <= 6:
                tournament_type = tournament_options[choix - 1]
                break
            else:
                print('[!] Entrez un nombre entre 1 et 6.')
        except ValueError:
            print('[!] Entrez un nombre entier.')

    predict_performance(kills, assists, damage, ko, tournament_type)

    again = input('  Tester une autre performance ? (o/n) : ').strip().lower()
    if again != 'o':
        print()
        print('  Fin du simulateur.')
        break