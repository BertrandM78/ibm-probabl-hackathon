# EDA — Méthodes / modèles alternatifs (référence e022b, CV 2.778)

Scripts : `lab/scratch/meth/` — `m0_dump.py` (features e022b du fold K, fit sur le train du fold seulement),
`m1_resid.py` (structure des résidus OOF e022b), `m2_gbm.py` (HGB / LightGBM), `m3_seq.py` (réseaux par
patient : BiGRU / Transformer / MLP témoin), `m4_blend.py`, `m5_blend3.py`. Logs `*.log` dans le même dossier.

## 0. Environnement

- `.venv` : ni torch, ni LightGBM utilisable (libomp absent), ni XGBoost, ni CatBoost.
- **torch 2.14.1 (cp314, arm64) s'installe par pip** (roue de 127 Mo). Il est installé dans un venv à part
  (venv temporaire), `.venv` n'a pas été touché.
- torch embarque `torch/lib/libomp.dylib` : avec `DYLD_LIBRARY_PATH=<site-packages>/torch/lib`,
  **LightGBM 4.7 et XGBoost 3.4 se chargent** sans `brew install libomp`. CatBoost 1.2.10 s'installe aussi.

## 1. Où e022b se trompe (OOF, 5 folds)

| Composante de l'erreur (MSE 7.72) | MSE | part |
|---|---|---|
| Biais moyen par patient | 4.88 | 63 % |
| Pente linéaire intra-patient | 1.84 | 24 % |
| Reste intra-patient | 1.00 | 13 % |

- Autocorrélation lag-1 du résidu intra-patient : 0.725 (0.91 en brut). L'erreur est une **erreur de trajectoire
  patient**, pas un bruit propre à chaque visite.
- Selon la position : 1re visite 2.88, milieu 2.63, **dernière 3.46**. Selon le nombre de visites : 4–5 → 2.90, 8–9 → 2.59.
- Les 10 % de patients les plus mal prédits font 52 % de la SSE. Il y a un rétrécissement vers la moyenne :
  biais +0.3/+0.5 sous target 33, et −1.49 dans le décile du haut (RMSE 4.13).

## 2. Sondes (folds 0 et 3, features e022b identiques, reconstruction niveau + delta λ = 4)

Référence e022b : fold 0 2.7448, fold 3 2.9050. Le pipeline HGB rejoué depuis le dump redonne 2.7448 à 1e-4.

| Modèle | fold 0 seul | ρ résidus | 50/50 Δ | fold 3 seul | ρ | 50/50 Δ |
|---|---|---|---|---|---|---|
| HGB v2 e018 (bag 3) | 2.688 (−0.057) | 0.979 | −0.043 | – | | |
| LightGBM l2, 63 feuilles, lr 0.03 × 1500, col 0.3, sub 0.8 | 2.698 (−0.047) | 0.975 | −0.040 | 2.844 (−0.061) | 0.977 | −0.047 |
| LightGBM l2, 127 feuilles, lr 0.01 × 5000 | 2.692 (−0.053) | 0.973 | −0.045 | – | | |
| LightGBM Huber (α = 4) | 2.681 (−0.063) | 0.973 | −0.050 | 2.861 (−0.044) | 0.975 | −0.040 |
| MLP par visite (témoin sans séquence), 5 seeds | 2.704 | 0.906 | −0.085 | – | | |
| **BiGRU 2 couches, 5 seeds** | **2.584** | 0.837 | −0.191 | **2.806** | 0.861 | −0.151 |
| **Transformer 2 couches (sans encodage de position), 5 seeds** | **2.571** | 0.859 | −0.183 | **2.782** | 0.881 | −0.148 |
| Transformer, cible target − hb_post_full | 2.561 | 0.865 | −0.184 | – | | |
| Transformer « amortized », 18 colonnes brutes/harmonisées seulement, 3 seeds | 2.774 | 0.820 | −0.114 | – | | |

Réseaux par patient (`m3_seq.py`) : entrée = matrice dense (one-hot, flags NaN, médiane, standardisation,
clip ±6, tout fitté sur le train du fold ; 578 colonnes), séquence = toutes les visites du patient (≤ 12, masque).
Deux têtes (niveau standardisé + delta, perte niveau + 0.5·delta), AdamW 2e-3, OneCycle 25 epochs, batch de
64 patients. L'epoch est choisie sur 10 % des patients du train (tirage différent pour chaque seed). Aucun
réglage n'a été fait sur la valid. Coût : ~1 min par seed et par fold sur CPU.

Moyenne GRU + Transformer, mélangée à e022b :

| | seq seul | ρ | w_seq 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|
| fold 0 | 2.541 | 0.860 | −0.196 | −0.213 | **−0.223** |
| fold 3 | 2.761 | 0.881 | −0.158 | −0.169 | **−0.173** |

- **Effet propre de la séquence** : à features identiques, MLP par visite 2.704 contre GRU 2.584 / Transformer 2.571,
  soit −0.12 dû à la couche séquentielle seule.
- D'où vient le gain (fold 0, mélange 0.6) : MSE intra-patient 2.81 → 2.21, biais patient 4.72 → 4.20.
  Par position : 1re visite 2.96 → 2.72, milieu 2.60 → 2.37, dernière 3.30 → 3.16. Fold 3 : mêmes sens.
- La moyenne de 3 réseaux (GRU, Transformer, Transformer offset) fait 2.509 seule, et −0.237 mélangée à 0.7.
  Une NNLS fittée sur la valid (optimiste) donne poids séquence 0.73, réseau amortized 0.21, e022b 0.06 :
  −0.266. Le réseau amortized (ρ 0.82 avec la séquence complète) apporte de la diversité.
- LightGBM : à peu près −0.05 comme simple remplaçant de HGB (ρ 0.975). Dans le mélange séquence + LightGBM,
  il ne gagne que 0.01 à 0.02 de plus que séquence + e022b (fold 0 −0.228, fold 3 −0.187 à w 0.7).
- Réseau amortized avec 100 epochs au lieu de 40 : pas mieux (2.774). Ce qui le limite, ce sont les entrées, pas le temps d'entraînement.

## 3. Limites

- 2 folds sur 5. La variance entre seeds est forte (une seed GRU va de 2.72 à 2.85 sur le fold 0) : il faut 5 seeds au minimum.
- La soumission dépendrait de torch, et le déterminisme CPU est à vérifier (`torch.use_deterministic_algorithms`).
- HGB v2 (e018 : 127 feuilles, lr 0.03 × 1500, max_features 0.3, 3 seeds) porté sur e022b, fold 0 :
  2.688 (−0.057), ρ 0.979, 50/50 −0.043. C'est le même gain que LightGBM l2 (−0.047 à −0.053), mais
  ~30× plus lent (2 500 s sur CPU partagé contre 86 s). Pour un « meilleur GBM », LightGBM est le bon choix.
