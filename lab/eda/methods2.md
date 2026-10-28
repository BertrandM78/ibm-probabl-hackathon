# EDA — Méthodes 2 : réseaux séquentiels, données limitées (réf. e023 2.610, b002 2.579)

Scripts : `lab/scratch/meth2/` — `a1_seeds.py` (courbe de bagging), `a2_resid.py` (résidus OOF), `p1.py` (sondes
GRU/TF fold 0/3 sur la matrice e022b `scratch/meth/f{K}.pkl`, tout fitté sur le train du fold), `a3_blend.py`,
`a4_ens.py`. Résultats bruts : `results.txt`, `log_f*.log`, `r{K}_*.pkl`.
Sondes : 5 seeds, rec = reconstruction λ = 4, fold 0 sauf mention. Réf. fold 0 : e022b 2.7448, e023 2.5411, b002 2.5211.

## 1. Où e023 / b002 se trompent (OOF 5 folds)

| | RMSE | MSE | niveau patient | pente | reste |
|---|---|---|---|---|---|
| e022b | 2.778 | 7.72 | 4.88 | 1.85 | 1.00 |
| e023 | 2.610 | 6.81 | 4.55 | 1.61 | 0.65 |
| b002 | 2.579 | 6.65 | **4.42 (66 %)** | 1.56 | 0.67 |

- Corr. erreur de niveau patient e022b/e023 0.88, intra-patient 0.82.
- Post-traitements sur la sortie (pas de fit) : projection cubique w = 1 2.6096 → 2.6103, quadratique w = 0.5 2.6102,
  max courant / règle ×1.02 ±0.0002 (22 baisses sur 44 590). **La forme est déjà juste → une tête structurée
  « coefficients cubiques + rule(P) » ne peut viser que le niveau, que la tête actuelle sert déjà. Piste fermée.**
- Sous-groupes b002 : n 4–5 visites 2.807 (e022b 2.897, le réseau n'y gagne que −0.03 contre −0.21 pour n ≥ 8),
  dernière visite 3.247, décile haut du target 3.779 (biais −1.32). Top 10 % patients = 53 % de la SSE.

## 2. Variance de seed et courbe d'apprentissage

| | 1 seed | 5 seeds | ∞ (extrapolé) |
|---|---|---|---|
| GRU fold 0 (niveau seul) | 2.775 | 2.606 | 2.561 |
| TF fold 0 | 2.795 | 2.613 | 2.565 |
| GRU fold 3 | 2.975 | 2.826 | 2.787 |

- Variance entre seeds 1.1–1.3 de MSE (≈ 17 % de la MSE) : réseau très sous-déterminé.
- Deux bags de 5 seeds GRU (42–46 contre 47–51) : rec 2.5839 contre 2.5585 → **bruit d'un bag de 5 ≈ ±0.025**.
  Toute sonde à 5 seeds non appariée sous 0.03 est du bruit.
- **Courbe d'apprentissage du réseau** (GRU, 50 % des patients du train) : 2.5839 → 2.6448 (+0.061 par division
  par 2), même pente que HGB (residual.md). Le réseau est limité par les données.

## 3. Sondes de méthode (fold 0, rec 5 seeds ; « +0.3 » = 0.7·b002 + 0.3·sonde, réf. « 5 seeds neuves » 2.5081)

| variante | seul | ρ résidus vs e023 | 0.7·b002 + 0.3·p |
|---|---|---|---|
| GRU base, seeds 42–46 (= e023) | 2.5839 | 0.986 | 2.5241 |
| **GRU base, seeds neuves 47–51** | 2.5585 | 0.970 | **2.5081 (réf. « plus de seeds »)** |
| weight decay 1e-2 | 2.5832 | 0.986 | 2.5239 |
| dropout de features 0.1 / 0.2 / 0.4 | 2.563 / 2.562 / 2.621 | 0.97 / 0.95 / 0.92 | 2.512 / 2.506 / 2.508 |
| dropout features 0.2, 50 epochs | 2.5842 | **0.933** | **2.5011** |
| dropout par bloc (t_*, trajectoire, L_*) 0.3, GRU / TF | 2.565 / 2.549 | 0.951 / 0.954 | 2.504 / **2.503** |
| dropout de visites 0.2 | 2.5810 | 0.968 | 2.5196 |
| perte sur la reconstruction (w 1, seeds 47–51 appariées) GRU / TF | 2.556 / 2.557 (base 2.559 / 2.550) | 0.97 | 2.509 / 2.510 |
| perte niveau patient (w 1, appariée) | 2.567 (base 2.559) | 0.970 | 2.512 |
| snapshots (moyenne epochs 12–24) | −0.005 à −0.017 seul | | ±0 dans un ensemble (a4) |
| pseudo-labels valid = OOF b002 (poids 1), GRU / TF | **2.545 / 2.543** | 0.987 | 2.520 / 2.520 |
| pseudo-labels, fold 3 GRU | 2.743 (base 2.806) | 0.990 | 2.7275 (b002 2.7317) |
| pseudo-labels = OOF e023 / e022b | 2.567 / 2.612 | | 2.519 / 2.533 |

- **Pseudo-labeling : faux gain.** −0.04 / −0.06 seul, mais c'est de la distillation de b002 (ρ 0.987–0.990) :
  dans le blend, pire que 5 seeds neuves (2.520 contre 2.508). Avec les labels e023 seuls : rien. Piste fermée
  (côté réseau ; les données non étiquetées valid/test n'apportent pas d'information au-delà du professeur).
- Pertes niveau patient / reconstruction, weight decay, dropout de visites : bruit (|Δ| apparié ≤ 0.009).
- Les régularisations fortes (dropout features 0.2 + 50 ep, dropout par bloc) n'améliorent pas le réseau seul,
  mais **décorrèlent** (ρ 0.93–0.95 contre 0.97 pour des seeds neuves) → apport d'ensemble.

## 4. Ensemble élargi (fold 0, `a4_ens.py`, partie séquence = moyenne, puis 0.72/0.28 avec e022b)

| partie séquence | seq seule | blend |
|---|---|---|
| e023 (10 réseaux) | 2.541 | 2.5211 |
| + 10 seeds neuves (GRU + TF base) | 2.521 | 2.5052 (−0.016) |
| + 10 régularisés (GRU fdrop 0.2 ep 50 + TF bdrop 0.3) | 2.504 | **2.4975 (−0.024)** |
| + 20 (neuves + régularisés) | 2.499 | **2.4927 (−0.028)** |
| idem avec snapshots | | +0.004 à +0.006 (inutile) |

Un seul fold : gain à confirmer sur 5 folds (bruit de bag ±0.02). Les poids 0.72/0.28 sont ceux de b002, non refittés.

## 5. Non sondé / couvert ailleurs

- NN end-to-end sur brutes : axe nn2 (`scratch/nn2/results.txt`) : TF + pos. sur brutes 3.057, ρ 0.72, NNLS3 −0.021
  (optimiste). Simulateur → axe augmentation (`scratch/aug`). Pré-entraînement masqué → axe Niveau.
- Le pseudo-labeling raté (§3) indique que les patients non étiquetés n'apportent presque rien via un professeur ;
  un pré-entraînement auto-supervisé sur les mêmes entrées a donc un gain attendu faible.
