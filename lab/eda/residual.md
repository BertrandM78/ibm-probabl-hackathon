# EDA — Diagnostic de l'erreur résiduelle + trajectoire (réf. e022b, CV 2.778)

Scripts : `lab/scratch/resid/` (`r1`–`r20`, sondes `p1`–`p4`). Frames e022b des 5 folds (train + valid, préd.
niveau/delta) : `build.py` → `feat_f{0..4}.pkl`, rejoue e022b au 1e-4 (f0 2.7448, f1 2.6942, f2 2.7277,
f3 2.9050, f4 2.8139). Les analyses qui utilisent `target` se font sur le train ou sur l'OOF.

## Résumé

1. **La part « niveau patient » fait 63 % de la MSE et le résidu n'est plus prévisible.** MSE 7.72 =
   niveau 4.88 · pente 1.85 · courbure 0.87 · cubique 0.08 · reste 0.05. Avec une forme parfaite, on resterait
   à 2.21 (√4.88). Un HGB sur le résidu OOF (toutes les features e022b, folds du lab) **dégrade** le score
   (2.863 ; 2.805 avec un facteur 0.5). Un second étage qui ajoute les surprises de y_v autour de la **préd.
   finale** (max, min, médiane, sd, skew, voisines) fait 2.793 contre 2.790 pour le contrôle sans elles. La
   projection des préd. sur une cubique patient donne −0.004 (w = 1), sur une quadratique −0.007 (w = 0.5),
   ce qui est négligeable.
2. **Pistes de forme de trajectoire réfutées** (sondes sur la trajectoire seule, fold 0, MAP 3.562) :
   moyenne postérieure SN par échantillonnage d'importance 3.591 (ESS médian ≈ 520/600) ; moyenne
   postérieure avec une densité de bruit **non paramétrique** (KDE par source) 3.607–3.609, mode KDE 3.669 ;
   moyenne sous contrainte de monotonie 3.582 (masse monotone médiane 0.96). La monotonie des préd. finales est
   déjà quasi respectée (68 baisses sur 39 014, isotonique ±0.000). Échelle hétéroscédastique : mu1 = préd. finale
   3.566, **mu1 = target oracle 3.594**. sig2 × 0.25 3.562 ; Σ × 0.5 3.747, Σ × 2 3.689 (Σ est donc bien calé).
   **Le coude n'est pas un poste d'erreur** : à durée égale, les patients « cubique + coude » ne sont pas plus
   mal prédits que les cubiques exacts (dur 7–10 ans : 2.66 contre 2.67 ; 10–13 ans : 3.52 contre 3.71). Le
   reste hors cubique ne pèse que 0.05 de MSE. Un Kalman ou un GP ne peut donc rien apporter sur la forme.
3. **Le modèle est limité par la quantité de données** (courbe d'apprentissage de l'étage HGB, features
   figées) :

| patients train | fold 0 | fold 3 |
|---|---|---|
| 25 % (1 115) | 2.996 | 3.113 |
| 50 % (2 230) | 2.827 | 3.008 |
| 100 % (4 460) | 2.745 | 2.905 |

   On gagne −0.08 à −0.10 à chaque doublement, sans saturation visible. Conséquence directe pour CV ↔ LB : le
   modèle final entraîné sur 100 % du train (contre 80 % en CV) devrait être meilleur d'environ 0.03 à 0.04 sur
   le test que le CV ne l'annonce.
4. **Le bruit réel n'est pas i.i.d., et HGB s'en sert.** Sonde `p4.py` (fold 0) : on duplique les patients
   train en re-bruitant `off`/`on` (même target, k = ε_new/ε_old appliqué à on et off à chaque visite), on
   calcule les features côté **valid** (modèles fittés sur le train réel seulement), puis on entraîne HGB sur
   ces copies seules et on score la vraie valid :

| générateur (copie × 1) | synth. seul | réel + synth. |
|---|---|---|
| identité (contrôle d'asymétrie train/valid) | 2.768 | 2.743 |
| gigue log N(0, 0.05) | 2.808 | 2.742 |
| ε permutés **intra-patient** (même source) | 3.077 | 2.844 |
| ε re-tirés, OFF seulement (strates traité × source × target/10) | 3.153 | 2.831 |
| ε permutés entre patients, même rater × target/5 × source | 3.375 | 2.916 |
| ε re-tirés on + off (strates source × traité × target/10) | 3.413 | 2.928 |

   (référence réelle : 2.745). Dès que l'on casse le lien entre un ε et **sa** visite, le meilleur modèle
   possible retombe au niveau de la trajectoire MAP (≈ 3.4–3.56), alors que la distribution marginale du
   bruit est conservée. Une gigue de 5 % coûte à elle seule +0.04. HGB se sert donc d'une structure
   visite/patient du bruit que le modèle SN i.i.d. de la MAP n'a pas. C'est ce qui explique l'écart MAP 3.59
   → final 2.78 (niveau 7.48 → 4.88, pente 3.61 → 1.85). Ce qu'on a pu identifier jusqu'ici :
   - ε dépend de la **pente locale** du target : médiane log ε −0.12 dans l'octile bas de slope_prev, +0.08
     dans l'octile haut. Ablation HGB sur ε (R² total 0.070) : slope_prev −0.0077, slope_next −0.0044,
     position relative −0.003. La trajectoire MAP en hérite un biais de niveau de **+1.47** dans l'octile bas de
     pente (−0.32 dans l'octile haut), que HGB ramène à +0.23.
   - l'échelle |ε| persiste un peu au niveau patient (corr avec la moyenne LOO 0.121 contre 0.069 ± 0.009 sous
     permutation), et le max patient est plus regroupé qu'en i.i.d. (sd 0.106 contre 0.114 ± 0.002).
   - pas de décalage temporel (target(âge + δ) : sd minimale à δ = 0), pas de persistance de la moyenne
     (corr LOO 0.00, lag-1 −0.03) ; corr. du bruit on/off à la même visite **0.83**.
   Ces éléments ne suffisent pas à expliquer tout l'écart (3.08 en intra-patient).

## Où sont les 0.58 (MSE 7.72 → 4.84 visée)

| sous-groupe | part visites | RMSE | part SSE |
|---|---|---|---|
| patients toujours traités | 22 % | 3.64 | 38 % |
| patients qui commencent le traitement en cours de suivi | 70 % | 2.57 | 60 % |
| patients jamais traités | 7 % | 1.29 | 2 % |
| dernière visite / 1re / milieu | 12.5 / 12.5 / 75 % | 3.46 / 2.88 / 2.63 | 19 / 13 / 67 % |
| source off / on_t / on_nt | 58 / 32 / 10 % | 2.37 / 3.11 / 3.68 | 42 / 41 / 18 % |
| n_off patient 0–2 | 12 % | 3.6–5.1 | 25 % |
| target 50–65 / 65–80 / > 80 | 19 / 5 / 0.2 % | 3.43 / 4.25 / 9.10 (biais +0.49 / +1.92 / +6.86) | 29 / 11 / 2.5 % |
| durée 11–14 / > 14 ans | 7 / 3 % | 3.53 / 4.13 | 12 / 6 % |
| cohorte B | 11 % | 2.98 | 13 % |

- L'erreur est concentrée : top 5 % des patients = 38 % de la SSE, top 10 % = 52 %. Le top 5 % est à 70 %
  sous-estimé (erreur moyenne de niveau +2.83). Ce sont des patients à target élevé (49 contre 35), qui ont moins
  de OFF (40 % des visites contre 61 %), plus de visites on_nt (2.1 contre 0.7) et qui sont plus souvent
  traités dès le départ (48 % contre 21 %).
- Erreur de niveau patient asymétrique (skew +1.10 ; q01 −5.1, q99 +7.5). Le niveau reste bien calibré en
  moyenne (pente target ~ préd. 1.00). Pente patient : sd préd. 1.09 contre 1.35 vrai, pente de calibration 1.045.
- Autour de l'initiation du traitement, l'erreur est plus **faible** (2.25 à k = −1/0) : l'initiation n'est pas
  un poste d'erreur.
- Aucun sous-groupe isolé ne contient 0.58. L'écart est diffus et tient au niveau patient, quel que soit le
  sous-groupe.

## Pistes fermées (ne pas re-proposer)

Moyenne postérieure au lieu du MAP (SN ou KDE), monotonie imposée (MAP ou isotonique), mélange coude / pas
coude, Kalman/GP (rien hors cubique : 0.05 MSE), échelle hétéroscédastique itérée sur la préd. finale, second
étage sur les surprises autour de la préd. finale, projection des préd. sur la cubique patient, augmentation de
données par re-bruitage naïf (+0.09 à +0.18 sur le fold 0) ou par gigue (−0.003), mélange trajectoire + préd.
(pire).
