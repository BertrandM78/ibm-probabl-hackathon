# EDA — axe DGP, 3e passe (vraisemblance des mesures)

Scripts : `lab/scratch/dgp3/` (`base.py` charge le train + l'OOF `b002_seq_e022b`, s = médiane off/target par
rater sur tout le train, EDA seulement). Les sondes `e4`, `e5`, `e8` et `e11` rejouent la MAP cubique e022b
sur le fold 0 (`scratch/meas/tr0.pkl`, `va0.pkl`, `prior0.pkl`). Base : valid hb_post_full **3.5688**,
LOO 4.4318. Les grilles sont choisies sur la valid du fold 0 : ce sont des sondes, pas des scores.

## Résumé

| Piste | Constat | Sonde fold 0 (trajectoire valid) |
|---|---|---|
| **Visite 0 = régime de bruit distinct** | IQR de log(off/(s·T)) centré : **0.33 à v0** contre 0.47–0.51 à v1–v3 (patients cubiques exacts : 0.36 contre 0.49–0.51), médiane +0.05. Aussi sur ON | template séparé `*_v0` : **3.5688 → 3.5387 (−0.030)**, RMSE à v0 4.04 → 3.69. Avec off_tr (H38) : **3.5370** |
| Dispersion OFF non traité qui croît avec le rang | IQR 0.50 (1 OFF avant) → 0.57 → 0.61 → 0.63 (≥ 4), médiane −0.03 → −0.07 ; chez les traités, plat (0.36–0.40) | non sondé (prolongement de la ligne du dessus) |
| Effet « pente locale » (H37) | **artefact de la règle ×1.02** : sur les patients cubiques exacts, l'effet par décile de ΔT va de −0.035 à +0.012, contre −0.13 à +0.10 sur l'ensemble des patients. Dépend de ΔT_prev (régression quantile, t = 2.9), pas de ΔT_next (t = 0.6) | — |
| Loi d'échelle du bruit | bord droit ∝ √T (q75−q50)/√T ≈ 0.8–1.2 constant ; puissance T^0.66 meilleure en vraisemblance (−85 nll sur des patients held-out, −120 avec la scission traités) ; erreur de niveau oracle 0.1047 → 0.0992 | **pas mieux en MAP** : γ 0.6 → 3.5786, 0.7 → 3.601, 1.0 → 4.10 (affine 3.5688). Réfuté |
| Σ du prior selon n visites | patients n = 4 : courbure par an² ≈ 3.5× celle des fenêtres de 4 visites des patients longs (IQR q2 12–30 contre 2.4–6.7 à fenêtre égale) | Σ par groupe n : 3.6087 (pire) ; n = 4 : 4.255 → 4.251. Réfuté |
| Initiation du traitement | hazard ~ T (+0.11/pt), ΔT (coef. T_prev −0.088), durée ; **non lié au bruit** (ε_prev z = −1.0, off courant : coef. < 0) ; jamais à T < 15.8 ; ≤ 4 % si dur ≤ 1 an quel que soit T ; 100 % si T ≥ 35 et dur ≥ 4 | terme logistique dans la MAP : full 3.5673 (−0.0015), **LOO 4.432 → 4.402–4.406** |
| Target ajusté sur les mesures ? | non : R² de la tendance linéaire des résidus additifs 0.1385 contre 0.1452 sous permutation (1.2 se) | — |
| Valeurs recopiées (LOCF) / doublons | P(off_i = off_{i−1}) 0.043 contre P(\|diff\| = 1)/2 0.043 : aucun excès ; 1 seule paire (gaps, targets) identique (motif 8, 9, 10, 11) | — |
| Rater × patient | corr. du bruit pour le même rater dans le même patient −0.014 contre −0.014 ± 0.008 sous permutation | — |

## 1. La visite 0

- Écart à la médiane (traité × décile de T), par rang de visite (tous patients) :

| rang | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7+ |
|---|---|---|---|---|---|---|---|---|
| médiane | +0.052 | −0.003 | −0.020 | −0.015 | −0.026 | −0.016 | 0.000 | −0.020 |
| IQR | **0.325** | 0.473 | 0.496 | 0.486 | 0.458 | 0.432 | 0.423 | 0.364 |
| q90 − q50 | 0.179 | 0.268 | 0.288 | 0.266 | 0.287 | 0.290 | 0.269 | 0.254 |

- L'écart vaut quelle que soit la longueur du suivi (IQR à v0 : 0.28–0.37 pour n = 4…8). Il est lié au **rang**
  et non au temps écoulé : aux visites 1–3, l'IQR reste à 0.48–0.51 quel que soit le délai depuis la 1re visite.
- Template SN refitté (fold 0) : off ω 1.49 → **1.19** à v0, α −3.7 → −5.5 ; on_t ξ 0.75 → 1.08, α −1.4 → −2.9.
- Le ratio on/off (sans target) n'est pas plus resserré à v0 (IQR 0.204 contre 0.194). C'est donc le choc
  commun de la visite qui est plus petit à v0.

## 2. Initiation du traitement : information sur P

Taux d'initiation empirique chez les patients à risque (hors 1re visite) :

| T \ durée | ≤ 1 an | 1–2 | 2–4 | 4–6 | 6–10 |
|---|---|---|---|---|---|
| 20–25 | 0.05 | 0.16 | 0.34 | 0.41 | 0.46 |
| 30–35 | 0.11 | 0.29 | 0.73 | 0.86 | 0.82 |
| 40–50 | 0.07 | 0.32 | 0.78 | 0.97 | 1.00 |

L'événement « traité » dépend du target latent seulement ; la MAP s'en sert peu, car le prior HGB voit déjà
`ledd_na` et la durée.

## 3. Patients courts, T0 entier, incrément nul à v1

- T0 est entier chez 17.9 % des patients (≈ 10 % attendus), et chez 25 % des patients à n = 4. On trouve
  68 patients dont **tous** les targets sont entiers (65 à n = 4), avec des incréments entiers, pour 0.16
  attendu.
- Incrément nul à v1 (Z1) : 20 % si T0 est entier, contre 0.6 % sinon (174/202 Z1 ont T0 entier). Chez les
  Z1, les mesures valent le target à v0 (+0.12) puis passent **−0.34 / −0.23 / −0.22** sous lui à v1–v3 (P
  chute, la règle maintient T).
- La règle ×1.02 se déclenche tôt chez les patients courts (v1 : 18 % à n = 4 contre 2.7 % à n = 12).
  Erreurs b002 : n = 4 2.92, n = 4 et T0 entier **3.12 (biais +0.46, +0.85 à v0)**, patients tout entiers 3.78
  (biais +1.47). Le poids est faible : T0 entier = 16 % de la SSE, Z1 2.8 %.
- Le mécanisme exact de T0 entier n'est pas identifié (off0 = T0 dans seulement 7 % des cas).

## 4. Calibration des modèles finaux (OOF)

Pente de calibration de la pente patient par groupe n : b002 0.96–1.04, e023 0.91–0.98 ; niveau 0.99–1.00.
Il n'y a pas de sur-rétrécissement à corriger : ce qui reste est de l'incertitude, pas un biais.

## Pistes fermées

Loi d'échelle en puissance, Σ par nombre de visites, H37 sous sa forme actuelle (remplacée par la règle),
target ajusté sur les mesures, LOCF/doublons, interaction rater × patient, sélection du traitement par le bruit.
