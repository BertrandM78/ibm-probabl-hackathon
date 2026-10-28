# EDA — Manquants + Pharmaco, suite (visites mal ancrées, réf. e023_seq_nn CV 2.610)

Scripts : `lab/scratch/miss2/` (`base.py`, `a1`–`a21`, sondes `p1_mask.py`, `p2_h39.py`, `p3_aug.py`).
Résidu r = target − pred OOF de `lab/runs/e023_seq_nn/oof.csv` (joint par Index) ; e022b en comparaison.
Ce qui utilise `target` relève de l'EDA. Les facteurs rater des sondes de correction sont recalculés par fold.

## Résumé

1. **Le poste nouveau : 20 % des patients traités n'ont aucun délai de prise (tsi_on ET tsi_off tous NaN).**
   Train 1 000 / 4 977 patients traités (18.8 % des visites), test 21.7 % / 19.0 %. Pour ces patients, A_p vaut
   A0 = 0.467 pour 100 % des visites. Profil identique aux autres (n 8.4 / 8.3, n_off 4.6 / 4.5, target 38.1 / 38.8,
   cohorte B 21 %, ledd NaN 35 / 33 %), bruit identique (IQR log off/(s·T) 0.389 / 0.375, ro 0.151 / 0.156,
   quantiles de on/(s·T) identiques à ±0.003 → tsi manquant **MCAR**). Pourtant e023 fait **3.13 contre 2.50**,
   OFF traités 3.13 / 2.51, ON seuls 3.65 / 2.91, zone après le dernier OFF 4.10 / 3.26. **e023 ne fait pas mieux
   que e022b sur ce groupe en zone après (4.10 contre 4.06)**, alors qu'il gagne −0.20 ailleurs. MSE patient
   7.68 contre 4.92 (visites traitées) : c'est une erreur de niveau. Si ce groupe était au niveau des autres, le
   CV passerait de 2.61 à environ 2.48.
2. **Sonde de masquage (e022b, fold 0, `p1_mask.py`)** : on met tsi_on et tsi_off à NaN côté valid pour la moitié
   des patients qui ont des tsi (modèle inchangé). Fold 2.745 → **2.938** ; patients masqués 2.918 → **3.412**
   (ON seuls 3.31 → 3.99, ON+OFF 2.88 → 3.26, OFF seuls 2.02 → 2.14, biais −0.11). Le délai de prise vaut
   donc environ +0.5 RMSE pour un patient traité. C'est l'information la plus précieuse de l'axe.
3. **H39 (paires sans tsi) n'apporte presque rien à e023.** Résidu patient de e023 sur les patients tsi-NaN :
   corr(r/niveau, A estimé sur les paires sans tsi) = **0.012** (e022b : 0.096). Une correction linéaire en
   cross-fit dégrade (2.6096 → 2.6115). Le réseau reconstruit déjà A à partir des on/off bruts. Les corr. avec un
   « A_vrai » (0.25–0.29) sont en partie **mécaniques** : A_vrai = 1 − on/(s·T) contient le target, donc
   r ↑ ⇒ A_vrai ↑. La « borne oracle −0.05 » de `missing.md` est surestimée.
   **Mais sur e022b, fold 0, H39 donne −0.040** (2.7448 → 2.7043 ; patients tsi-NaN 3.011 → 2.902) : voir §3.
4. **La courbe PK est commune** en durée, ledd et âge : la courbe normalisée au plateau varie de ±0.03 par tercile.
   Elle dépend en revanche **de A** : montée plus rapide quand A est haut (normalisée à 0.75–1 h : 0.53 au
   quintile bas, 0.70 au quintile haut ; à 1–1.25 h : 0.64 → 0.86 ; facteur de temps β = 0.79 par unité de A).
   Le biais de e023 par (tsi × tercile de A) reste plat (≤ 0.25), donc le réseau s'en accommode.
   Pour prédire une paire au plateau en LOO, l'estimateur de A gagne : seuil 0.5 0.0592, seuil 0.85 0.0536,
   poids g² 0.0547, noyau en tsi (τ 0.3) 0.0617 contre 0.0635 sur toutes les paires. Le gain est faible.
5. **Rien d'autre ne mord** : pente r ~ surprise de la visite ≈ 0 pour toutes les configurations (OFF 0.010,
   ON tsi 0.001, ON tsi? 0.006), même au niveau patient par zone (≤ 0.008 hors zone OFF). La correction de la
   zone OFF en cross-fit dégrade (2.610 → 2.618). Les poids de blend e023/e022b par zone × tsi-NaN : +0.001 ± 0.004.
   ledd NaN chez les traités est MCAR (9.9 % des visites, 0.6 % des patients tout NaN), 2.81 contre 2.92 : sans effet.
   Valeurs erronées : off == 0 1.24, on == 0 2.43, aucune incohérence structurelle (on > 1.2·off : 0 ;
   tsi sans mesure : 0 ; ledd sans on : 0 ; durée < 0 : 0). L'effet résiduel du médicament sur le OFF
   (tsi_off < 11 h, 988 visites, off/(s·T) 0.86–0.94, RMSE 3.31 / 2.69) pèse ≈ 2.5 % de la SSE, soit ≤ 0.008 RMSE.

## 1. Tableaux

| config visite (e023) | n | RMSE e023 | RMSE e022b | part SSE |
|---|---|---|---|---|
| OFF seul (non traité) | 13 220 | 1.71 | 1.85 | 12.7 % |
| OFF + ON, tsi présent | 9 489 | 2.50 | 2.69 | 19.6 % |
| OFF + ON, tsi NaN | 2 968 | 3.05 | 3.17 | 9.1 % |
| ON seul, tsi présent | 14 423 | 2.88 | 3.11 | 39.4 % |
| ON seul, tsi NaN | 4 490 | 3.60 | 3.68 | 19.2 % |

| patients traités | zone | n | e023 | e022b |
|---|---|---|---|---|
| tsi présent | OFF mesuré | 21 048 | 2.11 | 2.27 |
| tsi tout NaN | OFF mesuré | 4 629 | 2.63 | 2.75 |
| tsi présent | après le dernier OFF | 5 032 | 3.26 | 3.46 |
| tsi tout NaN | après le dernier OFF | 1 290 | **4.10** | **4.06** |
| tsi présent | avant le 1er OFF | 1 282 | 3.63 | 4.03 |
| tsi tout NaN | avant le 1er OFF | 288 | 4.73 | 4.93 |

Zone après : RMSE 3.10 / 3.43 / 3.81 / 3.85 / 4.43 pour k = 1…≥ 5 visites après le dernier OFF. Écart au dernier
OFF : 3.02 (< 1 an) → 4.39 (> 5 ans).

## 2. Information perdue sans tsi (EDA train)

- Visite ON : sd du log-écart on/(s·T(1−A·g)) 0.365 avec g connu contre 0.393 avec g = gbar. Une vraisemblance
  de mélange sur la loi de t (moyenne 0.395, mode 0.393) ne bat pas gbar.
- A_p sans tsi (paires présentes, tsi caché) : corr avec A estimé 0.73 → 0.66 (moyenne / gbar) → 0.70 (paires
  rognées au quart bas, g plateau 0.97), pour n ≥ 4 paires.
- Ces deux pertes réunies ne suffisent pas à expliquer +0.49 de RMSE. La sonde de masquage montre que l'essentiel
  vient de la façon dont le modèle **utilise** les visites ON sans délai (§3).

## 3. Sondes e022b fold 0 (base 2.7448)

| sonde (e022b, fold 0) | fold | tsi-NaN naturels (1 681 v.) | autres | masqués (3 098 v.) |
|---|---|---|---|---|
| base | 2.7448 | 3.011 | 2.678 | 2.918 (après masquage 3.412) |
| H39 : paires sans tsi, g = gbar, poids 1 (`p2_h39.py 0 1.0 gbar`) | **2.7043** | **2.902** | 2.656 | 3.388 |
| H39 rogné (e > 0.6·A0, g 0.97) | 2.7089 | 2.979 | 2.641 | 3.355 |
| augmentation : copies de 100 % des patients avec tsi, tsi masqués (`p3_aug.py 0 1.0`) | 2.7418 | 2.897 | 2.704 | |
| augmentation 30 % (`p3_aug.py 0 0.3`) | **2.7261** | 2.934 | 2.675 | |
| H39 + augmentation 100 % (`p4_h39_aug.py`) | 2.7349 | 2.898 | 2.695 | |
| moyenne des préd. H39 et augmentation 100 % | 2.6811 | | | |
| PK couplée à A : g(t·(1 + β(A_p − A0))), β = 0.83 fitté sur les paires train (`p5_pkA.py`) | 2.7370 | | | ON tsi < 1.5 h 2.976 → 2.958 |

Lecture :
- H39 donne −0.040 sur e022b, fold 0. C'est contraire à la sonde linéaire sur e023 (§ résumé 3). L'apport passe
  par les features (on_eq, MAP), pas par une correction a posteriori. Les patients qui ont des tsi en profitent
  aussi (2.678 → 2.656) : ce sont leurs 4.9 % de visites ON sans délai. Avant le 1er OFF, chez les tsi-NaN :
  5.00 → 3.71 (n 55).
- L'augmentation par masquage est fidèle, parce que tsi manque au hasard (MCAR) : chaque ε reste attaché à sa
  visite, contrairement aux re-bruitages de H36. Sur les tsi-NaN, elle gagne autant que H39 (−0.11) mais elle se
  confond avec lui : H39 + augmentation 100 % ne fait pas mieux que H39 seul sur ce groupe (2.898 / 2.902). À 100 %
  elle dégrade les autres (+0.026), à 30 % elle est neutre pour eux (−0.003). Les deux modèles se complètent en
  blend (moyenne 2.681, soit −0.023 de mieux que H39 seul).
- La PK couplée à A donne −0.008, ce qui est négligeable sans être nul.

## Pistes fermées (ne pas re-proposer)

Correction post-hoc par la surprise de la visite ou de la zone, poids de blend par configuration de manquants,
PK par ledd / durée / âge, dérive de A_p, imputation de ledd, détection de valeurs erronées, débiais du OFF par
tsi_off, vraisemblance de mélange en t pour les ON sans délai, H39 vu depuis e023 (corr 0.012).
