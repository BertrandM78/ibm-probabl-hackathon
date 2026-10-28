# H63 — Progression ON hors de la fenêtre OFF (2026-10-13)

**Verdict : ABANDONNER.** Les séries ON ne contiennent pas d'information sur la progression que
b003_seq4 (ni même e022b) laisserait inexploitée, hors fenêtre OFF ou ailleurs. Sonde sur OOF
uniquement : aucun réseau réentraîné.

Scripts : `lab/scratch/h63/diag.py` (phase 1, exploratoire) et `lab/scratch/h63/probe.py` (sonde
prédictive, règles déclarées avant le run). Logs : `diag.log`, `diag_level.log`, `probe.log`,
`probe_e022b.log`. Revue leak-reviewer : GO sous conditions pour la sonde OOF. Les 6 conditions sont
appliquées : règle `off > 0`, `check_order`, liste `FEATURES` avec assert anti-target, contrôle nul,
seuils déclarés, pas d'adoption côté test.

## 1. Features ON déjà exploitées

| Pipeline | Information ON utilisée |
|---|---|
| e022b (HGB) | `on`, `on_n` (rater), `on_eq` / `on_eq_n` (effet médicament A_p·g(tsi)) + prev/next/pmean ; bloc temporel `t_on_*` (lags, leads, rolling, `t_on_slope`, `t_on_trend`, `t_on_trend_res`) ; **y_v = on_eq_n quand OFF manque**, intégré à la MAP cubique skew-normal (sources `on_t` / `on_nt`, bruit propre) → `hb_*`, `hc_*`, `q_*`, surprises `L_*` |
| e023_seq_nn, e025_seq_offset | toute la matrice e022b dense, séquence complète du patient |
| e026_seq_amortized | `y_v`, `w_v`, `on_n`, `off_n`, `s_rater`, `g_on`, `on_eq_n`, `A_p` |
| e025_gru_raw | `on`, `tsi_on`, masques, rater en embedding, Δt (correction apprise par le réseau) |

La progression ON entre donc déjà dans la trajectoire latente, avec un modèle de bruit dédié.

## 2. Diagnostic (phase 1, exploratoire)

Zones, définies par X seul : A = OFF observé 57.6 %, B = entre deux OFF 24.1 %, C = hors fenêtre
18.3 % (après le dernier OFF 14.2 %, avant le premier 3.5 %, aucun OFF 0.65 %). Répartition identique
en test (vérifié par leak-reviewer).

**Pentes par patient (≥ 3 ON, n = 4 206)** — corrélation avec la pente vraie :

| Indicateur | corr. | moyenne | sd |
|---|---|---|---|
| pente ON brute | 0.17 | +1.21 | 2.38 |
| pente ON corrigée rater (`on_n`) | 0.18 | +1.40 | 2.69 |
| pente ON corrigée médicament (`on_eq_n`) | 0.20 | +2.27 | 4.03 |
| pente ON robuste (Theil-Sen) | 0.19 | +2.26 | 4.17 |
| pente y_v (OFF + ON harmonisés) | 0.52 | +2.17 | 2.42 |
| **pente b003_seq4** | **0.84** | +2.48 | 1.05 |
| pente e022b | 0.83 | +2.51 | 0.99 |
| pente vraie | — | +2.51 | 1.25 |

La correction médicament recentre bien la pente ON (+2.27, contre +2.51 pour la vraie), mais les
pentes ON sont 3 à 4 fois plus bruitées que la pente vraie.

**Information résiduelle**. Corrélation entre (pente vraie − pente modèle) et (pente ON − pente modèle) :

| Indicateur | ≤ 2 OFF | 3–4 OFF | ≥ 5 OFF |
|---|---|---|---|
| ON corrigée médicament | −0.013 | +0.017 | −0.016 |
| ON robuste | −0.012 | +0.017 | −0.016 |
| ON locale (après le dernier OFF, ≥ 3 ON) | −0.011 | +0.038 | −0.013 |

- **Niveau** : corrélation entre le résidu moyen patient et (niveau ON − niveau modèle),
  (y_v − modèle), (off_n − modèle) ou A_p. Elle vaut −0.02 à +0.02 au global, ±0.15 maximum par fold,
  et le signe varie d'un fold à l'autre.
- **Visites après le dernier OFF** : corrélation entre le résidu et (pente ON locale − pente modèle
  locale)·Δt. Par fold : +0.03 / −0.11 / +0.04 / +0.05 / −0.09.
- Aucune association stable, ni par fold ni par nombre de OFF.

## 3. Information résiduelle : test prédictif croisé

HGB fixé à l'avance, entraîné sur les résidus OOF de zone C des 4 autres folds et appliqué au fold
évalué. 20 features sans target : zone, Δt au dernier / premier OFF, n_off, n_on, n_tail_on,
divergences pente / niveau ON (globales et locales), y_v, A_p, n_av, préd., rang, Δt voisins.

| Base | R² zone C par fold | moyenne | contrôle nul | seuil (≥ 0.02 sur 4/5) |
|---|---|---|---|---|
| b003_seq4 | −0.110 / −0.173 / −0.089 / −0.061 / −0.110 | −0.109 | −0.081 | **ÉCHEC** |
| e022b | −0.077 / −0.098 / −0.092 / −0.021 / −0.066 | −0.071 | −0.080 | **ÉCHEC** |

R² négatif sur tous les folds, et pas mieux que le contrôle par permutation : rien de prédictible.

## 4. Expérience minimale (descriptive, le seuil ayant échoué)

Correction appliquée en zone C seulement ; zones A et B inchangées.

| Variante (base b003_seq4 = 2.5408) | RMSE | Δ ± std | folds meilleurs | zone C | zone B | ≤ 2 OFF |
|---|---|---|---|---|---|---|
| b003_seq4 | 2.5408 | — | — | 3.445 | 2.505 | 3.742 |
| HGB α = 0.25 | 2.5468 | +0.006 ± 0.004 | 0/5 | 3.467 | 2.505 | 3.756 |
| **HGB α = 0.5 (principale)** | 2.5562 | **+0.016 ± 0.007** | **0/5** | 3.503 | 2.505 | 3.778 |
| HGB α = 1.0 | 2.5850 | +0.045 ± 0.015 | 0/5 | 3.610 | 2.505 | 3.846 |
| linéaire α = 0.25 / 0.5 / 1 | 2.5410 / 2.5411 / 2.5415 | +0.000 à +0.001 | 0/5 | 3.445 | 2.505 | 3.743 |

- **Variante principale, par fold** : +0.016 / +0.028 / +0.013 / +0.010 / +0.011.
- **Erreur de niveau patient (RMS)** : 2.154 → 2.166.
- **β linéaire par fold** : −0.003 / +0.006 / −0.005 / −0.007 / +0.002, soit environ 0 avec un signe
  instable.
- **Avec e022b comme base** : α = 0.5 donne +0.008 ± 0.006 (1/5). Le β linéaire est négatif sur les
  5 folds mais minuscule : Δ ≤ 0.0001.

Toute correction dégrade la zone visée. Elle ajoute la variance du modèle de résidus sans apporter
d'information.

## 5. Interprétation

- ON = OFF·(1 − A_p·g(tsi))·rater + bruit. La progression ne s'y lit qu'à travers A_p et g, qui sont
  inconnus et propres au patient. Une fois rater et médicament corrigés, la pente ON est non biaisée
  mais très bruitée (sd 4.0 contre 1.25) : un patient de 3–5 ON n'en tire presque rien de plus que ce
  que la MAP a déjà pondéré via y_v.
- L'erreur de niveau des patients pauvres en OFF (corrélée à 0.9 entre modèles, cf. `ensemble3.md`) est
  donc bien une **limite d'information** des données observées, pas une information ON mal exploitée.
- Conséquence pour H40 / H42 (incertitude A_p) : elles visent la même source d'erreur. H42a a déjà
  donné NOISE (e024_ap_pairs_notsi). Ce résultat rend un gain important de leur côté peu probable.

## 6. Verdict

**ABANDONNER H63.** Le seuil déclaré échoue sur les deux bases, toutes les corrections dégradent le
score, sans aucun signal exploratoire stable. Pas de réentraînement de réseau justifié.
Pistes plus prometteuses : celles qui n'ajoutent pas d'information ON, par exemple H64 (poids
d'ensemble par zone, `ensemble3.md`) ou les hypothèses de structure du générateur (H33/H46).
