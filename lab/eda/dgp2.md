# EDA — axe DGP, suite (rétro-ingénierie du target, plancher d'erreur)

Scripts : `lab/scratch/dgp2/*.py` (train seul ; résidus = `lab/runs/e022b_cubic_traj_quad/oof.csv`).
Tout ce qui touche `target` est de l'EDA : refitter in-fold toute constante (sauf 1.02, constante du générateur).

## 1. Le « coude » = règle de monotonie multiplicative 1.02 par visite (pas un coude en âge)

- `q1.py` : `off`, `on`, `ledd` entiers (100 %, train et test) ; `target`, âges, tsi au 0.1. Pas de pas 0.5.
- `g1.py` : histogramme de `target_i / target_{i−1} − 1` : pic à **0.020** (bins 0.019/0.021 : 2 417 + 2 652
  visites contre ~700 de fond), ≈ 3 500 visites en excès (~8 %).
- `v4.py`, `r1.py`, `r2.py` : règle du générateur (visites triées par âge, P = cubique latente du patient) :
  `T_0 = P(t_0)` ; `T_i = P(t_i)` si `P(t_i) ≥ T_{i−1}`, sinon `T_i = 1.02 · T_{i−1}` (indépendant de Δâge),
  puis plancher 0 et arrondi 0.1.
  - Visites post-pic (P < T_{i−1}, n = 723) : ratio médian 1.0200, q5–q95 1.0192–1.0208 ;
    `|T − round(c·T_{i−1}, 1)| < 0.05` : 100 % pour c = 1.02, 7 % pour 1.018/1.022.
  - La règle reproduit 99.5 % des visites à 0.08 près chez les patients dont la cubique est identifiable.
- Cubique exacte (rm < 0.045, patients ≥ 8 visites) : 67 % brut → **85 %** en retirant les visites
  G (×1.02) / Z (0 incrément) → **≈ 95 %** avec un fit non linéaire `rule(P(θ))` (`r3.py` : 75 % des 363 restants).
  Résidu inexpliqué ≈ 3–4 % des patients, surtout les visites Z (350 incréments nuls, 58 % à la visite 1,
  suivies d'un G dans 64 % des cas : 1er déclenchement parfois « plat », règle exacte non trouvée).
- L'ancien « coude à 11 ans, pente avant ≈ 0, saut +2 » = artefact de cette règle : déclenchée quand la
  cubique passe par son maximum (fin de suivi, durée longue, patients toujours traités). Aucun lien avec
  le début du traitement (`k4.py` : coude 6.6 ans après la 1re visite traitée en médiane), LEDD, cohorte.
- Les incohérences sont **en fin de trajectoire** (`v2.py` : patients non exacts n ≥ 10, retirer les 2
  dernières visites → 62 % exacts ; retirer les 2 premières → 0.3 %).

## 2. Les mesures suivent la cubique latente P, pas le target

`n1.py` (visites post-pic avec off, facteur rater = médiane off/target) :

| écart target/P | n | méd. off/(s·target) | méd. off/(s·P) |
|---|---|---|---|
| 1.03–1.06 | 90 | 0.913 | 0.960 |
| 1.06–1.10 | 92 | 0.911 | 0.987 |
| 1.10–1.20 | 72 | 0.877 | 0.993 |
| référence (visites non post-pic) | | 1.015 | |

off/(s·P) est plat, off/(s·target) baisse avec l'écart. Même chose côté ON (0.517 vs P 0.557, référence 0.581).
→ **le générateur produit les mesures à partir de P (qui peut décroître), puis le target = rule(P)**.
Conséquence : aux visites post-pic, les mesures sous-estiment le target de 3 à 20 % (médiane 7 %) ; le template
de bruit (`y_v − target`) et le prior cubique (OLS sur target) des expériences sont légèrement contaminés.

## 3. Degrés de liberté / hypothèse leaspy (logistique, time-shift + vitesse)

- `p1.py`, `p2.py`, `p3.py` : en base durée brute, c2/c3 « prédictibles » depuis c0/c1 (R² 0.81/0.86) mais
  c'est un artefact de base : en base centrée patient (niveau, pente, courbure au milieu de la fenêtre), la
  courbure n'est pas prédictible depuis niveau/pente/durée/âge au diagnostic (R² linéaire 0.11, HGB < 0) →
  **4 ddl réels**, prior gaussien 4D déjà adapté.
- Logistique leaspy (forme fixe, time-shift τ, vitesse α) **rejetée** : (i) polynôme exact à l'arrondi 0.1
  sur 8–12 visites pour 85–95 % des patients ; (ii) signe de c3 réparti 48 % / 52 % ; (iii) c3/β³ (β = pente à
  l'inflexion) non constant (q10–q90 −3e−4 à 2.9e−2) ; (iv) inflexion en durée très dispersée (q5–q95 −10 à 19 ans).
  Une forme sigmoïde commune ne s'applique pas ; la cubique + règle 1.02 suffit.

## 4. Où sont les erreurs de e022b (CV 2.778)

`o1.py` (visites classées sur le target, à des fins de diagnostic seulement) :

| groupe | n | RMSE | biais | part du SSE |
|---|---|---|---|---|
| visites ×1.02 (G) | 3 649 | 3.21 | −0.27 | 10.9 % |
| visites G post-pic certaines | 1 084 | 3.42 | −0.23 | 3.7 % |
| incrément nul (Z) | 393 | 3.21 | +0.70 | 1.2 % |
| suite G/Z terminale | 2 249 | 3.53 | −0.17 | 8.1 % |
| autres visites des patients à queue G/Z | 8 486 | 3.04 | **−0.50** | 22.8 % |
| patients sans G/Z | 26 570 | 2.62 | +0.14 | 52.8 % |

RMSE si suites G/Z parfaites 2.66 ; si patients à queue parfaits 2.31.

## 5. Plancher / oracle

- `b1.py`, `b3.py` (folds 0/1, colonnes e022b côté valid) : écart-type postérieur Laplace de la trajectoire
  cubique `sqrt(mean hb_se_full²)` 3.45 / 3.50 contre une erreur réelle de `hb_post_full` de 3.55 / 3.54 →
  **l'estimation de la trajectoire à partir de y_v est déjà à son plancher bayésien** (≈ bruit/√n). Par
  nombre de y_v : 3.77 (≤ 4) → 3.15–3.29 (6–10). Le HGB (2.74 / 2.69) bat ce plancher en utilisant une
  information hors y_v (statut traité déclenché par le target, surprises, etc.).
- Forme du target : déterministe (rule(cubique)) à ~95 % → l'erreur de forme résiduelle est faible ;
  les 0.58 qui manquent ne viennent pas d'une forme mal spécifiée mais (a) des patients à queue post-pic
  (31 % du SSE, biais −0.5 sur toute leur trajectoire) et (b) d'informations de niveau hors trajectoire.
- Règle appliquée à la trajectoire seule : `rule(hb_post_full)` 3.552 → 3.524 (fold 0), 3.544 → 3.491 (fold 1).
  Mais P̂ ne déclenche la règle que sur 1 % des visites (vs ~5–8 % vraies) : le prior (fitté sur le target
  monotonisé) rend P̂ trop monotone.
- Post-traitement `pred + a·(rule(P̂) − P̂)` sur e022b : fold 0 −0.0005 (a = 0.25) puis pire ; fold 1
  −0.009 (a = 0.75) → incohérent, négligeable.
- Sonde `b4.py` (pipeline e022b entraîné sur P latent de train = cubique là où la règle est déclenchée,
  `lat.py`, 2.8 % des visites train ; puis rule(préd.)) : fold 0 2.768 (+0.023 vs 2.745), fold 1 2.681
  (−0.013 vs 2.694). Seul : non. **Moyenne 50/50 avec e022b : 2.733 (−0.012) / 2.670 (−0.024)** → diversité.
