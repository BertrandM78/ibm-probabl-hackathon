---
name: brainstorm
description: Fan out analyst subagents in parallel, one per exploration axis, then merge their hypotheses into lab/IDEAS.md and prioritise. Use when the user types /brainstorm or asks for new ideas.
argument-hint: [axes séparés par des virgules] (défaut : tous)
---

Brainstorming structuré.

1. Axes = `$ARGUMENTS` ou, par défaut : Pharmaco, Trajectoire, Manquants, Rater, Structure, Méthode.
   Rappels par axe (à transmettre dans le prompt de l'agent) :
   - Pharmaco : retrouver la courbe PK (absorption rapide puis décroissance) depuis `time_since_intake_*`
     et le ratio on/off ; débiaiser `off` et `on`.
   - Trajectoire : pente par patient, effets aléatoires, lags/leads, interpolation, temps depuis le diagnostic.
   - Manquants : structure de missingness, imputation ON↔OFF via ratio, modèles NaN-natifs.
   - Rater : biais par clinicien (additif / multiplicatif), shrinkage, estimation hors-fold.
   - Structure : relations entre colonnes, cohortes, gènes, contraintes du générateur.
   - Méthode : GBM, modèles mixtes, petits réseaux, stacking, cibles transformées (résidu vs off).
2. Lance un agent `analyst` par axe, **en parallèle dans un seul message**, avec : l'axe, le meilleur run
   actuel, et l'instruction de renvoyer 2–4 fiches au format imposé.
3. Fusionne les fiches dans `lab/IDEAS.md` (renumérote les IDs, supprime les doublons), complète le
   tableau de priorisation (priorité = gain attendu / effort).
4. Lance un agent `leak-reviewer` sur les 3 fiches les mieux classées.
5. Présente le top 5 à l'utilisateur et demande lesquelles lancer.
