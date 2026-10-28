---
name: review
description: Run the leak-reviewer critic on an experiment, hypothesis or blend before a costly run or a Kaggle submission. Use when the user types /review.
argument-hint: <eNNN | HNN | chemin>
---

Lance un agent `leak-reviewer` sur `$ARGUMENTS` (fichier d'expérience, fiche d'IDEAS.md ou blend
`lab/runs/bNNN/metrics.json` avant `/submit`). Restitue ses findings `BLOQUANT` en premier, puis le verdict go / no-go.
Ne corrige rien sans accord de l'utilisateur.
