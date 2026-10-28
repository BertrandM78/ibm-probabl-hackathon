---
name: status
description: Summarise the lab state - top runs, open hypotheses, CV vs leaderboard correlation. Use when the user types /status or at the start of a session.
---

1. Lis `lab/results.csv` : top 5 par `rmse_oof` (avec fold_std et verdict).
2. Lis `lab/IDEAS.md` : hypothèses `TODO` par priorité, et `EN COURS`.
3. Lis le tableau « Leaderboard Kaggle » de `lab/LOG.md` : écart CV ↔ LB public ; alerte s'il diverge.
4. Réponds en ≤ 12 lignes et propose la prochaine action (gain attendu / effort).
