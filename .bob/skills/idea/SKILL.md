---
name: idea
description: Add a formatted hypothesis to lab/IDEAS.md. Use when the user types /idea or proposes a new modelling idea to keep for later.
argument-hint: <description libre de l'idée>
---

Ajoute l'idée `$ARGUMENTS` au backlog `lab/IDEAS.md`.

1. Relis `lab/IDEAS.md` et `lab/LOG.md` : si l'idée existe déjà ou a déjà été testée, dis-le et arrête-toi.
2. Prends le prochain ID `HNN` libre.
3. Rédige la fiche au format « Modèle de fiche » du fichier (tous les champs, test en ≤ 10 lignes,
   référence = meilleur run actuel de `lab/results.csv`). Signale toute fuite potentielle.
4. Ajoute une ligne au tableau de priorisation (gain attendu, effort, priorité proposée).
5. Ne lance aucun run. Réponds avec l'ID et la priorité proposée.
