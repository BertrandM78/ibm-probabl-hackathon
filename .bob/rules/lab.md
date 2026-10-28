# Projet true OFF — contexte permanent

Après la setup gate (`AGENTS.md`), lis `Bob.md` à la racine : problème, protocole de CV verrouillé,
pièges, boucle de travail et procédure de soumission. Il fait foi pour tout le projet, et pour
chaque agent de `.bob/agents/`.

## Garde-fous (`.bob/hooks/`)

- Avant de créer ou modifier un fichier, vérifie-le avec
  `echo '{"tool_input": {"file_path": "<chemin>"}, "cwd": "'"$PWD"'"}' | python3 .bob/hooks/guard_protected.py` :
  code de sortie 2 = fichier protégé (`data/`, `lab/folds.csv`), n'y touche pas sans accord explicite
  de l'utilisateur. Même règle pour `lab/cv.py` (constantes, métrique).
- Après toute commande qui écrit des prédictions (`cv.py --test`, `blend.py`, `submit.py`), lance
  `python3 .bob/hooks/check_new_predictions.py` : code 2 = un `lab/runs/*/test.csv` récent est invalide,
  corrige-le avant de le déclarer prêt.
- Une soumission Kaggle passe **uniquement** par `lab/submit.py` (rapport skore + CSV validé) ; jamais
  de CSV écrit à la main ni de soumission sans URL Skore Hub dans la description.
