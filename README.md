# Snake RL

Projet d'apprentissage par renforcement (Q-learning) applique au jeu Snake.
Dashboard interactif Streamlit pour comparer differentes configurations d'entrainement.

## Prerequis

- Python 3.10+
- `pip`

## Installation

```bash
pip install -e .
```

Pour le developpement (linter, tests, pre-commit) :

```bash
make install-dev
```

## Structure

```
src/snake_rl/
  env.py             Environnement Gymnasium (Snake)
  agent.py           Agent Q-learning + save/load
  training.py        Boucle d'entrainement
  evaluation.py      Metriques d'evaluation
  visualization.py   Visualisation HTML d'une partie
  grid_search.py     Grid search d'hyperparametres
  cli_train.py       CLI d'entrainement
  cli_eval.py        CLI d'evaluation
  cli_visualize.py   CLI d'export HTML

app/
  streamlit_app.py   Dashboard - page d'accueil
  pages/             Pages Streamlit (comparaison, visualisation, entrainement)
  utils.py           Utilitaires partages
  style.css          Theme personnalise

tests/               Tests unitaires (pytest)
```

## Utilisation

### Entrainer un modele

```bash
snake-train --episodes 10000 --model-path artifacts/best_model.pkl
```

### Evaluer un modele

```bash
snake-eval --model-path artifacts/best_model.pkl --episodes 200
```

### Generer une visualisation HTML

```bash
snake-visualize --model-path artifacts/best_model.pkl --output artifacts/episode.html --greedy
```

### Lancer le grid search

Entraine plusieurs modeles avec differents hyperparametres :

```bash
make grid-search
```

### Lancer le dashboard

```bash
make app
```

Ouvre ensuite http://localhost:8501 dans un navigateur.

## Docker

```bash
make docker-build
make docker-run
```

Le dashboard est accessible sur http://localhost:8501.

## Qualite du code

```bash
make lint      # Verification (ruff check + format --check)
make format    # Correction automatique
make test      # Tests unitaires
```

## CI/CD

Le projet utilise GitHub Actions (`.github/workflows/ci.yml`) :
- **Lint** : verification du code avec ruff
- **Test** : execution des tests avec pytest
- **Docker** : build et push de l'image sur `ghcr.io/maeltremouille/snake_rl`

Le pipeline se declenche automatiquement a chaque push sur `main` ou `mael-features`.

## Deploiement (SSP Cloud)

L'application est deployee sur le SSP Cloud via ArgoCD (GitOps) :

1. Le CI push l'image Docker sur ghcr.io
2. ArgoCD surveille le dossier `k8s/` du repo
3. Kubernetes deploie l'application automatiquement

Les manifestes Kubernetes se trouvent dans `k8s/` :
- `deployment.yaml` : configuration du container
- `service.yaml` : exposition du port 8501
- `ingress.yaml` : acces externe via URL

## Notebook

Le notebook `snake.ipynb` reste disponible pour l'exploration interactive :

```python
from snake_rl import make_env, SnakeAgent, build_training_config, visualize_episode

env = make_env(size=10)
cfg = build_training_config(n_episodes=1000)
agent = SnakeAgent(env, cfg["learning_rate"], 0.0, cfg["epsilon_decay"], cfg["final_epsilon"])
agent.load("artifacts/best_model.pkl")

html = visualize_episode(agent, env, greedy=True)
```

## Licence

MIT
