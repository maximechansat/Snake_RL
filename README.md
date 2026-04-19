# Snake RL

Projet d'apprentissage par renforcement (Q-learning) applique au jeu Snake.
Dashboard interactif Streamlit pour comparer differentes configurations d'entrainement.
La page web est https://snake-rl-maximechansat.lab.sspcloud.fr.

## Prerequis

- Python 3.10+
- `pip`
- Acces S3/MinIO SSPCloud si vous voulez mettre a jour les artefacts d'entrainement

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

Les resultats sont ecrits localement dans `artifacts/grid_results/`, puis peuvent etre envoyes vers S3.

### Mettre a jour les artefacts S3

Les modeles entraines sont stockes sur le S3/MinIO du SSPCloud :

```text
s3://mchansat/snake-rl/artifacts/grid_results
```

Apres un entrainement, envoyez les artefacts vers S3 :

```bash
make upload-artifacts
```

Pour entrainer puis envoyer les artefacts :

```bash
make train-grid-upload
```

### Lancer le dashboard

```bash
make app
```

Ouvre ensuite http://localhost:8501 dans un navigateur.

Par defaut, le dashboard lit les artefacts locaux dans `artifacts/grid_results/`.
Pour lire les artefacts depuis S3 en acces public anonyme :

```bash
export SNAKE_RL_ARTIFACTS_URI="s3://mchansat/snake-rl/artifacts/grid_results"
export S3_ENDPOINT_URL="https://minio.lab.sspcloud.fr"
export S3_ANON="true"
streamlit run app/streamlit_app.py
```

Sous PowerShell :

```powershell
$env:SNAKE_RL_ARTIFACTS_URI="s3://mchansat/snake-rl/artifacts/grid_results"
$env:S3_ENDPOINT_URL="https://minio.lab.sspcloud.fr"
$env:S3_ANON="true"
streamlit run app/streamlit_app.py
```

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
- **Docker** : build et push de l'image sur `ghcr.io/maximechansat/snake_rl`
- **Deploy** : mise a jour automatique de `k8s/deployment.yaml` avec le SHA du commit

Le pipeline se declenche automatiquement a chaque push sur `main`.

## Deploiement (SSP Cloud)

L'application est deployee sur le SSP Cloud via ArgoCD (GitOps) :

1. Le CI push l'image Docker sur ghcr.io
2. Le CI remplace ensuite le tag d'image dans `k8s/deployment.yaml` par le SHA du commit
3. ArgoCD surveille le dossier `k8s/` du repo
4. Kubernetes deploie automatiquement cette image versionnee

Les manifestes Kubernetes se trouvent dans `k8s/` :
- `deployment.yaml` : configuration du container
- `service.yaml` : exposition du port 8501
- `ingress.yaml` : acces externe via URL

Le deploiement configure le dashboard pour lire les artefacts depuis S3 :

```yaml
SNAKE_RL_ARTIFACTS_URI=s3://mchansat/snake-rl/artifacts/grid_results
S3_ENDPOINT_URL=https://minio.lab.sspcloud.fr
S3_ANON=true
```

Le dossier S3 correspondant doit rester accessible en lecture anonyme, par exemple avec :

```bash
mc anonymous set download s3/mchansat/snake-rl/artifacts/
```

## Notebook

Le notebook `snake.ipynb` reste disponible pour l'exploration interactive :

```python
import os

from app.utils import load_agent
from snake_rl import visualize_episode

os.environ["SNAKE_RL_ARTIFACTS_URI"] = "s3://mchansat/snake-rl/artifacts/grid_results"
os.environ["S3_ENDPOINT_URL"] = "https://minio.lab.sspcloud.fr"
os.environ["S3_ANON"] = "true"

agent, env = load_agent("lr0.005_df0.9_ep50000", grid_size=10)

html = visualize_episode(agent, env, greedy=True)
```

## Licence

MIT
