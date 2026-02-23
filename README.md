# Snake_RL

Projet d'apprentissage par renforcement (Q-learning) applique au jeu Snake.

## Prerequis

- Python 3.10+
- `pip`

## Installation

Depuis le dossier `Snake_RL` :

```bash
pip install -r requirements.txt
```

## Structure

- `snake.ipynb` : notebook de demo
- `src/snake_rl/env.py` : environnement Gymnasium
- `src/snake_rl/agent.py` : agent Q-learning + save/load des poids
- `src/snake_rl/training.py` : boucle d'entrainement
- `src/snake_rl/evaluation.py` : metriques d'evaluation
- `src/snake_rl/visualization.py` : visualisation d'une partie auto
- `src/snake_rl/cli_train.py` : CLI d'entrainement
- `src/snake_rl/cli_eval.py` : CLI d'evaluation
- `src/snake_rl/cli_visualize.py` : CLI d'export de demo HTML
- `requirements.txt` : dependances

## Utilisation rapide

1. Configurer l'import local du package:

```powershell
$env:PYTHONPATH="src"
```

2. Entraine le modele:

```powershell
python -m snake_rl.cli_train --episodes 10000 --model-path artifacts/best_model.pkl
```

3. Evalue le modele:

```powershell
python -m snake_rl.cli_eval --model-path artifacts/best_model.pkl --episodes 200
```

4. Genere une visualisation auto en HTML:

```powershell
python -m snake_rl.cli_visualize --model-path artifacts/best_model.pkl --output artifacts/episode_preview.html --greedy
```

Ouvre ensuite `artifacts/episode_preview.html` dans un navigateur.

## Utiliser dans un notebook

```powershell
$env:PYTHONPATH="src"
```

```python
from snake_rl.env import make_env
from snake_rl.agent import SnakeAgent
from snake_rl.training import build_training_config
from snake_rl.visualization import visualize_episode

env = make_env(size=10)
cfg = build_training_config(n_episodes=1000)
agent = SnakeAgent(env, cfg["learning_rate"], 0.0, cfg["epsilon_decay"], cfg["final_epsilon"], cfg["discount_factor"])
agent.load("artifacts/best_model.pkl")

html = visualize_episode(agent, env, greedy=True)
```

Dans Jupyter:

```python
from IPython.display import HTML
HTML(html)
```

## Notes

- Le notebook reste utile pour l'exploration, mais la logique principale est maintenant dans `src/`.
- Les poids du meilleur modele sont sauvegardes via `SnakeAgent.save(...)`.
