# Snake_RL

Projet d'apprentissage par renforcement (Q-learning) applique au jeu Snake, implemente dans un notebook Jupyter.

## Prerequis

- Python 3.10+ (recommande)
- `pip`

## Installation

Depuis le dossier `Snake_RL` :

```bash
pip install -r requirements.txt
```

## Lancement

```bash
jupyter notebook snake.ipynb
```

Puis executer les cellules dans l'ordre.

## Dependances principales

- `gymnasium` : interface environnement RL
- `numpy` : calcul numerique
- `matplotlib` : visualisation des courbes et animations
- `tqdm` : barre de progression d'entrainement
- `ipython` / `jupyter` : execution et rendu notebook

## Structure

- `snake.ipynb` : code de l'environnement Snake, entrainement Q-learning, visualisations
- `requirements.txt` : dependances Python du projet

## Notes

- Les hyperparametres (episodes, epsilon, alpha, gamma) sont modifiables directement dans le notebook.
- Pour des resultats stables, lancer plusieurs milliers d'episodes d'entrainement.
