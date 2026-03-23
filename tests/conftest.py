from __future__ import annotations

import pytest

from snake_rl.agent import SnakeAgent
from snake_rl.env import make_env
from snake_rl.training import build_training_config


@pytest.fixture
def env():
    """Petit environnement 6x6 pour des tests rapides."""
    return make_env(size=6, record_stats=False)


@pytest.fixture
def config():
    """Configuration minimale pour les tests."""
    return build_training_config(n_episodes=50)


@pytest.fixture
def agent(env, config):
    """Agent non entraine connecte a l'environnement de test."""
    return SnakeAgent(
        env=env,
        learning_rate=config["learning_rate"],
        initial_epsilon=config["start_epsilon"],
        epsilon_decay=config["epsilon_decay"],
        final_epsilon=config["final_epsilon"],
        discount_factor=config["discount_factor"],
    )
