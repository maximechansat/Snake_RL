from __future__ import annotations

import numpy as np
import pytest

from snake_rl.agent import SnakeAgent
from snake_rl.env import make_env
from snake_rl.training import build_training_config, train_agent


def _new_agent_and_env(seed: int):
    env = make_env(size=6, record_stats=False)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    cfg = build_training_config(n_episodes=20, learning_rate=0.02)
    agent = SnakeAgent(
        env=env,
        learning_rate=cfg["learning_rate"],
        initial_epsilon=cfg["start_epsilon"],
        epsilon_decay=cfg["epsilon_decay"],
        final_epsilon=cfg["final_epsilon"],
        discount_factor=cfg["discount_factor"],
    )
    return agent, env


def _q_snapshot(agent: SnakeAgent):
    return {k: tuple(np.round(v, 8)) for k, v in agent.q_values.items()}


def test_build_training_config_uses_given_learning_rate():
    cfg = build_training_config(n_episodes=100, learning_rate=0.123)
    assert cfg["learning_rate"] == 0.123
    assert cfg["epsilon_decay"] > 0.0


def test_build_training_config_rejects_non_positive_episodes():
    with pytest.raises(ValueError, match="n_episodes must be > 0"):
        build_training_config(n_episodes=0)


def test_train_agent_is_deterministic_with_seed():
    agent_a, env_a = _new_agent_and_env(seed=7)
    agent_b, env_b = _new_agent_and_env(seed=7)

    train_agent(agent=agent_a, env=env_a, n_episodes=12, progress=False, seed=7)
    train_agent(agent=agent_b, env=env_b, n_episodes=12, progress=False, seed=7)

    assert _q_snapshot(agent_a) == _q_snapshot(agent_b)
    assert agent_a.epsilon == agent_b.epsilon
