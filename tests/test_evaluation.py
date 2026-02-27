from __future__ import annotations

import numpy as np
import pytest

from snake_rl.agent import SnakeAgent
from snake_rl.env import make_env
from snake_rl.evaluation import aggregate_runs, evaluate_agent


def _make_agent_and_env(seed: int = 0):
    env = make_env(size=6, record_stats=False)
    obs, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    np.random.seed(seed)
    agent = SnakeAgent(
        env=env,
        learning_rate=0.01,
        initial_epsilon=0.0,
        epsilon_decay=0.0,
        final_epsilon=0.0,
        discount_factor=0.95,
    )
    _ = obs
    return agent, env


def test_evaluate_agent_returns_expected_keys_and_restores_epsilon():
    agent, env = _make_agent_and_env(seed=123)
    agent.epsilon = 0.42

    metrics = evaluate_agent(agent=agent, env=env, num_episodes=5, max_steps=100, seed=123)

    expected = {
        "num_episodes",
        "max_steps",
        "seed",
        "avg_apples",
        "std_apples",
        "avg_steps",
        "std_steps",
        "apple_rate_per_1000_steps",
        "death_rate",
        "end_events",
        "apples_quantiles_10_50_90",
        "steps_quantiles_10_50_90",
        "avg_efficiency_gap",
    }
    assert expected.issubset(metrics.keys())
    assert agent.epsilon == 0.42


def test_aggregate_runs_ignores_none_and_non_finite_values():
    results = [
        {"avg_apples": 1.0, "apple_rate_per_1000_steps": 5.0, "avg_efficiency_gap": None, "death_rate": 1.0, "avg_steps": 20},
        {"avg_apples": 3.0, "apple_rate_per_1000_steps": float("nan"), "avg_efficiency_gap": 2.0, "death_rate": 0.8, "avg_steps": 25},
        {"avg_apples": float("inf"), "apple_rate_per_1000_steps": 9.0, "avg_efficiency_gap": 3.0, "death_rate": 0.9, "avg_steps": 30},
    ]

    summary = aggregate_runs(results)

    assert summary["num_runs"] == 3
    assert summary["avg_apples"]["mean"] == 2.0
    assert summary["apple_rate_per_1000_steps"]["mean"] == 7.0
    assert summary["avg_efficiency_gap"]["mean"] == 2.5


class _FailingEnv:
    def reset(self, seed=None):
        obs = {
            "grid": np.zeros((3, 3), dtype=np.int8),
            "dir": np.array([1, 0], dtype=np.int8),
        }
        obs["grid"][1, 1] = 2
        obs["grid"][1, 2] = 3
        return obs, {"length": 1}

    def step(self, action):
        raise RuntimeError("step failed")


def test_evaluate_agent_restores_epsilon_when_exception_occurs():
    agent, _ = _make_agent_and_env(seed=5)
    agent.epsilon = 0.73

    with pytest.raises(RuntimeError, match="step failed"):
        evaluate_agent(agent=agent, env=_FailingEnv(), num_episodes=1, max_steps=10, seed=1)

    assert agent.epsilon == 0.73
