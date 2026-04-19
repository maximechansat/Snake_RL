from __future__ import annotations

from snake_rl.evaluation import evaluate_agent
from snake_rl.training import train_agent

EXPECTED_KEYS = {
    "num_episodes",
    "max_steps",
    "avg_apples",
    "std_apples",
    "avg_length",
    "std_length",
    "avg_steps",
    "std_steps",
    "death_rate",
    "end_events",
    "apples_quantiles_10_50_90",
    "steps_quantiles_10_50_90",
}


def test_evaluate_returns_expected_keys(agent, env):
    train_agent(agent, env, n_episodes=20, progress=False)
    metrics = evaluate_agent(agent, env, num_episodes=5)
    assert set(metrics.keys()) == EXPECTED_KEYS


def test_evaluate_num_episodes_matches(agent, env):
    train_agent(agent, env, n_episodes=20, progress=False)
    metrics = evaluate_agent(agent, env, num_episodes=5)
    assert metrics["num_episodes"] == 5


def test_evaluate_death_rate_between_zero_and_one(agent, env):
    train_agent(agent, env, n_episodes=20, progress=False)
    metrics = evaluate_agent(agent, env, num_episodes=10)
    assert 0.0 <= metrics["death_rate"] <= 1.0


def test_evaluate_restores_epsilon(agent, env):
    agent.epsilon = 0.42
    evaluate_agent(agent, env, num_episodes=5)
    assert agent.epsilon == 0.42
