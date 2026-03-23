from __future__ import annotations

from snake_rl.training import build_training_config, train_agent


def test_train_agent_runs(agent, env):
    train_agent(agent, env, n_episodes=20, progress=False)


def test_training_populates_q_values(agent, env):
    train_agent(agent, env, n_episodes=50, progress=False)
    assert len(agent.q_values) > 0


def test_training_populates_error(agent, env):
    train_agent(agent, env, n_episodes=20, progress=False)
    assert len(agent.training_error) > 0


def test_build_training_config_keys():
    config = build_training_config()
    expected = {"n_episodes", "learning_rate", "start_epsilon", "epsilon_decay", "final_epsilon", "discount_factor"}
    assert set(config.keys()) == expected


def test_build_training_config_epsilon_decay():
    config = build_training_config(n_episodes=10000, start_epsilon=1.0)
    assert config["epsilon_decay"] == 1.0 / (0.8 * 10000)
