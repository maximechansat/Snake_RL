from __future__ import annotations

import numpy as np

from snake_rl.agent import SnakeAgent


def test_agent_creation(agent):
    assert agent.lr == 0.01
    assert agent.epsilon == 1.0
    assert len(agent.q_values) == 0


def test_get_action_returns_valid_int(agent, env):
    obs, _ = env.reset(seed=42)
    action = agent.get_action(obs)
    assert int(action) == action
    assert 0 <= action < 4


def test_obs_to_state_returns_tuple_of_nine(agent, env):
    obs, _ = env.reset(seed=42)
    state = SnakeAgent.obs_to_state(obs)
    assert isinstance(state, tuple)
    assert len(state) == 9
    assert all(v in (-1, 0, 1) for v in state)


def test_update_changes_q_values(agent, env):
    obs, _ = env.reset(seed=42)
    state = SnakeAgent.obs_to_state(obs)
    q_before = agent.q_values[state].copy()

    action = 0
    next_obs, reward, _, _, _ = env.step(action)
    agent.update(obs, action, reward, False, next_obs)

    assert not np.array_equal(agent.q_values[state], q_before)


def test_decay_epsilon(agent):
    initial = agent.epsilon
    for _ in range(100):
        agent.decay_epsilon()
    assert agent.epsilon < initial
    assert agent.epsilon >= agent.final_epsilon


def test_save_and_load_roundtrip(agent, env, tmp_path):
    obs, _ = env.reset(seed=42)
    action = agent.get_action(obs)
    next_obs, reward, _, _, _ = env.step(action)
    agent.update(obs, action, reward, False, next_obs)

    path = tmp_path / "model.pkl"
    agent.save(path)

    new_agent = SnakeAgent(
        env=env,
        learning_rate=0.01,
        initial_epsilon=1.0,
        epsilon_decay=0.001,
        final_epsilon=0.02,
    )
    new_agent.load(path)

    state = SnakeAgent.obs_to_state(obs)
    np.testing.assert_array_equal(agent.q_values[state], new_agent.q_values[state])
