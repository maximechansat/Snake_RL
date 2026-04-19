from __future__ import annotations

import gymnasium as gym
import numpy as np

from snake_rl.env import make_env


def test_make_env_returns_gymnasium_env(env):
    assert isinstance(env, gym.Env)


def test_reset_returns_obs_and_info(env):
    obs, info = env.reset(seed=42)
    assert "grid" in obs
    assert "dir" in obs
    assert "length" in info


def test_obs_shapes(env):
    obs, _ = env.reset(seed=42)
    assert obs["grid"].shape == (6, 6)
    assert obs["dir"].shape == (2,)


def test_step_returns_five_tuple(env):
    env.reset(seed=42)
    result = env.step(0)
    assert len(result) == 5
    obs, reward, terminated, truncated, info = result
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_initial_snake_length_is_three(env):
    _, info = env.reset(seed=42)
    assert info["length"] == 3


def test_reverse_direction_ignored(env):
    """Le snake commence vers la droite (dx=1). Aller a gauche (action=2) ne doit pas inverser."""
    env.reset(seed=42)
    head_before = env.unwrapped.snake[0]

    # Action 0 = droite (direction initiale)
    env.step(0)
    head_after_right = env.unwrapped.snake[0]
    assert head_after_right[0] == head_before[0] + 1

    # Action 2 = gauche (inverse de la direction courante), doit etre ignore
    env.step(2)
    head_after_left_attempt = env.unwrapped.snake[0]
    assert head_after_left_attempt[0] == head_after_right[0] + 1  # Continue a droite


def test_record_stats_wrapper():
    env = make_env(size=6, record_stats=True, buffer_length=10)
    assert hasattr(env, "return_queue")
    env.close()


def test_wall_collision_terminates(env):
    env.reset(seed=42)
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(0)
        if terminated:
            assert reward < 0
            assert info["event"] == "wall"
            return
    raise AssertionError("Le snake aurait dû mourir en heurtant le mur")


def test_eating_food_increases_length(env):
    env.reset(seed=42)
    initial_length = len(env.unwrapped.snake)
    head_x, head_y = env.unwrapped.snake[0]
    env.unwrapped.apple = np.array([head_x + 1, head_y], dtype=np.int32)

    obs, reward, terminated, truncated, info = env.step(0)

    assert not terminated
    assert not truncated
    assert reward > 0
    assert info["event"] == "eat"
    assert info["length"] == initial_length + 1
