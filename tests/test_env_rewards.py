from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from snake_rl.env import SnakeEnv


def test_size_below_3_is_rejected():
    with pytest.raises(ValueError, match="size must be >= 3"):
        SnakeEnv(size=2, render_mode=None)


def test_wall_collision_uses_configured_reward():
    env = SnakeEnv(size=5, render_mode=None, reward_wall=-9.0)
    env.reset(seed=0)

    _, reward, terminated, _, info = env.step(0)
    assert not terminated
    _, reward, terminated, _, info = env.step(0)
    assert not terminated
    _, reward, terminated, _, info = env.step(0)
    assert terminated
    assert reward == -9.0
    assert info["event"] == "wall"


def test_self_collision_uses_configured_reward():
    env = SnakeEnv(size=6, render_mode=None, reward_self=-7.0)
    env.reset(seed=0)
    env.snake = deque([(2, 2), (2, 3), (1, 3), (1, 2)])
    env.direction = np.array([1, 0], dtype=np.int32)
    env.apple = np.array([5, 5], dtype=np.int32)

    _, reward, terminated, _, info = env.step(1)
    assert terminated
    assert reward == -7.0
    assert info["event"] == "self"


def test_eat_and_move_rewards_are_configurable():
    env = SnakeEnv(size=6, render_mode=None, reward_eat=33.0, reward_step=-0.2, reward_closer_bonus=0.05)
    env.reset(seed=0)
    env.apple = np.array([4, 2], dtype=np.int32)

    _, reward, terminated, _, info = env.step(0)
    assert not terminated
    assert info["event"] == "move"
    assert reward == pytest.approx(-0.15)

    # Head is now at (4, 3), so place apple on the next forward cell.
    env.apple = np.array([5, 3], dtype=np.int32)
    _, reward, terminated, _, info = env.step(0)
    assert info["event"] == "eat"
    assert reward == 33.0
