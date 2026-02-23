from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

ENV_ID = "gymnasium_env/SnakeEnv-v1"


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 8}

    def __init__(self, size: int = 10, render_mode: Optional[str] = "human"):
        self.size = size
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Dict(
            {
                "grid": gym.spaces.Box(low=0, high=3, shape=(size, size), dtype=np.int8),
                "dir": gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int8),
            }
        )

        self._action_to_dir = {
            0: np.array([1, 0], dtype=np.int32),
            1: np.array([0, 1], dtype=np.int32),
            2: np.array([-1, 0], dtype=np.int32),
            3: np.array([0, -1], dtype=np.int32),
        }

        self.snake = deque()
        self.direction = np.array([1, 0], dtype=np.int32)
        self.apple = np.array([0, 0], dtype=np.int32)

    def _spawn_apple(self) -> Optional[np.ndarray]:
        occupied = set(self.snake)
        free = [(x, y) for x in range(self.size) for y in range(self.size) if (x, y) not in occupied]
        if not free:
            return None
        x, y = free[self.np_random.integers(0, len(free))]
        return np.array([x, y], dtype=np.int32)

    def _get_obs(self) -> Dict[str, np.ndarray]:
        grid = np.zeros((self.size, self.size), dtype=np.int8)
        for (x, y) in list(self.snake)[1:]:
            grid[y, x] = 1

        hx, hy = self.snake[0]
        grid[hy, hx] = 2

        if self.apple is not None:
            ax, ay = int(self.apple[0]), int(self.apple[1])
            grid[ay, ax] = 3

        return {"grid": grid, "dir": self.direction.astype(np.int8)}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        cx, cy = self.size // 2, self.size // 2
        self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.direction = np.array([1, 0], dtype=np.int32)

        self.apple = self._spawn_apple()
        obs = self._get_obs()
        info = {"length": len(self.snake)}
        return obs, info

    @staticmethod
    def manhattan(a: Any, b: Any) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, action: int):
        new_dir = self._action_to_dir[int(action)]

        if np.array_equal(new_dir, -self.direction):
            new_dir = self.direction
        else:
            self.direction = new_dir

        hx, hy = self.snake[0]
        nx, ny = hx + int(self.direction[0]), hy + int(self.direction[1])

        terminated = False
        truncated = False
        reward = 0.0

        if nx < 0 or nx >= self.size or ny < 0 or ny >= self.size:
            terminated = True
            reward = -5.0
            return self._get_obs(), reward, terminated, truncated, {"length": len(self.snake), "dead": True, "event": "wall"}

        old_dist = self.manhattan(self.snake[0], self.apple)
        new_head = (nx, ny)
        will_eat = self.apple is not None and nx == int(self.apple[0]) and ny == int(self.apple[1])
        body_set = set(self.snake)
        tail = self.snake[-1]

        if (new_head in body_set) and not (new_head == tail and not will_eat):
            terminated = True
            reward = -1.0
            return self._get_obs(), reward, terminated, truncated, {"length": len(self.snake), "dead": True, "event": "self"}

        self.snake.appendleft(new_head)
        new_dist = self.manhattan(new_head, self.apple)

        if will_eat:
            reward = 20.0
            self.apple = self._spawn_apple()
            if self.apple is None:
                terminated = True
                reward = 10.0
                info = {"length": len(self.snake), "dead": False, "event": "filled_grid"}
            else:
                info = {"length": len(self.snake), "dead": False, "event": "eat"}
        else:
            self.snake.pop()
            reward = -0.01
            info = {"length": len(self.snake), "dead": False, "event": "move"}
            if new_dist < old_dist:
                reward += 0.1

        return self._get_obs(), reward, terminated, truncated, info


def register_env(env_id: str = ENV_ID) -> None:
    try:
        gym.spec(env_id)
    except gym.error.Error:
        gym.register(id=env_id, entry_point=SnakeEnv, max_episode_steps=100000)


def make_env(size: int = 10, record_stats: bool = False, buffer_length: int = 1000, env_id: str = ENV_ID) -> gym.Env:
    register_env(env_id)
    env = gym.make(env_id, size=size)
    if record_stats:
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=buffer_length)
    return env
