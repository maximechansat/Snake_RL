from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

import gymnasium as gym
import numpy as np

State = Tuple[int, ...]


class SnakeAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs: dict) -> int:
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return int(np.argmax(self.q_values[self.obs_to_state(obs)]))

    @staticmethod
    def obs_to_state(obs: dict) -> State:
        grid = obs["grid"]
        dx, dy = int(obs["dir"][0]), int(obs["dir"][1])

        hy, hx = np.argwhere(grid == 2)[0]
        apple_pos = np.argwhere(grid == 3)
        if apple_pos.size == 0:
            ay, ax = hy, hx
        else:
            ay, ax = apple_pos[0]

        body = set(map(tuple, np.argwhere(grid == 1)))
        size = grid.shape[0]
        neck = (hy - dy, hx - dx)

        def body_degree(y: int, x: int) -> int:
            neighbors = ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
            return sum((ny, nx) in body for (ny, nx) in neighbors)

        def collision(y: int, x: int) -> int:
            if x < 0 or x >= size or y < 0 or y >= size:
                return 1
            if (y, x) in body:
                will_eat = (y == ay and x == ax)
                is_tail_candidate = (y, x) != neck and body_degree(y, x) <= 1
                if is_tail_candidate and not will_eat:
                    return 0
                return 1
            return 0

        dir_up = int(dy == 1)
        dir_down = int(dy == -1)
        dir_right = int(dx == 1)
        dir_left = int(dx == -1)

        danger_front = collision(hy + dy, hx + dx)
        danger_left = collision(hy + dx, hx - dy)
        danger_right = collision(hy - dx, hx + dy)

        dx_food = int(np.sign(ax - hx))
        dy_food = int(np.sign(ay - hy))

        return (
            danger_front,
            danger_left,
            danger_right,
            dir_up,
            dir_down,
            dir_left,
            dir_right,
            dx_food,
            dy_food,
        )

    def update(self, obs: dict, action: int, reward: float, done: bool, next_obs: dict) -> None:
        state = self.obs_to_state(obs)
        next_state = self.obs_to_state(next_obs)

        future_q_value = (not done) * np.max(self.q_values[next_state])
        target = reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[state][action]
        self.q_values[state][action] = self.q_values[state][action] + self.lr * temporal_difference
        self.training_error.append(temporal_difference)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "q_values": {state: values for state, values in self.q_values.items()},
            "epsilon": self.epsilon,
            "lr": self.lr,
            "discount_factor": self.discount_factor,
            "epsilon_decay": self.epsilon_decay,
            "final_epsilon": self.final_epsilon,
        }
        with path.open("wb") as f:
            pickle.dump(payload, f)

    def load(self, path: str | Path) -> None:
        with Path(path).open("rb") as f:
            payload = pickle.load(f)

        q_values: Dict[State, np.ndarray] = payload["q_values"]
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.q_values.update(q_values)
        self.epsilon = payload.get("epsilon", self.epsilon)
