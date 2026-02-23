from __future__ import annotations

from typing import Any, Dict

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from .agent import SnakeAgent


def train_agent(agent: SnakeAgent, env: gym.Env, n_episodes: int, progress: bool = True) -> None:
    iterator = tqdm(range(n_episodes)) if progress else range(n_episodes)

    for _ in iterator:
        obs, _ = env.reset()
        terminated = False
        truncated = False
        dead = False

        while not (terminated or truncated or dead):
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            dead = bool(info.get("dead", False))
            done = terminated or truncated or dead

            agent.update(obs, action, reward, done, next_obs)
            obs = next_obs

        agent.decay_epsilon()


def get_moving_average(values: Any, window: int, mode: str = "valid") -> np.ndarray:
    values_np = np.array(values).flatten()
    return np.convolve(values_np, np.ones(window), mode=mode) / window


def build_training_config(
    n_episodes: int = 100000,
    learning_rate: float = 0.01,
    start_epsilon: float = 1.0,
    final_epsilon: float = 0.02,
    discount_factor: float = 0.95,
) -> Dict[str, float]:
    epsilon_decay = start_epsilon / (0.8 * n_episodes)
    return {
        "n_episodes": n_episodes,
        "learning_rate": learning_rate,
        "start_epsilon": start_epsilon,
        "epsilon_decay": epsilon_decay,
        "final_epsilon": final_epsilon,
        "discount_factor": discount_factor,
    }
