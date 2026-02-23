from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .agent import SnakeAgent


def evaluate_agent(agent: SnakeAgent, env: Any, num_episodes: int = 200, max_steps: int = 5000) -> Dict[str, Any]:
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    apples_list = []
    lengths_list = []
    steps_list = []
    died_list = []
    death_event_list = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        dead = False

        apples = 0
        steps = 0
        death_event = None

        while not (terminated or truncated or dead) and steps < max_steps:
            action = agent.get_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1

            dead = bool(info.get("dead", False))
            event = info.get("event")

            if event == "eat":
                apples += 1

            if dead and death_event is None:
                death_event = event

            if terminated and death_event is None:
                death_event = event

        apples_list.append(apples)
        lengths_list.append(int(info.get("length", 0)))
        steps_list.append(steps)
        died_list.append(bool(dead))
        death_event_list.append(death_event)

    agent.epsilon = old_epsilon

    apples_arr = np.array(apples_list, dtype=float)
    lengths_arr = np.array(lengths_list, dtype=float)
    steps_arr = np.array(steps_list, dtype=float)
    died_arr = np.array(died_list, dtype=bool)

    counts: Dict[str, int] = {}
    for event in death_event_list:
        key = str(event)
        counts[key] = counts.get(key, 0) + 1

    return {
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "avg_apples": float(apples_arr.mean()),
        "std_apples": float(apples_arr.std()),
        "avg_length": float(lengths_arr.mean()),
        "std_length": float(lengths_arr.std()),
        "avg_steps": float(steps_arr.mean()),
        "std_steps": float(steps_arr.std()),
        "death_rate": float(died_arr.mean()),
        "end_events": counts,
        "apples_quantiles_10_50_90": np.quantile(apples_arr, [0.1, 0.5, 0.9]).tolist(),
        "steps_quantiles_10_50_90": np.quantile(steps_arr, [0.1, 0.5, 0.9]).tolist(),
    }
