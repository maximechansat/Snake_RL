from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .agent import SnakeAgent


def _head_to_apple_manhattan(obs: Dict[str, np.ndarray]) -> int | None:
    grid = obs["grid"]
    head_pos = np.argwhere(grid == 2)
    apple_pos = np.argwhere(grid == 3)

    if head_pos.size == 0 or apple_pos.size == 0:
        return None

    hy, hx = head_pos[0]
    ay, ax = apple_pos[0]
    return int(abs(hx - ax) + abs(hy - ay))


def aggregate_runs(results: list[Dict[str, Any]]) -> Dict[str, Any]:
    kpi_keys = [
        "avg_apples",
        "apple_rate_per_1000_steps",
        "avg_efficiency_gap",
        "death_rate",
        "avg_steps",
    ]

    summary: Dict[str, Any] = {"num_runs": len(results)}

    for key in kpi_keys:
        values = [
            float(run[key])
            for run in results
            if key in run and run[key] is not None and np.isfinite(run[key])
        ]

        if not values:
            summary[key] = {"mean": None, "std": None, "p10": None}
            continue

        arr = np.array(values, dtype=float)
        summary[key] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "p10": float(np.quantile(arr, 0.1)),
        }

    return summary


def evaluate_agent(
    agent: SnakeAgent,
    env: Any,
    num_episodes: int = 200,
    max_steps: int = 5000,
    seed: int | None = None,
) -> Dict[str, Any]:
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        apples_list = []
        lengths_list = []
        steps_list = []
        died_list = []
        death_event_list = []
        efficiency_gaps = []

        for episode_idx in range(num_episodes):
            reset_seed = (seed + episode_idx) if seed is not None else None
            obs, info = env.reset(seed=reset_seed)
            terminated = False
            truncated = False
            dead = False

            apples = 0
            steps = 0
            death_event = None
            steps_since_target = 0
            target_start_distance = _head_to_apple_manhattan(obs)

            while not (terminated or truncated or dead) and steps < max_steps:
                action = agent.get_action(obs)
                obs, _, terminated, truncated, info = env.step(action)
                steps += 1
                steps_since_target += 1

                dead = bool(info.get("dead", False))
                event = info.get("event")

                if event == "eat":
                    apples += 1
                    if target_start_distance is not None:
                        efficiency_gaps.append(float(steps_since_target - target_start_distance))
                    steps_since_target = 0
                    target_start_distance = _head_to_apple_manhattan(obs)

                if dead and death_event is None:
                    death_event = event

                if terminated and death_event is None:
                    death_event = event

            apples_list.append(apples)
            lengths_list.append(int(info.get("length", 0)))
            steps_list.append(steps)
            died_list.append(bool(dead))
            death_event_list.append(death_event)

        apples_arr = np.array(apples_list, dtype=float)
        steps_arr = np.array(steps_list, dtype=float)
        died_arr = np.array(died_list, dtype=bool)
        gap_arr = np.array(efficiency_gaps, dtype=float)
        total_apples = float(apples_arr.sum())
        total_steps = float(steps_arr.sum())

        counts: Dict[str, int] = {}
        for event in death_event_list:
            key = str(event)
            counts[key] = counts.get(key, 0) + 1

        return {
            "num_episodes": num_episodes,
            "max_steps": max_steps,
            "seed": seed,
            "avg_apples": float(apples_arr.mean()),
            "std_apples": float(apples_arr.std()),
            "avg_steps": float(steps_arr.mean()),
            "std_steps": float(steps_arr.std()),
            "apple_rate_per_1000_steps": (1000.0 * total_apples / total_steps) if total_steps > 0 else 0.0,
            "death_rate": float(died_arr.mean()),
            "end_events": counts,
            "apples_quantiles_10_50_90": np.quantile(apples_arr, [0.1, 0.5, 0.9]).tolist(),
            "steps_quantiles_10_50_90": np.quantile(steps_arr, [0.1, 0.5, 0.9]).tolist(),
            "avg_efficiency_gap": float(gap_arr.mean()) if gap_arr.size else None,
        }
    finally:
        agent.epsilon = old_epsilon
