from __future__ import annotations

from typing import Any, List

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from .agent import SnakeAgent


def snake_to_frame(env: Any) -> np.ndarray:
    e = env.unwrapped
    size = e.size

    frame = np.zeros((size, size, 3), dtype=np.uint8)

    if getattr(e, "apple", None) is not None:
        ax, ay = int(e.apple[0]), int(e.apple[1])
        frame[ay, ax] = (180, 180, 180)

    for index, (x, y) in enumerate(list(e.snake)):
        if index == 0:
            frame[y, x] = (255, 255, 255)
        else:
            frame[y, x] = (120, 120, 120)

    return frame[::-1, :, :]


def rollout_frames(agent: SnakeAgent, env: Any, max_steps: int = 100000, greedy: bool = True) -> List[np.ndarray]:
    old_eps = agent.epsilon
    if greedy:
        agent.epsilon = 0.0

    obs, _ = env.reset()
    frames = [snake_to_frame(env)]
    terminated = False
    truncated = False
    dead = False
    steps = 0

    while not (terminated or truncated or dead) and steps < max_steps:
        action = agent.get_action(obs)
        obs, _, terminated, truncated, info = env.step(action)
        dead = bool(info.get("dead", False))
        frames.append(snake_to_frame(env))
        steps += 1

    agent.epsilon = old_eps
    return frames


def build_animation_html(frames: List[np.ndarray], interval_ms: int = 120) -> str:
    fig = plt.figure()
    image = plt.imshow(frames[0], interpolation="nearest")
    plt.axis("off")

    def update(index: int):
        image.set_data(frames[index])
        return (image,)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=interval_ms, blit=True)
    html = ani.to_jshtml()
    plt.close(fig)
    return html


def visualize_episode(agent: SnakeAgent, env: Any, max_steps: int = 100000, greedy: bool = True, interval_ms: int = 120) -> str:
    frames = rollout_frames(agent=agent, env=env, max_steps=max_steps, greedy=greedy)
    return build_animation_html(frames=frames, interval_ms=interval_ms)
