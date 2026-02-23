"""Core package for the Snake RL project."""

from .agent import SnakeAgent
from .env import SnakeEnv, make_env, register_env
from .evaluation import evaluate_agent
from .training import train_agent
from .visualization import rollout_frames, visualize_episode

__all__ = [
    "SnakeAgent",
    "SnakeEnv",
    "make_env",
    "register_env",
    "evaluate_agent",
    "train_agent",
    "rollout_frames",
    "visualize_episode",
]
