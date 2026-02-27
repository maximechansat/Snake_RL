from __future__ import annotations

import argparse
import json

import numpy as np

from .agent import SnakeAgent
from .env import make_env
from .evaluation import evaluate_agent
from .training import build_training_config, train_agent


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Snake Q-learning agent.")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--episodes", type=positive_int, default=10000)
    parser.add_argument("--model-path", type=str, default="artifacts/best_model.pkl")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible training and evaluation.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Q-learning learning rate.")
    parser.add_argument("--reward-wall", type=float, default=-5.0, help="Reward when hitting a wall.")
    parser.add_argument("--reward-self", type=float, default=-1.0, help="Reward when colliding with self.")
    parser.add_argument("--reward-eat", type=float, default=20.0, help="Reward when eating an apple.")
    parser.add_argument("--reward-filled-grid", type=float, default=10.0, help="Reward when filling the whole grid.")
    parser.add_argument("--reward-step", type=float, default=-0.01, help="Base reward for a regular move.")
    parser.add_argument("--reward-closer-bonus", type=float, default=0.1, help="Extra reward when a move gets closer to apple.")
    args = parser.parse_args()

    config = build_training_config(n_episodes=args.episodes, learning_rate=args.learning_rate)
    env = make_env(
        size=args.size,
        record_stats=True,
        buffer_length=args.episodes,
        reward_wall=args.reward_wall,
        reward_self=args.reward_self,
        reward_eat=args.reward_eat,
        reward_filled_grid=args.reward_filled_grid,
        reward_step=args.reward_step,
        reward_closer_bonus=args.reward_closer_bonus,
    )
    if args.seed is not None:
        np.random.seed(args.seed)
        env.action_space.seed(args.seed)

    agent = SnakeAgent(
        env=env,
        learning_rate=config["learning_rate"],
        initial_epsilon=config["start_epsilon"],
        epsilon_decay=config["epsilon_decay"],
        final_epsilon=config["final_epsilon"],
        discount_factor=config["discount_factor"],
    )

    train_agent(agent=agent, env=env, n_episodes=args.episodes, progress=True, seed=args.seed)
    metrics = evaluate_agent(agent=agent, env=env, num_episodes=args.eval_episodes, seed=args.seed)
    agent.save(args.model_path)

    print("Training completed.")
    print(f"Model saved to: {args.model_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
