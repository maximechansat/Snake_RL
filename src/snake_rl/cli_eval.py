from __future__ import annotations

import argparse
import json

from .agent import SnakeAgent
from .env import make_env
from .evaluation import evaluate_agent
from .training import build_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Snake Q-learning agent.")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--model-path", type=str, default="artifacts/best_model.pkl")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible evaluation.")
    args = parser.parse_args()

    config = build_training_config(n_episodes=1000)
    env = make_env(size=args.size, record_stats=False)
    agent = SnakeAgent(
        env=env,
        learning_rate=config["learning_rate"],
        initial_epsilon=0.0,
        epsilon_decay=config["epsilon_decay"],
        final_epsilon=config["final_epsilon"],
        discount_factor=config["discount_factor"],
    )
    agent.load(args.model_path)
    metrics = evaluate_agent(agent=agent, env=env, num_episodes=args.episodes, seed=args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
