from __future__ import annotations

import argparse
import json

from .agent import SnakeAgent
from .env import make_env
from .evaluation import evaluate_agent
from .training import build_training_config, train_agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Snake Q-learning agent.")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--model-path", type=str, default="artifacts/best_model.pkl")
    parser.add_argument("--eval-episodes", type=int, default=100)
    args = parser.parse_args()

    config = build_training_config(n_episodes=args.episodes)
    env = make_env(size=args.size, record_stats=True, buffer_length=args.episodes)
    agent = SnakeAgent(
        env=env,
        learning_rate=config["learning_rate"],
        initial_epsilon=config["start_epsilon"],
        epsilon_decay=config["epsilon_decay"],
        final_epsilon=config["final_epsilon"],
        discount_factor=config["discount_factor"],
    )

    train_agent(agent=agent, env=env, n_episodes=args.episodes, progress=True)
    metrics = evaluate_agent(agent=agent, env=env, num_episodes=args.eval_episodes)
    agent.save(args.model_path)

    print("Training completed.")
    print(f"Model saved to: {args.model_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
