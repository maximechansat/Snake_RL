from __future__ import annotations

import argparse
from pathlib import Path

from .agent import SnakeAgent
from .env import make_env
from .training import build_training_config
from .visualization import visualize_episode


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a Snake autoplay HTML visualization from a trained model.")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--model-path", type=str, default="artifacts/best_model.pkl")
    parser.add_argument("--output", type=str, default="artifacts/episode_preview.html")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--interval-ms", type=int, default=120)
    parser.add_argument("--greedy", action="store_true", help="Run with epsilon forced to 0.")
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

    html = visualize_episode(
        agent=agent,
        env=env,
        max_steps=args.max_steps,
        greedy=args.greedy,
        interval_ms=args.interval_ms,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Visualization saved to: {output_path}")


if __name__ == "__main__":
    main()
