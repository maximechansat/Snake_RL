"""Grid search script: train agents with different hyperparameter combinations."""

from __future__ import annotations

import argparse
import json
import logging
from itertools import product
from pathlib import Path

from .agent import SnakeAgent
from .env import make_env
from .evaluation import evaluate_agent
from .training import build_training_config, train_agent

logger = logging.getLogger(__name__)

DEFAULT_LEARNING_RATES = [0.005, 0.01, 0.05, 0.1]
DEFAULT_DISCOUNT_FACTORS = [0.90, 0.95, 0.99]
DEFAULT_EPISODES = 50_000


def run_grid_search(
    output_dir: Path,
    learning_rates: list[float] | None = None,
    discount_factors: list[float] | None = None,
    n_episodes: int = DEFAULT_EPISODES,
    grid_size: int = 10,
    eval_episodes: int = 200,
) -> dict:
    """Run a grid search and save results to output_dir."""
    learning_rates = learning_rates or DEFAULT_LEARNING_RATES
    discount_factors = discount_factors or DEFAULT_DISCOUNT_FACTORS
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []

    for lr, df in product(learning_rates, discount_factors):
        run_id = f"lr{lr}_df{df}_ep{n_episodes}"
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Training %s ...", run_id)

        config = build_training_config(
            n_episodes=n_episodes,
            learning_rate=lr,
            discount_factor=df,
        )

        env = make_env(size=grid_size, record_stats=True, buffer_length=n_episodes)
        agent = SnakeAgent(
            env=env,
            learning_rate=config["learning_rate"],
            initial_epsilon=config["start_epsilon"],
            epsilon_decay=config["epsilon_decay"],
            final_epsilon=config["final_epsilon"],
            discount_factor=config["discount_factor"],
        )

        train_agent(agent, env, n_episodes=n_episodes, progress=True)

        # Training history from RecordEpisodeStatistics wrapper
        history = {
            "episode_rewards": [float(r) for r in env.return_queue],
            "episode_lengths": [int(length) for length in env.length_queue],
        }

        metrics = evaluate_agent(agent, env, num_episodes=eval_episodes)

        # Save model, config, metrics, history
        agent.save(run_dir / "model.pkl")

        with open(run_dir / "config.json", "w") as f:
            json.dump(
                {**config, "grid_size": grid_size, "run_id": run_id},
                f,
                indent=2,
            )

        with open(run_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f)

        runs.append(
            {
                "run_id": run_id,
                "config": {**config, "grid_size": grid_size},
                "metrics_summary": {
                    "avg_apples": metrics["avg_apples"],
                    "avg_steps": metrics["avg_steps"],
                    "death_rate": metrics["death_rate"],
                },
            }
        )

        logger.info("  -> avg_apples=%.1f, death_rate=%.2f", metrics["avg_apples"], metrics["death_rate"])
        env.close()

    # Write manifest with all runs
    manifest = {"runs": runs}
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Grid search done. %d configs saved to %s", len(runs), output_dir)
    return manifest


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Hyperparameter grid search for Snake RL")
    parser.add_argument("--output-dir", type=str, default="artifacts/grid_results")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--eval-episodes", type=int, default=200)
    args = parser.parse_args()

    run_grid_search(
        output_dir=Path(args.output_dir),
        n_episodes=args.episodes,
        grid_size=args.size,
        eval_episodes=args.eval_episodes,
    )


if __name__ == "__main__":
    main()
