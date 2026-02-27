from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_cli_train_help_exposes_new_arguments():
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else str(src)

    proc = subprocess.run(
        [sys.executable, "-m", "snake_rl.cli_train", "--help"],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        check=True,
    )
    help_text = proc.stdout
    for arg in (
        "--seed",
        "--learning-rate",
        "--reward-wall",
        "--reward-self",
        "--reward-eat",
        "--reward-filled-grid",
        "--reward-step",
        "--reward-closer-bonus",
    ):
        assert arg in help_text


def test_cli_train_rejects_non_positive_episodes():
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else str(src)

    proc = subprocess.run(
        [sys.executable, "-m", "snake_rl.cli_train", "--episodes", "0"],
        capture_output=True,
        text=True,
        cwd=root,
        env=env,
        check=False,
    )
    assert proc.returncode != 0
    assert "expected a positive integer" in proc.stderr
