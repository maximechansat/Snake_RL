"""Shared utilities for Streamlit pages."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import fsspec
import streamlit as st

# Add src/ to path so we can import snake_rl
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from snake_rl.agent import SnakeAgent  # noqa: E402
from snake_rl.env import make_env  # noqa: E402

DEFAULT_ARTIFACTS_URI = str(Path(__file__).resolve().parent.parent / "artifacts" / "grid_results")
ARTIFACTS_URI = os.getenv("SNAKE_RL_ARTIFACTS_URI", DEFAULT_ARTIFACTS_URI)
S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL", "https://minio.lab.sspcloud.fr")
S3_ANON = os.getenv("S3_ANON", "false").lower() == "true"


def open_artifact(path: str, mode: str = "r"):
    """Open an artifact from local storage or S3, depending on ARTIFACTS_URI."""
    artifact_uri = f"{ARTIFACTS_URI.rstrip('/')}/{path}"
    storage_options = {}
    if artifact_uri.startswith("s3://"):
        storage_options = {"anon": S3_ANON, "client_kwargs": {"endpoint_url": S3_ENDPOINT_URL}}
    return fsspec.open(artifact_uri, mode=mode, **storage_options).open()


def inject_css() -> None:
    """Inject custom CSS into the page."""
    css_path = Path(__file__).parent / "style.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_manifest() -> dict | None:
    """Load the grid search manifest file."""
    try:
        with open_artifact("manifest.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


@st.cache_data(ttl=300)
def load_history(run_id: str) -> dict:
    """Load training history for a given run."""
    with open_artifact(f"{run_id}/history.json") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_metrics(run_id: str) -> dict:
    """Load evaluation metrics for a given run."""
    with open_artifact(f"{run_id}/metrics.json") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_config(run_id: str) -> dict:
    """Load config for a given run."""
    with open_artifact(f"{run_id}/config.json") as f:
        return json.load(f)


def load_agent(run_id: str, grid_size: int = 10) -> tuple[SnakeAgent, any]:
    """Load a trained agent and create a matching environment."""
    env = make_env(size=grid_size, record_stats=False)
    agent = SnakeAgent(
        env=env,
        learning_rate=0.01,
        initial_epsilon=0.0,
        epsilon_decay=0.0,
        final_epsilon=0.0,
    )
    with open_artifact(f"{run_id}/model.pkl", mode="rb") as f:
        agent.load(f)
    agent.epsilon = 0.0
    return agent, env
