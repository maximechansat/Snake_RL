"""Game visualization page."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components
from utils import inject_css, load_agent, load_config, load_manifest

st.set_page_config(page_title="Visualisation - Snake RL", layout="wide")
inject_css()

# Import apres le path setup
from snake_rl.visualization import build_animation_html, snake_to_frame  # noqa: E402

st.title("Visualisation du jeu")

manifest = load_manifest()
if not manifest:
    st.warning("Aucun modele disponible. Lancez le grid search d'abord.")
    st.stop()

run_ids = [r["run_id"] for r in manifest["runs"]]

# --- Selection du modele ---
selected_run = st.selectbox("Modele a visualiser", run_ids)
config = load_config(selected_run)

st.markdown(
    f"**Hyperparametres** : LR={config['learning_rate']}, "
    f"DF={config['discount_factor']}, "
    f"Episodes={int(config['n_episodes'])}"
)

# --- Lancement de la partie ---
if st.button("Lancer une partie", type="primary"):
    with st.spinner("Generation de la partie..."):
        grid_size = config.get("grid_size", 10)
        agent, env = load_agent(selected_run, grid_size=grid_size)

        # Rollout avec collecte de stats
        obs, info = env.reset()
        frames = [snake_to_frame(env)]
        apples = 0
        steps = 0
        death_cause = None
        terminated = truncated = dead = False

        while not (terminated or truncated or dead) and steps < 5000:
            action = agent.get_action(obs)
            obs, _, terminated, truncated, info = env.step(action)
            dead = bool(info.get("dead", False))
            frames.append(snake_to_frame(env))
            steps += 1

            if info.get("event") == "eat":
                apples += 1
            if dead:
                death_cause = info.get("event", "unknown")

        env.close()

        # Stats de la partie
        col1, col2, col3 = st.columns(3)
        col1.metric("Steps", steps)
        col2.metric("Pommes", apples)
        col3.metric("Fin", death_cause or "survie")

        # Animation
        html = build_animation_html(frames, interval_ms=100)
        components.html(html, height=600, scrolling=False)
