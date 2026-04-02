"""Custom training page."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import streamlit.components.v1 as components
from utils import inject_css

st.set_page_config(page_title="Entrainement - Snake RL", layout="wide")
inject_css()

from snake_rl.agent import SnakeAgent  # noqa: E402
from snake_rl.env import make_env  # noqa: E402
from snake_rl.evaluation import evaluate_agent  # noqa: E402
from snake_rl.training import build_training_config, train_agent  # noqa: E402
from snake_rl.visualization import build_animation_html, snake_to_frame  # noqa: E402

st.title("Entrainement personnalise")

st.markdown("Choisissez des hyperparametres et lancez un entrainement.")

# --- Hyperparametres ---
col1, col2 = st.columns(2)

with col1:
    learning_rate = st.slider("Learning rate", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f")
    discount_factor = st.slider("Discount factor", min_value=0.80, max_value=0.99, value=0.95, step=0.01)

with col2:
    n_episodes = st.number_input("Nombre d'episodes", min_value=1000, max_value=100000, value=10000, step=1000)
    final_epsilon = st.slider("Final epsilon", min_value=0.0, max_value=0.1, value=0.02, step=0.01)

# --- Entrainement ---
if st.button("Entrainer", type="primary"):
    config = build_training_config(
        n_episodes=n_episodes,
        learning_rate=learning_rate,
        final_epsilon=final_epsilon,
        discount_factor=discount_factor,
    )

    env = make_env(size=10, record_stats=False)
    agent = SnakeAgent(
        env=env,
        learning_rate=config["learning_rate"],
        initial_epsilon=config["start_epsilon"],
        epsilon_decay=config["epsilon_decay"],
        final_epsilon=config["final_epsilon"],
        discount_factor=config["discount_factor"],
    )

    # Entrainement par batches pour la barre de progression
    progress_bar = st.progress(0, text="Entrainement en cours...")
    batch_size = max(1, n_episodes // 100)

    for i in range(0, n_episodes, batch_size):
        chunk = min(batch_size, n_episodes - i)
        train_agent(agent, env, chunk, progress=False)
        progress_bar.progress((i + chunk) / n_episodes, text=f"Episode {i + chunk}/{n_episodes}")

    progress_bar.progress(1.0, text="Entrainement termine !")

    # Sauvegarde dans session_state
    st.session_state["trained_agent"] = agent
    st.session_state["trained_env"] = env

    # Evaluation
    st.subheader("Resultats")
    metrics = evaluate_agent(agent, env, num_episodes=100)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pommes (moy)", f"{metrics['avg_apples']:.1f}")
    col2.metric("Steps (moy)", f"{metrics['avg_steps']:.0f}")
    col3.metric("Taux de mort", f"{metrics['death_rate']:.0%}")
    col4.metric("Longueur (moy)", f"{metrics['avg_length']:.1f}")

    env.close()

# --- Visualisation de l'agent entraine ---
if "trained_agent" in st.session_state:
    st.divider()
    if st.button("Voir une partie"):
        agent = st.session_state["trained_agent"]
        env = make_env(size=10, record_stats=False)

        with st.spinner("Generation de la partie..."):
            obs, _ = env.reset()
            frames = [snake_to_frame(env)]
            agent.epsilon = 0.0
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

            c1, c2, c3 = st.columns(3)
            c1.metric("Steps", steps)
            c2.metric("Pommes", apples)
            c3.metric("Fin", death_cause or "survie")

            html = build_animation_html(frames, interval_ms=100)
            components.html(html, height=600, scrolling=False)
