"""Training comparison page."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow imports from app/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from utils import inject_css, load_config, load_history, load_manifest, load_metrics

st.set_page_config(page_title="Comparaison - Snake RL", layout="wide")
inject_css()

st.title("Comparaison des entrainements")

manifest = load_manifest()
if not manifest:
    st.warning("Aucun resultat disponible. Lancez le grid search d'abord.")
    st.stop()

run_ids = [r["run_id"] for r in manifest["runs"]]

# --- Selection des runs a comparer ---
selected = st.multiselect("Configurations a comparer", run_ids, default=run_ids)

if not selected:
    st.info("Selectionnez au moins une configuration.")
    st.stop()

# --- Tableau recapitulatif ---
st.subheader("Metriques d'evaluation")

rows = []
for run_id in selected:
    config = load_config(run_id)
    metrics = load_metrics(run_id)
    rows.append(
        {
            "Run": run_id,
            "Learning Rate": config["learning_rate"],
            "Discount Factor": config["discount_factor"],
            "Episodes": int(config["n_episodes"]),
            "Pommes (moy)": round(metrics["avg_apples"], 1),
            "Pommes (std)": round(metrics["std_apples"], 1),
            "Steps (moy)": round(metrics["avg_steps"], 0),
            "Taux de mort": round(metrics["death_rate"], 2),
        }
    )

df = pd.DataFrame(rows)
st.dataframe(df, use_container_width=True, hide_index=True)

# --- Courbes d'entrainement ---
st.subheader("Courbes d'entrainement")

smoothing = st.slider("Fenetre de lissage (episodes)", min_value=10, max_value=5000, value=500, step=10)

fig_reward = go.Figure()
for run_id in selected:
    history = load_history(run_id)
    rewards = np.array(history["episode_rewards"])
    if len(rewards) > smoothing:
        smoothed = np.convolve(rewards, np.ones(smoothing) / smoothing, mode="valid")
        x = list(range(smoothing, smoothing + len(smoothed)))
        fig_reward.add_trace(go.Scatter(x=x, y=smoothed, mode="lines", name=run_id))

fig_reward.update_layout(
    title="Recompense par episode (lissee)",
    xaxis_title="Episode",
    yaxis_title="Recompense",
    template="plotly_dark",
    height=500,
)
st.plotly_chart(fig_reward, use_container_width=True)

# --- Bar chart des metriques ---
st.subheader("Comparaison des metriques")

col1, col2 = st.columns(2)

with col1:
    apples_data = []
    for run_id in selected:
        metrics = load_metrics(run_id)
        apples_data.append({"Run": run_id, "Pommes moyennes": metrics["avg_apples"]})
    fig_apples = px.bar(pd.DataFrame(apples_data), x="Run", y="Pommes moyennes", color="Run", template="plotly_dark")
    fig_apples.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_apples, use_container_width=True)

with col2:
    death_data = []
    for run_id in selected:
        metrics = load_metrics(run_id)
        death_data.append({"Run": run_id, "Taux de mort": metrics["death_rate"]})
    fig_death = px.bar(pd.DataFrame(death_data), x="Run", y="Taux de mort", color="Run", template="plotly_dark")
    fig_death.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_death, use_container_width=True)

# --- Causes de mort ---
st.subheader("Causes de mort")

death_rows = []
for run_id in selected:
    metrics = load_metrics(run_id)
    for cause, count in metrics.get("end_events", {}).items():
        if cause != "None":
            death_rows.append({"Run": run_id, "Cause": cause, "Nombre": count})

if death_rows:
    df_death = pd.DataFrame(death_rows)
    fig_causes = px.bar(
        df_death,
        x="Run",
        y="Nombre",
        color="Cause",
        barmode="group",
        template="plotly_dark",
    )
    fig_causes.update_layout(height=400)
    st.plotly_chart(fig_causes, use_container_width=True)
