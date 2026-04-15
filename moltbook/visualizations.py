"""Chart generation for the HTML report."""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_palette("husl")
plt.style.use("seaborn-v0_8-whitegrid")

_SENTIMENT_COLORS = {"Positive": "#27AE60", "Neutral": "#F39C12", "Negative": "#E74C3C"}


def _fig_to_base64(fig: plt.Figure) -> str:
    """Return a base64-encoded PNG string (for inline HTML embedding)."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def generate_dashboard_charts(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> dict[str, str]:
    """Return a dict of chart_name → base64-encoded PNG strings."""
    charts: dict[str, str] = {}

    # 1. Sentiment distribution pie
    fig, ax = plt.subplots(figsize=(5, 4))
    if "sentiment_category" in posts_df.columns:
        counts = posts_df["sentiment_category"].value_counts()
        ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=[_SENTIMENT_COLORS.get(c, "#999") for c in counts.index],
        )
        ax.set_title("Sentiment Distribution", fontweight="bold")
    charts["sentiment_pie"] = _fig_to_base64(fig)

    # 2. Daily post volume
    if "date" in posts_df.columns:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        daily = posts_df["date"].value_counts().sort_index()
        ax.plot(daily.index, daily.values, "o-", linewidth=2, markersize=5)
        ax.set_title("Posts per Day", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        charts["daily_volume"] = _fig_to_base64(fig)

    # 3. Daily sentiment trend
    if "date" in posts_df.columns and "avg_sentiment" in posts_df.columns:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ds = posts_df.groupby("date")["avg_sentiment"].mean()
        ax.plot(ds.index, ds.values, "o-", linewidth=2, color="#3498DB")
        ax.set_title("Daily Sentiment Trend", fontweight="bold")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        charts["sentiment_trend"] = _fig_to_base64(fig)

    # 4. Top 10 communities
    if "submolt" in posts_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        top = posts_df["submolt"].value_counts().head(10)
        ax.barh(range(len(top)), top.values, color="#4ECDC4")
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index, fontsize=9)
        ax.invert_yaxis()
        ax.set_title("Top 10 Communities", fontweight="bold")
        plt.tight_layout()
        charts["top_communities"] = _fig_to_base64(fig)

    # 5. Hourly activity
    if "hour" in posts_df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        hourly = posts_df["hour"].value_counts().sort_index()
        ax.bar(hourly.index, hourly.values, alpha=0.7)
        ax.set_title("Posts by Hour (UTC)", fontweight="bold")
        ax.set_xlabel("Hour")
        plt.tight_layout()
        charts["hourly_activity"] = _fig_to_base64(fig)

    # 6. Top 10 posters
    fig, ax = plt.subplots(figsize=(6, 4))
    top_p = posts_df["agent_name"].value_counts().head(10)
    ax.barh(range(len(top_p)), top_p.values, color="#E74C3C")
    ax.set_yticks(range(len(top_p)))
    ax.set_yticklabels(top_p.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Top 10 Posters", fontweight="bold")
    plt.tight_layout()
    charts["top_posters"] = _fig_to_base64(fig)

    # 7. Score distribution
    if "score" in posts_df.columns:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        scores = posts_df[posts_df["score"] > 0]["score"]
        ax.hist(scores, bins=50, alpha=0.7, color="#9B59B6")
        ax.set_yscale("log")
        ax.set_title("Score Distribution (log)", fontweight="bold")
        plt.tight_layout()
        charts["score_dist"] = _fig_to_base64(fig)

    return charts


def generate_temporal_charts(temporal: dict) -> dict[str, str]:
    """Charts for temporal analysis: sentiment velocity, agent influx, bursts."""
    charts: dict[str, str] = {}

    # Sentiment velocity + acceleration
    sv = temporal.get("sentiment_velocity", {})
    daily = sv.get("daily", [])
    if len(daily) >= 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        dates = [r["date"] for r in daily]
        sentiments = [r["mean_sentiment"] for r in daily]
        velocities = [r["velocity"] if r["velocity"] is not None else 0 for r in daily]

        ax1.plot(dates, sentiments, "o-", linewidth=2, color="#3498DB")
        ax1.set_title("Sentiment Over Time", fontweight="bold", fontsize=10)
        ax1.set_ylabel("Mean Sentiment")

        colors_v = ["#E74C3C" if v < 0 else "#27AE60" for v in velocities]
        ax2.bar(dates, velocities, color=colors_v, alpha=0.8)
        ax2.axhline(0, color="#333", linewidth=0.5)
        ax2.set_title("Sentiment Velocity (daily Δ)", fontweight="bold", fontsize=10)
        ax2.set_ylabel("Change")
        ax2.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        charts["sentiment_velocity"] = _fig_to_base64(fig)

    # New agent influx
    influx = temporal.get("agent_influx", {})
    daily_new = influx.get("daily_new_agents", [])
    if daily_new:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        dates = [r["date"] for r in daily_new]
        counts = [r["new_agents"] for r in daily_new]
        colors_i = ["#E74C3C" if r.get("is_spike") else "#4ECDC4" for r in daily_new]
        ax.bar(dates, counts, color=colors_i, alpha=0.8)
        ax.set_title("New Agents per Day", fontweight="bold")
        ax.set_ylabel("New Agents")
        ax.tick_params(axis="x", rotation=45)
        # Mark spikes
        spike_days = influx.get("spike_days", [])
        if spike_days:
            ax.legend(["Red = spike (>2σ)"], loc="upper right", fontsize=8)
        plt.tight_layout()
        charts["agent_influx"] = _fig_to_base64(fig)

    return charts


def generate_agent_score_charts(agent_scores: dict) -> dict[str, str]:
    """Charts for agent risk scoring: tier distribution, top risky agents."""
    charts: dict[str, str] = {}

    # Tier distribution pie
    tiers = agent_scores.get("tier_counts", {})
    if tiers:
        fig, ax = plt.subplots(figsize=(5, 4))
        tier_colors = {"low": "#4CAF50", "moderate": "#FBC02D", "high": "#F57C00", "critical": "#D32F2F"}
        labels = list(tiers.keys())
        values = list(tiers.values())
        colors = [tier_colors.get(l, "#999") for l in labels]
        ax.pie(values, labels=[f"{l.title()} ({v})" for l, v in zip(labels, values)],
               colors=colors, autopct="%1.1f%%", startangle=90)
        ax.set_title("Agent Risk Tiers", fontweight="bold")
        charts["risk_tiers"] = _fig_to_base64(fig)

    # Top risky agents bar chart
    top = agent_scores.get("top_risky", [])
    if top:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        agents = [r["agent"][:20] for r in top[:15]]
        scores = [r["risk_score"] for r in top[:15]]
        tier_colors_map = {"low": "#4CAF50", "moderate": "#FBC02D", "high": "#F57C00", "critical": "#D32F2F"}
        colors = [tier_colors_map.get(r.get("risk_tier", ""), "#999") for r in top[:15]]
        ax.barh(range(len(agents)), scores, color=colors)
        ax.set_yticks(range(len(agents)))
        ax.set_yticklabels(agents, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Risk Score (0–100)")
        ax.set_title("Top Risky Agents", fontweight="bold")
        plt.tight_layout()
        charts["top_risky_agents"] = _fig_to_base64(fig)

    return charts


def generate_engagement_charts(engagement: dict) -> dict[str, str]:
    """Chart for engagement quality breakdown."""
    charts: dict[str, str] = {}

    si = engagement.get("self_interaction", {})
    be = engagement.get("bot_engagement", {})
    organic = engagement.get("organic_ratio", 0)

    total = si.get("total_comments", 0)
    if total > 0:
        fig, ax = plt.subplots(figsize=(5, 4))
        self_c = si.get("self_comments", 0)
        bot_c = be.get("bot_comments", 0)
        organic_c = max(total - self_c - bot_c, 0)
        ax.pie(
            [organic_c, self_c, bot_c],
            labels=[f"Organic ({organic_c:,})", f"Self ({self_c:,})", f"Bot ({bot_c:,})"],
            colors=["#27AE60", "#F39C12", "#E74C3C"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Engagement Quality", fontweight="bold")
        charts["engagement_quality"] = _fig_to_base64(fig)

    return charts
