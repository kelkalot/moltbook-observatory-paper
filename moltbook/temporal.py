"""Temporal analysis: activity bursts, sentiment velocity, new-agent influx."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_activity_bursts(
    posts_df: pd.DataFrame,
    window: str = "1h",
    threshold_sigma: float = 3.0,
) -> dict:
    """Detect hourly posting bursts that exceed *threshold_sigma* standard deviations.

    Returns dict with burst windows, overall stats, and per-burst detail.
    """
    if "created_at" not in posts_df.columns or posts_df.empty:
        return {"bursts": [], "total_bursts": 0, "mean_rate": 0, "std_rate": 0}

    ts = posts_df.set_index("created_at").resample(window).size()
    mean_rate = float(ts.mean())
    std_rate = float(ts.std())
    threshold = mean_rate + threshold_sigma * std_rate

    bursts = []
    for period, count in ts.items():
        if count > threshold:
            # Find top agents in this burst window
            mask = (posts_df["created_at"] >= period) & (
                posts_df["created_at"] < period + pd.Timedelta(window)
            )
            window_posts = posts_df.loc[mask]
            top_agents = window_posts["agent_name"].value_counts().head(5).to_dict()
            bursts.append({
                "start": str(period),
                "count": int(count),
                "z_score": round((count - mean_rate) / max(std_rate, 1e-9), 2),
                "top_agents": top_agents,
            })

    return {
        "bursts": sorted(bursts, key=lambda b: b["count"], reverse=True),
        "total_bursts": len(bursts),
        "mean_hourly_rate": round(mean_rate, 2),
        "std_hourly_rate": round(std_rate, 2),
        "threshold": round(threshold, 2),
    }


def compute_sentiment_velocity(posts_df: pd.DataFrame) -> dict:
    """Compute the daily rate of change in sentiment and its acceleration.

    Velocity = day-over-day change in mean sentiment.
    Acceleration = day-over-day change in velocity.
    A negative and accelerating velocity means sentiment is declining faster.
    """
    if "date" not in posts_df.columns or "avg_sentiment" not in posts_df.columns:
        return {"daily": [], "trend": "unknown"}

    daily = posts_df.groupby("date")["avg_sentiment"].mean().sort_index()
    if len(daily) < 2:
        return {"daily": [], "trend": "insufficient_data"}

    velocity = daily.diff()
    acceleration = velocity.diff()

    records = []
    for d in daily.index:
        records.append({
            "date": str(d),
            "mean_sentiment": round(float(daily[d]), 4),
            "velocity": round(float(velocity.get(d, 0)), 4) if pd.notna(velocity.get(d)) else None,
            "acceleration": round(float(acceleration.get(d, 0)), 4) if pd.notna(acceleration.get(d)) else None,
        })

    # Determine overall trend from linear regression slope
    x = np.arange(len(daily), dtype=float)
    y = daily.values.astype(float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else 0

    if slope > 0.01:
        trend = "improving"
    elif slope < -0.01:
        trend = "declining"
    else:
        trend = "stable"

    # Check if decline is accelerating
    avg_accel = float(acceleration.dropna().mean()) if len(acceleration.dropna()) > 0 else 0

    return {
        "daily": records,
        "trend": trend,
        "slope": round(slope, 5),
        "avg_velocity": round(float(velocity.dropna().mean()), 4) if len(velocity.dropna()) else 0,
        "avg_acceleration": round(avg_accel, 5),
    }


def track_agent_influx(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> dict:
    """Track new-agent appearance rate per day.

    A surge of new agents is an early warning for spam waves.
    """
    if "date" not in posts_df.columns:
        return {"daily_new_agents": [], "total_new_agents_per_day": {}}

    # Combine agents from posts and comments
    agent_dates: dict[str, str] = {}  # agent → first seen date

    for _, row in posts_df.iterrows():
        agent = row.get("agent_name")
        d = row.get("date")
        if pd.notna(agent) and pd.notna(d):
            d_str = str(d)
            if agent not in agent_dates or d_str < agent_dates[agent]:
                agent_dates[agent] = d_str

    if "date" in comments_df.columns:
        for _, row in comments_df.iterrows():
            agent = row.get("agent_name")
            d = row.get("date")
            if pd.notna(agent) and pd.notna(d):
                d_str = str(d)
                if agent not in agent_dates or d_str < agent_dates[agent]:
                    agent_dates[agent] = d_str

    # Count new agents per day
    from collections import Counter
    new_per_day = Counter(agent_dates.values())
    sorted_days = sorted(new_per_day.keys())

    cumulative = 0
    records = []
    for d in sorted_days:
        cumulative += new_per_day[d]
        records.append({
            "date": d,
            "new_agents": new_per_day[d],
            "cumulative_agents": cumulative,
        })

    # Detect influx spike (day with >2σ above mean new agents)
    if records:
        counts = [r["new_agents"] for r in records]
        mean_new = float(np.mean(counts))
        std_new = float(np.std(counts)) if len(counts) > 1 else 0
        for r in records:
            r["is_spike"] = r["new_agents"] > mean_new + 2 * std_new if std_new > 0 else False
    else:
        mean_new = 0
        std_new = 0

    return {
        "daily_new_agents": records,
        "mean_daily_new": round(mean_new, 1),
        "std_daily_new": round(std_new, 1),
        "spike_days": [r["date"] for r in records if r.get("is_spike")],
    }


def run_temporal_analysis(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> dict:
    """Run all temporal analyses and return combined results."""
    return {
        "bursts": detect_activity_bursts(posts_df),
        "sentiment_velocity": compute_sentiment_velocity(posts_df),
        "agent_influx": track_agent_influx(posts_df, comments_df),
    }
