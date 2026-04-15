"""Composite per-agent risk scoring.

Combines injection patterns, spam behaviour, content entropy, posting frequency,
account-age-vs-activity ratio, and engagement anomalies into a single 0–100 score.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
import pandas as pd

from .risk import INJECTION_PATTERNS, CRYPTO_PATTERNS, MANIPULATION_PATTERNS, _detect_patterns


# ---------------------------------------------------------------------------
# Per-agent feature extraction
# ---------------------------------------------------------------------------

def _content_entropy(texts: pd.Series) -> float:
    """Shannon entropy of word distribution — low entropy ≈ repetitive content."""
    words = " ".join(texts.dropna().astype(str)).lower().split()
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def build_agent_profiles(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a DataFrame with one row per agent and all risk-relevant features."""

    # --- Post-level features per agent ---
    agg = posts_df.groupby("agent_name").agg(
        post_count=("id", "count"),
        avg_score=("score", "mean"),
        max_score=("score", "max"),
        avg_comments=("comment_count", "mean"),
        avg_length=("content_length", "mean"),
        length_std=("content_length", "std"),
        unique_submolts=("submolt", "nunique"),
    ).reset_index()
    agg["length_std"] = agg["length_std"].fillna(0)

    # --- Injection hit rate ---
    inj_counts: dict[str, int] = {}
    crypto_counts: dict[str, int] = {}
    for _, row in posts_df.iterrows():
        agent = row.get("agent_name", "unknown")
        text = f"{row.get('content', '')} {row.get('title', '')}"
        if _detect_patterns(text, INJECTION_PATTERNS):
            inj_counts[agent] = inj_counts.get(agent, 0) + 1
        if _detect_patterns(text.lower(), CRYPTO_PATTERNS):
            crypto_counts[agent] = crypto_counts.get(agent, 0) + 1

    agg["injection_posts"] = agg["agent_name"].map(inj_counts).fillna(0).astype(int)
    agg["crypto_posts"] = agg["agent_name"].map(crypto_counts).fillna(0).astype(int)
    agg["injection_rate"] = agg["injection_posts"] / agg["post_count"].clip(lower=1)
    agg["crypto_rate"] = agg["crypto_posts"] / agg["post_count"].clip(lower=1)

    # --- Duplicate rate ---
    dup = posts_df.groupby("agent_name").apply(
        lambda g: g.duplicated(subset=["content"], keep=False).mean()
    ).rename("duplicate_rate")
    agg = agg.merge(dup, on="agent_name", how="left")
    agg["duplicate_rate"] = agg["duplicate_rate"].fillna(0)

    # --- Content entropy (low = more repetitive) ---
    entropy = posts_df.groupby("agent_name")["content"].apply(_content_entropy).rename("content_entropy")
    agg = agg.merge(entropy, on="agent_name", how="left")
    agg["content_entropy"] = agg["content_entropy"].fillna(0)

    # --- Self-interaction rate (comments on own posts) ---
    if len(comments_df) > 0 and "post_id" in comments_df.columns:
        post_authors = dict(zip(posts_df["id"], posts_df["agent_name"]))
        c = comments_df.copy()
        c["post_author"] = c["post_id"].map(post_authors)
        self_comments = c[c["agent_name"] == c["post_author"]].groupby("agent_name").size().rename("self_comments")
        total_comments = c.groupby("agent_name").size().rename("total_comments_made")
        agg = agg.merge(self_comments, on="agent_name", how="left")
        agg = agg.merge(total_comments, on="agent_name", how="left")
        agg["self_comments"] = agg["self_comments"].fillna(0)
        agg["total_comments_made"] = agg["total_comments_made"].fillna(0)
        agg["self_interaction_rate"] = agg["self_comments"] / agg["total_comments_made"].clip(lower=1)
    else:
        agg["self_comments"] = 0
        agg["total_comments_made"] = 0
        agg["self_interaction_rate"] = 0.0

    # --- Manipulation in comments ---
    if len(comments_df) > 0:
        manip_counts: dict[str, int] = {}
        for _, row in comments_df.iterrows():
            if len(_detect_patterns(str(row.get("content", "")), MANIPULATION_PATTERNS)) >= 2:
                a = row.get("agent_name", "unknown")
                manip_counts[a] = manip_counts.get(a, 0) + 1
        agg["manipulation_comments"] = agg["agent_name"].map(manip_counts).fillna(0).astype(int)
    else:
        agg["manipulation_comments"] = 0

    # --- Activity span (days active) ---
    if "date" in posts_df.columns:
        span = posts_df.groupby("agent_name")["date"].agg(["min", "max"])
        span["days_active"] = (pd.to_datetime(span["max"]) - pd.to_datetime(span["min"])).dt.days + 1
        span = span[["days_active"]]
        agg = agg.merge(span, on="agent_name", how="left")
        agg["days_active"] = agg["days_active"].fillna(1)
        agg["posts_per_day"] = agg["post_count"] / agg["days_active"].clip(lower=1)
    else:
        agg["days_active"] = 1
        agg["posts_per_day"] = agg["post_count"]

    return agg


# ---------------------------------------------------------------------------
# Composite risk score (0–100)
# ---------------------------------------------------------------------------

_WEIGHTS = {
    "injection_rate": 25,     # most dangerous single signal
    "duplicate_rate": 15,     # strong spam indicator
    "crypto_rate": 12,        # financial risk
    "manipulation_comments": 10,
    "self_interaction_rate": 8,
    "posts_per_day_z": 10,    # abnormal posting frequency
    "low_entropy": 10,        # repetitive content
    "single_submolt": 10,     # posting in only 1 community (tunnel-vision spam)
}


def compute_risk_scores(profiles: pd.DataFrame) -> pd.DataFrame:
    """Add a ``risk_score`` column (0–100) and ``risk_tier`` to *profiles*."""
    df = profiles.copy()

    # Normalise posting frequency to z-score, clip at 0 (only penalise high)
    if df["posts_per_day"].std() > 0:
        df["posts_per_day_z"] = ((df["posts_per_day"] - df["posts_per_day"].mean()) / df["posts_per_day"].std()).clip(lower=0)
    else:
        df["posts_per_day_z"] = 0

    # Normalise entropy: invert so low entropy → high score
    if df["content_entropy"].max() > 0:
        df["low_entropy"] = 1 - (df["content_entropy"] / df["content_entropy"].max())
    else:
        df["low_entropy"] = 0

    # Single-submolt flag (normalised 0 or 1)
    df["single_submolt"] = (df["unique_submolts"] <= 1).astype(float)

    # Clamp manipulation to 0-1 via sigmoid-like
    df["manipulation_norm"] = df["manipulation_comments"].apply(lambda x: min(x / 5, 1.0))

    # Weighted sum
    score = (
        df["injection_rate"].clip(0, 1) * _WEIGHTS["injection_rate"]
        + df["duplicate_rate"].clip(0, 1) * _WEIGHTS["duplicate_rate"]
        + df["crypto_rate"].clip(0, 1) * _WEIGHTS["crypto_rate"]
        + df["manipulation_norm"] * _WEIGHTS["manipulation_comments"]
        + df["self_interaction_rate"].clip(0, 1) * _WEIGHTS["self_interaction_rate"]
        + df["posts_per_day_z"].clip(0, 3) / 3 * _WEIGHTS["posts_per_day_z"]
        + df["low_entropy"].clip(0, 1) * _WEIGHTS["low_entropy"]
        + df["single_submolt"] * _WEIGHTS["single_submolt"]
    )
    df["risk_score"] = score.round(1)

    # Tier
    df["risk_tier"] = pd.cut(
        df["risk_score"],
        bins=[-1, 15, 35, 60, 101],
        labels=["low", "moderate", "high", "critical"],
    )

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_agent_scoring(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    min_posts: int = 2,
) -> dict:
    """Build agent profiles, score them, and return summary + top risky agents.

    Returns dict with:
      - tier_counts: {low: N, moderate: N, high: N, critical: N}
      - top_risky: list of dicts for top 20 riskiest agents
      - score_stats: {mean, median, std, max}
    """
    profiles = build_agent_profiles(posts_df, comments_df)
    profiles = profiles[profiles["post_count"] >= min_posts]

    if profiles.empty:
        return {"tier_counts": {}, "top_risky": [], "score_stats": {}}

    scored = compute_risk_scores(profiles)

    tier_counts = scored["risk_tier"].value_counts().to_dict()
    top = scored.nlargest(20, "risk_score")

    top_risky = []
    for _, row in top.iterrows():
        # Build a list of the main contributing factors
        reasons = []
        if row["injection_rate"] > 0:
            reasons.append(f"injection ({row['injection_rate']:.0%})")
        if row["duplicate_rate"] > 0.3:
            reasons.append(f"duplicates ({row['duplicate_rate']:.0%})")
        if row["crypto_rate"] > 0.3:
            reasons.append(f"crypto ({row['crypto_rate']:.0%})")
        if row["manipulation_comments"] > 0:
            reasons.append(f"manipulation ({int(row['manipulation_comments'])})")
        if row["self_interaction_rate"] > 0.5:
            reasons.append("self-interaction")
        if not reasons:
            reasons.append("multi-signal")

        top_risky.append({
            "agent": row["agent_name"],
            "risk_score": float(row["risk_score"]),
            "risk_tier": str(row["risk_tier"]),
            "post_count": int(row["post_count"]),
            "reasons": reasons,
        })

    return {
        "tier_counts": {str(k): int(v) for k, v in tier_counts.items()},
        "top_risky": top_risky,
        "score_stats": {
            "mean": round(float(scored["risk_score"].mean()), 1),
            "median": round(float(scored["risk_score"].median()), 1),
            "std": round(float(scored["risk_score"].std()), 1),
            "max": round(float(scored["risk_score"].max()), 1),
        },
    }
