"""Engagement quality analysis.

Separates organic engagement from self-interaction and bot-driven patterns.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def analyse_engagement_quality(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
) -> dict:
    """Break down engagement into organic vs suspicious components.

    Returns dict with:
      - self_interaction: stats on agents commenting on their own posts
      - bot_engagement: posts where all/most comments come from known duplicate commenters
      - organic_ratio: fraction of engagement estimated as organic
      - top_self_interactors: agents ranked by self-comment count
      - engagement_anomalies: posts with abnormally high engagement relative to their content
    """
    result: dict = {}

    # --- Self-interaction analysis ---
    if len(comments_df) > 0 and "post_id" in comments_df.columns:
        post_authors = dict(zip(posts_df["id"], posts_df["agent_name"]))
        c = comments_df.copy()
        c["post_author"] = c["post_id"].map(post_authors)

        self_mask = c["agent_name"] == c["post_author"]
        self_comments = int(self_mask.sum())
        total_comments = len(c)
        self_rate = self_comments / max(total_comments, 1)

        top_self = (
            c[self_mask]
            .groupby("agent_name")
            .size()
            .sort_values(ascending=False)
            .head(10)
            .to_dict()
        )

        result["self_interaction"] = {
            "self_comments": self_comments,
            "total_comments": total_comments,
            "self_rate": round(self_rate, 4),
            "top_self_interactors": top_self,
        }
    else:
        result["self_interaction"] = {
            "self_comments": 0, "total_comments": 0,
            "self_rate": 0.0, "top_self_interactors": {},
        }

    # --- Bot-driven engagement (duplicate comment content) ---
    if len(comments_df) > 0 and "content" in comments_df.columns:
        cc = comments_df.groupby("content").size().reset_index(name="count")
        bot_content = cc[cc["count"] > 3]  # same comment text >3 times
        bot_comment_ids = set(
            comments_df[comments_df["content"].isin(bot_content["content"])].index
        )
        bot_comment_count = len(bot_comment_ids)
        bot_rate = bot_comment_count / max(len(comments_df), 1)

        result["bot_engagement"] = {
            "bot_comments": bot_comment_count,
            "bot_rate": round(bot_rate, 4),
            "unique_bot_phrases": len(bot_content),
        }
    else:
        result["bot_engagement"] = {
            "bot_comments": 0, "bot_rate": 0.0, "unique_bot_phrases": 0,
        }

    # --- Organic ratio estimate ---
    total_c = result["self_interaction"]["total_comments"]
    suspicious = result["self_interaction"]["self_comments"] + result["bot_engagement"]["bot_comments"]
    organic = max(total_c - suspicious, 0)
    result["organic_ratio"] = round(organic / max(total_c, 1), 4)

    # --- Engagement anomalies (high score/comments relative to content length) ---
    if "score" in posts_df.columns and "content_length" in posts_df.columns:
        df = posts_df[posts_df["content_length"] > 10].copy()
        if len(df) > 50:
            df["engagement_density"] = (df["score"] + df.get("comment_count", 0)) / df["content_length"]
            mean_ed = df["engagement_density"].mean()
            std_ed = df["engagement_density"].std()
            if std_ed > 0:
                df["ed_zscore"] = (df["engagement_density"] - mean_ed) / std_ed
                anomalies = df[df["ed_zscore"] > 3].nlargest(10, "ed_zscore")
                result["engagement_anomalies"] = [
                    {
                        "agent": row.get("agent_name", "unknown"),
                        "title": str(row.get("title", ""))[:60],
                        "score": int(row.get("score", 0)),
                        "content_length": int(row["content_length"]),
                        "z_score": round(float(row["ed_zscore"]), 2),
                    }
                    for _, row in anomalies.iterrows()
                ]
            else:
                result["engagement_anomalies"] = []
        else:
            result["engagement_anomalies"] = []
    else:
        result["engagement_anomalies"] = []

    return result
