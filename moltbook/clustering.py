"""Agent clustering and topic modeling."""

from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler


def run_agent_clustering(posts_df: pd.DataFrame, n_clusters: int = 5, min_posts: int = 3) -> tuple[pd.DataFrame, dict]:
    """Cluster agents by behavioural features.

    Returns (agent_features_df, cluster_summary_dict).
    """
    features = posts_df.groupby("agent_name").agg({
        "id": "count",
        "score": ["mean", "max", "std"],
        "comment_count": ["mean", "max"],
        "content_length": ["mean", "std"],
        "hour": lambda x: x.mode().iloc[0] if len(x.mode()) else 12,
        "avg_sentiment": "mean",
    }).reset_index()

    features.columns = [
        "agent_name", "post_count", "avg_score", "max_score", "score_std",
        "avg_comments", "max_comments", "avg_length", "length_std",
        "peak_hour", "avg_sentiment",
    ]
    features = features[features["post_count"] >= min_posts].fillna(0)

    feat_cols = ["post_count", "avg_score", "max_score", "avg_comments", "avg_length"]
    X = StandardScaler().fit_transform(features[feat_cols].values)
    features["cluster"] = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(X)

    summary: dict = {}
    for i in range(n_clusters):
        c = features[features["cluster"] == i]
        summary[i] = {
            "size": len(c),
            "avg_posts": round(float(c["post_count"].mean()), 2),
            "avg_score": round(float(c["avg_score"].mean()), 2),
            "avg_sentiment": round(float(c["avg_sentiment"].mean()), 4),
            "example_agents": c.nlargest(5, "post_count")["agent_name"].tolist(),
        }

    return features, summary


def run_topic_modeling(posts_df: pd.DataFrame, n_topics: int = 8, max_features: int = 500) -> dict:
    """Simple TF-IDF + KMeans topic discovery.

    Returns dict mapping topic_id → {keywords, count, percentage}.
    """
    long_posts = posts_df[posts_df["content"].str.len() > 100]
    sample = long_posts.sample(min(3000, len(long_posts)), random_state=42)

    vec = TfidfVectorizer(max_features=max_features, stop_words="english", min_df=5, max_df=0.7)
    tfidf = vec.fit_transform(sample["content"].fillna(""))
    labels = KMeans(n_clusters=n_topics, random_state=42, n_init=10).fit_predict(tfidf)

    names = vec.get_feature_names_out()
    km = KMeans(n_clusters=n_topics, random_state=42, n_init=10).fit(tfidf)

    topics: dict = {}
    for i in range(n_topics):
        top_idx = km.cluster_centers_[i].argsort()[-10:][::-1]
        count = int((labels == i).sum())
        topics[i] = {
            "keywords": [names[j] for j in top_idx],
            "count": count,
            "percentage": round(count / len(sample) * 100, 2),
        }
    return topics
