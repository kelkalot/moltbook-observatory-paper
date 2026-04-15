"""Sentiment analysis using VADER and TextBlob."""

from __future__ import annotations

import re

import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()


def _clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"<s>.*?</s>", "", text, flags=re.DOTALL)
    text = re.sub(r"POST /api.*?\n", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^\w\s.,!?;:]", " ", text)
    return text


def _textblob_score(text: str) -> float:
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0


def _vader_score(text: str) -> float:
    try:
        return _vader.polarity_scores(str(text))["compound"]
    except Exception:
        return 0.0


def _categorize(score: float) -> str:
    if score > 0.1:
        return "Positive"
    if score < -0.1:
        return "Negative"
    return "Neutral"


def run_sentiment_analysis(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
) -> dict:
    """Add sentiment columns to *posts_df* and *comments_df* (in-place) and return summary stats."""
    for df in (posts_df, comments_df):
        if "content" not in df.columns:
            continue
        df["clean_content"] = df["content"].apply(_clean_text)
        df["textblob_sentiment"] = df["clean_content"].apply(_textblob_score)
        df["vader_sentiment"] = df["clean_content"].apply(_vader_score)
        df["avg_sentiment"] = (df["textblob_sentiment"] + df["vader_sentiment"]) / 2
        df["sentiment_category"] = df["avg_sentiment"].apply(_categorize)

    distribution = posts_df["sentiment_category"].value_counts()
    daily = posts_df.groupby("date")["avg_sentiment"].agg(["mean", "std", "count"]) if "date" in posts_df.columns else pd.DataFrame()

    return {
        "overall_sentiment": float(posts_df["avg_sentiment"].mean()),
        "sentiment_distribution": distribution.to_dict(),
        "daily_sentiment": {
            str(k): round(float(v), 4)
            for k, v in (daily["mean"].to_dict().items() if len(daily) else {}.items())
        },
    }
