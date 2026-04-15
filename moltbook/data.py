"""Data loading from HuggingFace or local CSV files."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

HUGGINGFACE_DATASET = "SimulaMet/moltbook-observatory-archive"


def load_from_huggingface(dataset_name: str = HUGGINGFACE_DATASET) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load posts and comments from HuggingFace.

    Returns (posts_df, comments_df).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("'datasets' library not installed. Run: pip install datasets")
        sys.exit(1)

    logger.info("Loading dataset from HuggingFace: %s", dataset_name)

    posts_ds = load_dataset(dataset_name, "posts")
    split = "archive" if "archive" in posts_ds else list(posts_ds.keys())[0]
    posts_df = posts_ds[split].to_pandas()
    logger.info("Loaded %s posts", f"{len(posts_df):,}")

    comments_ds = load_dataset(dataset_name, "comments")
    split = "archive" if "archive" in comments_ds else list(comments_ds.keys())[0]
    comments_df = comments_ds[split].to_pandas()
    logger.info("Loaded %s comments", f"{len(comments_df):,}")

    return _prepare(posts_df, comments_df)


def load_from_csv(posts_path: str, comments_path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load posts and comments from local CSV files."""
    posts_df = pd.read_csv(posts_path)
    comments_df = pd.read_csv(comments_path) if comments_path and Path(comments_path).exists() else pd.DataFrame()
    logger.info("Loaded %s posts and %s comments from CSV", f"{len(posts_df):,}", f"{len(comments_df):,}")
    return _prepare(posts_df, comments_df)


def _prepare(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shared preprocessing for both data sources."""
    if "created_at" in posts_df.columns:
        posts_df["created_at"] = pd.to_datetime(posts_df["created_at"], errors="coerce")
        posts_df["date"] = posts_df["created_at"].dt.date
        posts_df["hour"] = posts_df["created_at"].dt.hour

    if "created_at" in comments_df.columns:
        comments_df["created_at"] = pd.to_datetime(comments_df["created_at"], errors="coerce")
        comments_df["date"] = comments_df["created_at"].dt.date
        comments_df["hour"] = comments_df["created_at"].dt.hour

    posts_df["content_length"] = posts_df["content"].fillna("").str.len()
    if "content" in comments_df.columns:
        comments_df["content_length"] = comments_df["content"].fillna("").str.len()

    return posts_df, comments_df


def filter_by_date(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter both DataFrames to rows within [date_from, date_to] (inclusive).

    Dates are strings like '2026-01-29'. Either bound can be None (open-ended).
    """
    from datetime import date as date_type

    def _parse(d: str) -> date_type:
        return pd.Timestamp(d).date()

    if date_from is not None:
        d = _parse(date_from)
        if "date" in posts_df.columns:
            posts_df = posts_df[posts_df["date"] >= d]
        if "date" in comments_df.columns:
            comments_df = comments_df[comments_df["date"] >= d]

    if date_to is not None:
        d = _parse(date_to)
        if "date" in posts_df.columns:
            posts_df = posts_df[posts_df["date"] <= d]
        if "date" in comments_df.columns:
            comments_df = comments_df[comments_df["date"] <= d]

    logger.info(
        "After date filter: %s posts, %s comments",
        f"{len(posts_df):,}",
        f"{len(comments_df):,}",
    )
    return posts_df, comments_df
