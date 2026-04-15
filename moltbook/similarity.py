"""Fuzzy content similarity detection using shingling + Jaccard estimation.

Catches slightly rephrased spam that exact-duplicate detection misses.
Uses a lightweight pure-Python approach (no extra dependencies) that
approximates MinHash with fixed random hash functions.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Shingling helpers
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip URLs, collapse whitespace."""
    text = re.sub(r"https?://\S+", "", str(text).lower())
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _shingles(text: str, k: int = 3) -> set[str]:
    """Return the set of word k-grams (shingles)."""
    words = text.split()
    if len(words) < k:
        return {text}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


# ---------------------------------------------------------------------------
# MinHash signature (lightweight, no external lib)
# ---------------------------------------------------------------------------

_NUM_HASHES = 64


def _hash_fn(shingle: str, seed: int) -> int:
    """Deterministic hash for a shingle with a given seed."""
    return int(hashlib.md5(f"{seed}:{shingle}".encode()).hexdigest()[:8], 16)


def _minhash_signature(shingle_set: set[str]) -> list[int]:
    """Compute a MinHash signature of length _NUM_HASHES."""
    sig = []
    for seed in range(_NUM_HASHES):
        min_h = min((_hash_fn(s, seed) for s in shingle_set), default=0)
        sig.append(min_h)
    return sig


def _jaccard_est(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)


# ---------------------------------------------------------------------------
# Cluster near-duplicates via LSH bands
# ---------------------------------------------------------------------------

def _lsh_buckets(sig: list[int], bands: int = 16) -> list[tuple]:
    """Split signature into *bands* bands and return bucket keys."""
    rows_per_band = len(sig) // bands
    buckets = []
    for b in range(bands):
        chunk = tuple(sig[b * rows_per_band : (b + 1) * rows_per_band])
        buckets.append((b, chunk))
    return buckets


def find_near_duplicates(
    posts_df: pd.DataFrame,
    similarity_threshold: float = 0.5,
    min_content_length: int = 80,
    sample_size: int = 5000,
) -> dict:
    """Detect clusters of near-duplicate posts.

    Returns dict with:
      - clusters: list of {size, agents, sample_title, similarity}
      - near_duplicate_posts: total posts involved
      - unique_clusters: number of distinct clusters
    """
    # Filter short posts and sample for performance
    valid = posts_df[posts_df["content_length"] >= min_content_length].copy()
    if len(valid) > sample_size:
        valid = valid.sample(sample_size, random_state=42)

    if valid.empty:
        return {"clusters": [], "near_duplicate_posts": 0, "unique_clusters": 0}

    # Build shingle sets and MinHash signatures
    sigs: dict[int, list[int]] = {}
    shingle_sets: dict[int, set[str]] = {}

    for idx, row in valid.iterrows():
        norm = _normalise(str(row.get("content", "")))
        sh = _shingles(norm)
        if sh:
            shingle_sets[idx] = sh
            sigs[idx] = _minhash_signature(sh)

    # LSH: group candidates by band buckets
    bucket_map: dict[tuple, list[int]] = defaultdict(list)
    for idx, sig in sigs.items():
        for bucket_key in _lsh_buckets(sig):
            bucket_map[bucket_key].append(idx)

    # Find candidate pairs from shared buckets
    candidate_pairs: set[tuple[int, int]] = set()
    for members in bucket_map.values():
        if len(members) > 1 and len(members) < 100:  # skip overly large buckets
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    candidate_pairs.add((members[i], members[j]))

    # Verify candidates with full Jaccard estimation
    from collections import defaultdict as _dd
    adjacency: dict[int, set[int]] = _dd(set)

    for a, b in candidate_pairs:
        if a in sigs and b in sigs:
            sim = _jaccard_est(sigs[a], sigs[b])
            if sim >= similarity_threshold:
                adjacency[a].add(b)
                adjacency[b].add(a)

    # Connected-component clustering
    visited: set[int] = set()
    clusters: list[set[int]] = []

    for node in adjacency:
        if node in visited:
            continue
        cluster: set[int] = set()
        stack = [node]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            cluster.add(n)
            stack.extend(adjacency[n] - visited)
        if len(cluster) >= 2:
            clusters.append(cluster)

    # Build summary
    cluster_summaries = []
    total_involved = 0
    for cl in sorted(clusters, key=len, reverse=True)[:30]:
        cl_rows = valid.loc[list(cl)]
        agents = cl_rows["agent_name"].value_counts().head(5).to_dict()
        sample_title = str(cl_rows.iloc[0].get("title", ""))[:80]
        cluster_summaries.append({
            "size": len(cl),
            "agents": agents,
            "sample_title": sample_title,
        })
        total_involved += len(cl)

    return {
        "clusters": cluster_summaries,
        "near_duplicate_posts": total_involved,
        "unique_clusters": len(clusters),
        "sample_size_analysed": len(valid),
    }
