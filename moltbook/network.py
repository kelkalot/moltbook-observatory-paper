"""Agent interaction network analysis.

Includes community detection (greedy modularity as Louvain-equivalent),
PageRank for influence ranking, and reciprocity analysis for collusion signals.
"""

from __future__ import annotations

import pandas as pd
import networkx as nx
from collections import Counter


def run_network_analysis(
    posts_df: pd.DataFrame,
    comments_df: pd.DataFrame,
    min_interactions: int = 2,
) -> dict:
    """Build a directed interaction graph and return comprehensive network stats.

    Returns dict with:
      - nodes, edges, density
      - top_commenters, most_commented (degree-based)
      - pagerank: top agents by PageRank score
      - communities: detected groups with member counts
      - reciprocity: overall rate + top reciprocal pairs (collusion signal)
    """
    empty = {
        "nodes": 0, "edges": 0, "density": 0,
        "top_commenters": [], "most_commented": [],
        "pagerank": [], "communities": [], "reciprocity": {},
    }
    if len(comments_df) == 0:
        return empty

    author_map = dict(zip(posts_df["id"], posts_df["agent_name"]))
    c = comments_df.copy()
    c["post_author"] = c["post_id"].map(author_map)

    interactions = (
        c.groupby(["agent_name", "post_author"])
        .size()
        .reset_index(name="weight")
    )
    interactions = interactions[
        (interactions["agent_name"] != interactions["post_author"])
        & (interactions["weight"] >= min_interactions)
    ]

    G = nx.DiGraph()
    for _, row in interactions.iterrows():
        G.add_edge(row["agent_name"], row["post_author"], weight=row["weight"])

    if G.number_of_nodes() == 0:
        return empty

    in_deg = dict(G.in_degree(weight="weight"))
    out_deg = dict(G.out_degree(weight="weight"))

    # ----- PageRank -----
    pr = {}
    try:
        pr = nx.pagerank(G, weight="weight", max_iter=100)
        top_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:15]
        pagerank_list = [
            {"agent": a, "score": round(s, 6)} for a, s in top_pr
        ]
    except Exception:
        pagerank_list = []

    # ----- Community detection (greedy modularity on undirected projection) -----
    communities_list = []
    try:
        G_undir = G.to_undirected()
        from networkx.algorithms.community import greedy_modularity_communities
        communities_raw = greedy_modularity_communities(G_undir, weight="weight")
        for i, comm in enumerate(sorted(communities_raw, key=len, reverse=True)[:15]):
            members = list(comm)
            if pr:
                rep = max(members, key=lambda a: pr.get(a, 0))
            else:
                rep = members[0]
            communities_list.append({
                "id": i,
                "size": len(members),
                "representative": rep,
                "top_members": sorted(members, key=lambda a: pr.get(a, 0) if pr else 0, reverse=True)[:5],
            })
    except Exception:
        pass

    # ----- Reciprocity analysis -----
    reciprocal_pairs = []
    reciprocal_weight_total = 0
    total_weight = sum(d["weight"] for _, _, d in G.edges(data=True))

    for u, v, d in G.edges(data=True):
        if G.has_edge(v, u):
            w_uv = d["weight"]
            w_vu = G[v][u]["weight"]
            reciprocal_weight_total += w_uv
            reciprocal_pairs.append({
                "agent_a": u,
                "agent_b": v,
                "a_to_b": w_uv,
                "b_to_a": w_vu,
                "total": w_uv + w_vu,
            })

    # Deduplicate pairs (A<->B and B<->A)
    seen = set()
    unique_pairs = []
    for p in sorted(reciprocal_pairs, key=lambda x: x["total"], reverse=True):
        key = tuple(sorted([p["agent_a"], p["agent_b"]]))
        if key not in seen:
            seen.add(key)
            unique_pairs.append(p)

    overall_reciprocity = reciprocal_weight_total / max(total_weight, 1)

    reciprocity = {
        "overall_rate": round(overall_reciprocity, 4),
        "reciprocal_pairs": len(unique_pairs),
        "top_pairs": unique_pairs[:15],
    }

    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": round(nx.density(G), 6),
        "top_commenters": sorted(out_deg.items(), key=lambda x: x[1], reverse=True)[:10],
        "most_commented": sorted(in_deg.items(), key=lambda x: x[1], reverse=True)[:10],
        "pagerank": pagerank_list,
        "communities": communities_list,
        "reciprocity": reciprocity,
    }
