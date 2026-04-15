#!/usr/bin/env python3
"""
Generate publication-quality figures for the Nature Scientific Data paper.

Each figure is a single standalone plot (not panels).
Output: paper_figures/ directory with PDF (vector) and PNG (300 DPI).

Usage:
    python generate_paper_figures.py
"""

import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

from moltbook.data import load_from_huggingface

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nature style settings
# ---------------------------------------------------------------------------
SINGLE_COL_MM = 89
DOUBLE_COL_MM = 183
MM_TO_INCH = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colour palette — muted, accessible
C_POST = "#2166AC"       # blue
C_COMMENT = "#B2182B"    # red
C_ACCENT = "#4DAF4A"     # green
C_GREY = "#999999"

OUT = Path("paper_figures")


def _save(fig, name):
    """Save figure as both PDF and PNG."""
    fig.savefig(OUT / f"{name}.pdf", format="pdf")
    fig.savefig(OUT / f"{name}.png", format="png")
    plt.close(fig)
    logger.info("Saved %s", name)


# ===================================================================
# Figure 1 — Daily Activity Timeline
# ===================================================================
def fig1_daily_timeline(posts_df, comments_df):
    logger.info("Generating Figure 1: Daily Activity Timeline")

    daily_p = posts_df.groupby("date").size()
    daily_c = comments_df.groupby("date").size()

    # Align indices
    all_dates = sorted(set(daily_p.index) | set(daily_c.index))
    dates = [pd.Timestamp(d) for d in all_dates]
    post_counts = [daily_p.get(d, 0) for d in all_dates]
    comment_counts = [daily_c.get(d, 0) for d in all_dates]

    w = DOUBLE_COL_MM * MM_TO_INCH
    fig, ax = plt.subplots(figsize=(w, w * 0.4))

    ax.fill_between(dates, post_counts, alpha=0.3, color=C_POST, linewidth=0)
    ax.plot(dates, post_counts, color=C_POST, linewidth=1.2, label="Posts")

    ax.fill_between(dates, comment_counts, alpha=0.3, color=C_COMMENT, linewidth=0)
    ax.plot(dates, comment_counts, color=C_COMMENT, linewidth=1.2, label="Comments")

    ax.set_xlabel("Date")
    ax.set_ylabel("Daily count")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
    ax.legend(frameon=False, loc="upper right")
    fig.autofmt_xdate(rotation=45, ha="right")

    _save(fig, "fig1_daily_timeline")


# ===================================================================
# Figure 2 — Agent Activity Distribution (CCDF)
# ===================================================================
def fig2_agent_activity(posts_df):
    logger.info("Generating Figure 2: Agent Activity Distribution")

    ppa = posts_df.groupby("agent_name").size().values
    ppa_sorted = np.sort(ppa)[::-1]
    ccdf_y = np.arange(1, len(ppa_sorted) + 1) / len(ppa_sorted)

    w = SINGLE_COL_MM * MM_TO_INCH
    fig, ax = plt.subplots(figsize=(w, w * 0.9))

    ax.loglog(ppa_sorted, ccdf_y, color=C_POST, linewidth=1.0, alpha=0.8)

    # Annotate key points
    median_val = int(np.median(ppa))
    mean_val = np.mean(ppa)
    ax.axvline(median_val, color=C_GREY, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.text(median_val * 1.3, 0.6, f"median = {median_val}", fontsize=6, color=C_GREY)
    ax.axvline(mean_val, color=C_ACCENT, linestyle="--", linewidth=0.6, alpha=0.7)
    ax.text(mean_val * 1.3, 0.3, f"mean = {mean_val:.0f}", fontsize=6, color=C_ACCENT)

    ax.set_xlabel("Posts per agent")
    ax.set_ylabel("CCDF  P(X ≥ x)")
    ax.set_xlim(left=1)

    _save(fig, "fig2_agent_activity_ccdf")


# ===================================================================
# Figure 3 — Agent Interaction Network
# ===================================================================
def fig3_network(posts_df, comments_df):
    logger.info("Generating Figure 3: Agent Interaction Network")

    # Build the interaction graph
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
        & (interactions["weight"] >= 3)
    ]

    G = nx.Graph()
    for _, row in interactions.iterrows():
        if G.has_edge(row["agent_name"], row["post_author"]):
            G[row["agent_name"]][row["post_author"]]["weight"] += row["weight"]
        else:
            G.add_edge(row["agent_name"], row["post_author"], weight=row["weight"])

    logger.info("Network: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # Keep largest connected component
    if G.number_of_nodes() == 0:
        logger.warning("Empty network, skipping figure 3")
        return

    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()
    logger.info("Largest component: %d nodes", G.number_of_nodes())

    # Community detection
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G, weight="weight"))
    communities = sorted(communities, key=len, reverse=True)

    # Assign community colors
    comm_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            comm_map[node] = i

    # Use a qualitative colormap
    n_comms = min(len(communities), 12)
    cmap = plt.cm.get_cmap("Set3", n_comms)
    node_colors = [cmap(comm_map.get(n, 0) % n_comms) for n in G.nodes()]

    # Node sizes by degree
    degrees = dict(G.degree(weight="weight"))
    max_deg = max(degrees.values())
    node_sizes = [max(2, 40 * (degrees[n] / max_deg) ** 0.5) for n in G.nodes()]

    # Layout — spring with weight
    logger.info("Computing layout...")
    pos = nx.spring_layout(G, k=0.8, iterations=50, seed=42, weight="weight")

    w = DOUBLE_COL_MM * MM_TO_INCH
    fig, ax = plt.subplots(figsize=(w, w * 0.8))

    nx.draw_networkx_edges(G, pos, alpha=0.03, width=0.3, edge_color=C_GREY, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_size=node_sizes, node_color=node_colors,
        alpha=0.7, linewidths=0, ax=ax,
    )

    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    ax.axis("off")

    # Legend for top communities
    from matplotlib.lines import Line2D
    legend_elements = []
    for i in range(min(6, len(communities))):
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(i),
                   markersize=5, label=f"Community {i+1} ({len(communities[i]):,})")
        )
    ax.legend(handles=legend_elements, loc="lower left", frameon=False, fontsize=6)

    _save(fig, "fig3_interaction_network")


# ===================================================================
# Figure 4 — Content Distribution Across Submolts
# ===================================================================
def fig4_submolt_distribution(posts_df):
    logger.info("Generating Figure 4: Submolt Distribution")

    top = posts_df["submolt"].value_counts().head(20)
    # Reverse for horizontal bars (top at top)
    top = top.iloc[::-1]

    w = SINGLE_COL_MM * MM_TO_INCH
    fig, ax = plt.subplots(figsize=(w, w * 1.3))

    bars = ax.barh(range(len(top)), top.values, color=C_POST, alpha=0.8, height=0.7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index, fontsize=6)
    ax.set_xlabel("Number of posts")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))

    # Add count labels
    for bar, val in zip(bars, top.values):
        ax.text(val + max(top.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=5.5, color=C_GREY)

    _save(fig, "fig4_submolt_distribution")


# ===================================================================
# Figure 5 — Hourly Posting Pattern (Polar)
# ===================================================================
def fig5_hourly_pattern(posts_df):
    logger.info("Generating Figure 5: Hourly Posting Pattern")

    hourly = posts_df.groupby("hour").size()
    hours = np.arange(24)
    counts = [hourly.get(h, 0) for h in hours]

    # Normalise to percentage
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]

    # Close the circle
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pcts_closed = pcts + [pcts[0]]
    angles_closed = np.append(angles, angles[0])

    w = SINGLE_COL_MM * MM_TO_INCH
    fig, ax = plt.subplots(figsize=(w, w), subplot_kw={"projection": "polar"})

    ax.fill(angles_closed, pcts_closed, alpha=0.25, color=C_POST)
    ax.plot(angles_closed, pcts_closed, color=C_POST, linewidth=1.2)

    # Hour labels
    ax.set_xticks(angles)
    ax.set_xticklabels([f"{h:02d}" for h in hours], fontsize=6)
    ax.set_theta_offset(np.pi / 2)  # 0h at top
    ax.set_theta_direction(-1)       # clockwise

    # Radial labels
    ax.set_ylabel("")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    ax.tick_params(axis="y", labelsize=5)

    # Uniform distribution reference
    uniform = 100 / 24
    ax.plot(angles_closed, [uniform] * 25, color=C_GREY, linewidth=0.5, linestyle="--", alpha=0.5)

    _save(fig, "fig5_hourly_pattern")


# ===================================================================
# Figure 6 — Content Length Distribution
# ===================================================================
def fig6_content_length(posts_df, comments_df):
    logger.info("Generating Figure 6: Content Length Distribution")

    post_lens = posts_df["content_length"].dropna()
    post_lens = post_lens[post_lens > 0]

    w = SINGLE_COL_MM * MM_TO_INCH
    fig, ax = plt.subplots(figsize=(w, w * 0.8))

    # Histogram with log-spaced bins
    bins = np.logspace(0, np.log10(post_lens.max()), 80)
    ax.hist(post_lens, bins=bins, alpha=0.5, color=C_POST, label="Posts", density=True, linewidth=0)

    if "content_length" in comments_df.columns:
        com_lens = comments_df["content_length"].dropna()
        com_lens = com_lens[com_lens > 0]
        ax.hist(com_lens, bins=bins, alpha=0.5, color=C_COMMENT, label="Comments", density=True, linewidth=0)

        # Median lines
        ax.axvline(com_lens.median(), color=C_COMMENT, linestyle="--", linewidth=0.7, alpha=0.7)
        ax.text(com_lens.median() * 1.2, ax.get_ylim()[1] * 0.7,
                f"median\n{com_lens.median():.0f}", fontsize=5.5, color=C_COMMENT)

    ax.axvline(post_lens.median(), color=C_POST, linestyle="--", linewidth=0.7, alpha=0.7)
    ax.text(post_lens.median() * 0.2, ax.get_ylim()[1] * 0.85,
            f"median\n{post_lens.median():.0f}", fontsize=5.5, color=C_POST)

    ax.set_xscale("log")
    ax.set_xlabel("Content length (characters)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper right")

    _save(fig, "fig6_content_length")


# ===================================================================
# Main
# ===================================================================
def main():
    OUT.mkdir(exist_ok=True)

    logger.info("Loading data...")
    posts_df, comments_df = load_from_huggingface()
    logger.info("Loaded %s posts and %s comments", f"{len(posts_df):,}", f"{len(comments_df):,}")

    fig1_daily_timeline(posts_df, comments_df)
    fig2_agent_activity(posts_df)
    fig3_network(posts_df, comments_df)
    fig4_submolt_distribution(posts_df)
    fig5_hourly_pattern(posts_df)
    fig6_content_length(posts_df, comments_df)

    logger.info("All figures saved to %s/", OUT)


if __name__ == "__main__":
    main()
