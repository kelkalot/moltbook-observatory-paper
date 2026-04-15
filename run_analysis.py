#!/usr/bin/env python3
"""
Moltbook Observatory — daily analysis runner.

Usage:
    python run_analysis.py                           # Full dataset from HuggingFace
    python run_analysis.py --date 2026-01-30         # Single day only
    python run_analysis.py --from 2026-01-29 --to 2026-01-30  # Date range
    python run_analysis.py --csv posts.csv comments.csv        # Local CSV files
    python run_analysis.py --output-dir ./my_reports           # Custom output location

Each run creates a date-stamped HTML report under reports/ and updates
the index page. To browse reports, run: python serve_reports.py

To compare two reports:
    python compare_reports.py reports/report_A.json reports/report_B.json
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from moltbook.data import load_from_huggingface, load_from_csv, filter_by_date
from moltbook.risk import run_risk_analysis, calculate_risk_level, get_injection_examples
from moltbook.sentiment import run_sentiment_analysis
from moltbook.clustering import run_agent_clustering, run_topic_modeling
from moltbook.network import run_network_analysis
from moltbook.temporal import run_temporal_analysis
from moltbook.agent_score import run_agent_scoring
from moltbook.similarity import find_near_duplicates
from moltbook.engagement import analyse_engagement_quality
from moltbook.visualizations import (
    generate_dashboard_charts,
    generate_temporal_charts,
    generate_agent_score_charts,
    generate_engagement_charts,
)
from moltbook.report import generate_html_report, update_index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Moltbook Observatory Analysis")
    parser.add_argument(
        "--csv", nargs=2, metavar=("POSTS", "COMMENTS"),
        help="Paths to local posts and comments CSV files",
    )
    parser.add_argument(
        "--dataset", default="SimulaMet/moltbook-observatory-archive",
        help="HuggingFace dataset name (default: SimulaMet/moltbook-observatory-archive)",
    )
    parser.add_argument(
        "--output-dir", default="reports",
        help="Directory for reports (default: reports/)",
    )
    parser.add_argument(
        "--date", metavar="YYYY-MM-DD",
        help="Analyse a single day only",
    )
    parser.add_argument(
        "--from", dest="date_from", metavar="YYYY-MM-DD",
        help="Start of date range (inclusive)",
    )
    parser.add_argument(
        "--to", dest="date_to", metavar="YYYY-MM-DD",
        help="End of date range (inclusive)",
    )
    args = parser.parse_args()

    # Resolve --date shorthand into --from / --to
    if args.date:
        args.date_from = args.date
        args.date_to = args.date

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    # Include date range in filename for clarity
    if args.date_from or args.date_to:
        tag = f"{args.date_from or 'start'}_to_{args.date_to or 'latest'}"
    else:
        tag = timestamp
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading data…")
    if args.csv:
        posts_df, comments_df = load_from_csv(args.csv[0], args.csv[1])
    else:
        posts_df, comments_df = load_from_huggingface(args.dataset)

    logger.info("Loaded %s posts and %s comments", f"{len(posts_df):,}", f"{len(comments_df):,}")

    # Apply date filter if requested
    if args.date_from or args.date_to:
        posts_df, comments_df = filter_by_date(posts_df, comments_df, args.date_from, args.date_to)

    # ------------------------------------------------------------------
    # 2. Risk analysis
    # ------------------------------------------------------------------
    logger.info("Running risk analysis…")
    risk = run_risk_analysis(posts_df, comments_df)
    risk_level = calculate_risk_level(risk)
    logger.info("Risk level: %s", risk_level)

    # ------------------------------------------------------------------
    # 3. Sentiment analysis
    # ------------------------------------------------------------------
    logger.info("Running sentiment analysis…")
    sentiment = run_sentiment_analysis(posts_df, comments_df)
    logger.info("Overall sentiment: %.4f", sentiment["overall_sentiment"])

    # ------------------------------------------------------------------
    # 4. Clustering & topics
    # ------------------------------------------------------------------
    logger.info("Running agent clustering…")
    agent_features, clusters = run_agent_clustering(posts_df)

    logger.info("Running topic modeling…")
    topics = run_topic_modeling(posts_df)

    # ------------------------------------------------------------------
    # 5. Network analysis
    # ------------------------------------------------------------------
    logger.info("Running network analysis (PageRank, communities, reciprocity)…")
    network = run_network_analysis(posts_df, comments_df)

    # ------------------------------------------------------------------
    # 6. Temporal analysis (bursts, sentiment velocity, agent influx)
    # ------------------------------------------------------------------
    logger.info("Running temporal analysis…")
    temporal = run_temporal_analysis(posts_df, comments_df)
    bursts = temporal["bursts"]
    sv = temporal["sentiment_velocity"]
    logger.info(
        "Sentiment trend: %s (slope %.5f) — %d activity burst(s) detected",
        sv.get("trend", "?"), sv.get("slope", 0), bursts.get("total_bursts", 0),
    )

    # ------------------------------------------------------------------
    # 7. Agent risk scoring
    # ------------------------------------------------------------------
    logger.info("Computing per-agent risk scores…")
    agent_scores = run_agent_scoring(posts_df, comments_df)
    logger.info(
        "Agent scores: mean %.1f, max %.1f — %d critical-tier agents",
        agent_scores["score_stats"].get("mean", 0),
        agent_scores["score_stats"].get("max", 0),
        agent_scores["tier_counts"].get("critical", 0),
    )

    # ------------------------------------------------------------------
    # 8. Near-duplicate / fuzzy similarity
    # ------------------------------------------------------------------
    logger.info("Detecting near-duplicate content…")
    similarity = find_near_duplicates(posts_df)
    logger.info(
        "%d near-duplicate clusters (%d posts involved)",
        similarity["unique_clusters"], similarity["near_duplicate_posts"],
    )

    # ------------------------------------------------------------------
    # 9. Engagement quality
    # ------------------------------------------------------------------
    logger.info("Analysing engagement quality…")
    engagement = analyse_engagement_quality(posts_df, comments_df)
    logger.info("Organic engagement ratio: %.1f%%", engagement["organic_ratio"] * 100)

    # ------------------------------------------------------------------
    # 10. Generate charts
    # ------------------------------------------------------------------
    logger.info("Generating charts…")
    charts = generate_dashboard_charts(posts_df, comments_df)
    charts.update(generate_temporal_charts(temporal))
    charts.update(generate_agent_score_charts(agent_scores))
    charts.update(generate_engagement_charts(engagement))

    # ------------------------------------------------------------------
    # 11. Write HTML report
    # ------------------------------------------------------------------
    report_path = output_dir / f"report_{tag}.html"
    generate_html_report(
        risk, sentiment, clusters, topics, network,
        temporal, agent_scores, similarity, engagement,
        charts, report_path,
    )
    logger.info("Report written to %s", report_path)

    # ------------------------------------------------------------------
    # 12. Write JSON (machine-readable companion)
    # ------------------------------------------------------------------
    json_path = output_dir / f"report_{tag}.json"
    combined = {
        "timestamp": timestamp,
        "date_filter": {"from": args.date_from, "to": args.date_to},
        "risk": risk.to_dict(),
        "sentiment": sentiment,
        "clusters": clusters,
        "topics": {str(k): v for k, v in topics.items()},
        "network": {
            "nodes": network["nodes"],
            "edges": network["edges"],
            "density": network.get("density", 0),
            "top_commenters": [[a, w] for a, w in network.get("top_commenters", [])],
            "most_commented": [[a, w] for a, w in network.get("most_commented", [])],
            "pagerank": network.get("pagerank", []),
            "communities": network.get("communities", []),
            "reciprocity": network.get("reciprocity", {}),
        },
        "temporal": temporal,
        "agent_scores": agent_scores,
        "similarity": similarity,
        "engagement": engagement,
    }
    json_path.write_text(json.dumps(combined, indent=2, default=str))
    logger.info("JSON written to %s", json_path)

    # ------------------------------------------------------------------
    # 13. Save injection examples
    # ------------------------------------------------------------------
    examples = get_injection_examples(posts_df)
    examples_path = output_dir / f"injection_examples_{tag}.json"
    examples_path.write_text(json.dumps(examples, indent=2, default=str))

    # ------------------------------------------------------------------
    # 14. Update index
    # ------------------------------------------------------------------
    idx = update_index(output_dir)
    logger.info("Index updated at %s", idx)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    sd = sentiment.get("sentiment_distribution", {})
    print(f"""
{'=' * 60}
🦞 MOLTBOOK ANALYSIS COMPLETE
{'=' * 60}

📊 Posts: {risk.total_posts:,}   Comments: {risk.total_comments:,}   Agents: {risk.unique_posting_agents:,}
📅 Data range: {risk.date_range_start[:10] if risk.date_range_start else 'N/A'} → {risk.date_range_end[:10] if risk.date_range_end else 'N/A'}

🚨 Risk level: {risk_level}
   Injections: {risk.injection_posts} ({risk.injection_percentage:.1f}%)
   Crypto:     {risk.crypto_posts:,} ({risk.crypto_percentage:.1f}%)
   Spam:       {risk.duplicate_posts:,} exact dupes + {similarity['near_duplicate_posts']} near-dupes

😊 Sentiment: {sentiment['overall_sentiment']:.4f} (trend: {sv.get('trend', '?')}, slope: {sv.get('slope', 0):.5f})
   Positive: {sd.get('Positive', 0):,}  Neutral: {sd.get('Neutral', 0):,}  Negative: {sd.get('Negative', 0):,}

🤖 Clusters: {len(clusters)}   Topics: {len(topics)}   Network: {network['nodes']} nodes, {len(network.get('communities', []))} communities
   Activity bursts: {bursts.get('total_bursts', 0)}   Agent influx spikes: {len(temporal['agent_influx'].get('spike_days', []))}

⚠️  Agent risk: mean {agent_scores['score_stats'].get('mean', 0):.1f}, max {agent_scores['score_stats'].get('max', 0):.1f} — {agent_scores['tier_counts'].get('critical', 0)} critical agents
   Engagement: {engagement['organic_ratio']:.0%} organic, {engagement['self_interaction'].get('self_rate', 0):.1%} self, {engagement['bot_engagement'].get('bot_rate', 0):.1%} bot

📁 Report: {report_path}
   JSON:   {json_path}
   Browse: python serve_reports.py
""")


if __name__ == "__main__":
    main()
