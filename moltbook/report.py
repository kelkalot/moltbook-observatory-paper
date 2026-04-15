"""Generate self-contained HTML reports and an index page for browsing."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from html import escape

from .risk import RiskMetrics, calculate_risk_level


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

_RISK_COLOURS = {
    "CRITICAL": "#D32F2F",
    "HIGH": "#F57C00",
    "MEDIUM": "#FBC02D",
    "LOW": "#4CAF50",
}


# ---------------------------------------------------------------------------
# Report HTML
# ---------------------------------------------------------------------------

def generate_html_report(
    risk: RiskMetrics,
    sentiment: dict,
    clusters: dict,
    topics: dict,
    network: dict,
    temporal: dict,
    agent_scores: dict,
    similarity: dict,
    engagement: dict,
    charts: dict[str, str],  # name → base64 PNG
    output_path: str | Path,
) -> Path:
    """Write a self-contained HTML report to *output_path*."""
    output_path = Path(output_path)
    risk_level = calculate_risk_level(risk)
    colour = _RISK_COLOURS.get(risk_level, "#757575")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_start = risk.date_range_start[:10] if risk.date_range_start else "N/A"
    date_end = risk.date_range_end[:10] if risk.date_range_end else "N/A"

    def _img(name: str) -> str:
        b64 = charts.get(name, "")
        if not b64:
            return ""
        return f'<img src="data:image/png;base64,{b64}" style="max-width:100%;border-radius:8px;">'

    def _table_rows(d: dict, limit: int = 10) -> str:
        rows = ""
        for k, v in list(d.items())[:limit]:
            rows += f"<tr><td>{escape(str(k))}</td><td>{v}</td></tr>\n"
        return rows

    def _topic_section() -> str:
        if not topics:
            return "<p>No topics extracted.</p>"
        parts = []
        for tid, t in topics.items():
            kw = ", ".join(t["keywords"][:8])
            parts.append(
                f'<div class="card"><strong>Topic {tid}</strong> '
                f'({t["count"]} posts, {t["percentage"]}%)<br>'
                f'<span class="muted">{escape(kw)}</span></div>'
            )
        return "\n".join(parts)

    def _cluster_section() -> str:
        if not clusters:
            return "<p>No clusters.</p>"
        parts = []
        for cid, c in clusters.items():
            agents = ", ".join(c.get("example_agents", [])[:3])
            parts.append(
                f'<div class="card"><strong>Cluster {cid}</strong> — '
                f'{c["size"]} agents<br>'
                f'Avg posts: {c["avg_posts"]}, Avg score: {c["avg_score"]}, '
                f'Sentiment: {c["avg_sentiment"]}<br>'
                f'<span class="muted">e.g. {escape(agents)}</span></div>'
            )
        return "\n".join(parts)

    def _network_list(items: list) -> str:
        if not items:
            return "<em>N/A</em>"
        return ", ".join(f"{escape(str(a))} ({w})" for a, w in items[:5])

    def _pagerank_table(pr_list: list) -> str:
        if not pr_list:
            return "<p class='muted'>Not enough data.</p>"
        rows = "".join(
            f"<tr><td>{escape(str(p['agent']))}</td><td>{p['score']:.6f}</td></tr>"
            for p in pr_list[:10]
        )
        return f'<table><tr><th>Agent</th><th>PageRank</th></tr>{rows}</table>'

    def _communities_section(comms: list) -> str:
        if not comms:
            return "<p class='muted'>No communities detected.</p>"
        parts = []
        for c in comms[:10]:
            members = ", ".join(c.get("top_members", [])[:4])
            parts.append(
                f'<div class="card"><strong>Community {c["id"]}</strong> — '
                f'{c["size"]} agents<br>'
                f'<span class="muted">Lead: {escape(str(c.get("representative", "?")))}'
                f' · {escape(members)}</span></div>'
            )
        return '<div class="grid">' + "\n".join(parts) + "</div>"

    def _reciprocity_table(pairs: list) -> str:
        if not pairs:
            return "<p class='muted'>No reciprocal interactions above threshold.</p>"
        rows = "".join(
            f"<tr><td>{escape(str(p['agent_a']))}</td><td>{escape(str(p['agent_b']))}</td>"
            f"<td>{p['a_to_b']}</td><td>{p['b_to_a']}</td><td>{p['total']}</td></tr>"
            for p in pairs[:10]
        )
        return (
            '<table><tr><th>Agent A</th><th>Agent B</th>'
            f'<th>A→B</th><th>B→A</th><th>Total</th></tr>{rows}</table>'
        )

    def _sentiment_velocity_section(sv: dict) -> str:
        trend = sv.get("trend", "unknown")
        slope = sv.get("slope", 0)
        avg_v = sv.get("avg_velocity", 0)
        trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}.get(trend, "❓")
        return (
            f'<div class="grid">'
            f'<div class="card"><div class="stat">{trend_icon} {trend.title()}</div><div class="stat-label">Trend</div></div>'
            f'<div class="card"><div class="stat">{slope:+.5f}</div><div class="stat-label">Slope (per day)</div></div>'
            f'<div class="card"><div class="stat">{avg_v:+.4f}</div><div class="stat-label">Avg Daily Δ</div></div>'
            f'</div>'
        )

    def _bursts_section(bursts: dict) -> str:
        burst_list = bursts.get("bursts", [])
        if not burst_list:
            return f'<p>No activity bursts detected (threshold: {bursts.get("threshold", "?")} posts/hour).</p>'
        rows = ""
        for b in burst_list[:10]:
            top_a = ", ".join(f"{a} ({c})" for a, c in list(b.get("top_agents", {}).items())[:3])
            rows += (
                f'<tr><td>{escape(b["start"])}</td><td>{b["count"]}</td>'
                f'<td>{b["z_score"]:.1f}σ</td><td>{escape(top_a)}</td></tr>'
            )
        return (
            f'<p>{len(burst_list)} burst(s) detected '
            f'(mean: {bursts.get("mean_hourly_rate", 0):.0f}/h, threshold: {bursts.get("threshold", 0):.0f}/h).</p>'
            f'<table><tr><th>Start</th><th>Posts</th><th>Z-score</th><th>Top Agents</th></tr>{rows}</table>'
        )

    def _influx_section(influx: dict) -> str:
        spikes = influx.get("spike_days", [])
        mean_d = influx.get("mean_daily_new", 0)
        if not spikes:
            return f'<p>No influx spikes detected (mean {mean_d:.0f} new agents/day).</p>'
        return (
            f'<p>Spike days (>2σ above mean of {mean_d:.0f}/day): '
            f'<strong>{", ".join(spikes)}</strong></p>'
        )

    def _risky_agents_table(agents: list) -> str:
        if not agents:
            return "<p class='muted'>No agents scored.</p>"
        tier_col = {"low": "#4CAF50", "moderate": "#FBC02D", "high": "#F57C00", "critical": "#D32F2F"}
        rows = ""
        for a in agents[:15]:
            col = tier_col.get(a.get("risk_tier", ""), "#999")
            reasons = ", ".join(a.get("reasons", []))
            rows += (
                f'<tr><td>{escape(str(a["agent"]))}</td>'
                f'<td><strong style="color:{col}">{a["risk_score"]:.0f}</strong></td>'
                f'<td style="color:{col}">{escape(a.get("risk_tier", ""))}</td>'
                f'<td>{a.get("post_count", 0)}</td>'
                f'<td class="muted">{escape(reasons)}</td></tr>'
            )
        return (
            '<table><tr><th>Agent</th><th>Score</th><th>Tier</th>'
            f'<th>Posts</th><th>Main Signals</th></tr>{rows}</table>'
        )

    def _similarity_section(clusters: list) -> str:
        if not clusters:
            return "<p class='muted'>No near-duplicate clusters found.</p>"
        rows = ""
        for i, cl in enumerate(clusters[:10]):
            agents = ", ".join(f"{a} ({c})" for a, c in list(cl.get("agents", {}).items())[:3])
            rows += (
                f'<tr><td>{i + 1}</td><td>{cl["size"]}</td>'
                f'<td>{escape(str(cl.get("sample_title", ""))[:50])}</td>'
                f'<td class="muted">{escape(agents)}</td></tr>'
            )
        return (
            '<table><tr><th>#</th><th>Cluster Size</th><th>Sample Title</th>'
            f'<th>Top Agents</th></tr>{rows}</table>'
        )

    def _self_interactors_table(top: dict) -> str:
        if not top:
            return ""
        rows = "".join(
            f"<tr><td>{escape(str(a))}</td><td>{c}</td></tr>"
            for a, c in list(top.items())[:10]
        )
        return f'<h3>Top Self-Interactors</h3><table><tr><th>Agent</th><th>Self-Comments</th></tr>{rows}</table>'

    def _engagement_anomalies_table(anomalies: list) -> str:
        if not anomalies:
            return ""
        rows = ""
        for a in anomalies[:10]:
            rows += (
                f'<tr><td>{escape(str(a.get("agent", "")))}</td>'
                f'<td>{escape(str(a.get("title", ""))[:50])}</td>'
                f'<td>{a.get("score", 0):,}</td>'
                f'<td>{a.get("content_length", 0)}</td>'
                f'<td>{a.get("z_score", 0):.1f}σ</td></tr>'
            )
        return (
            '<h3>Engagement Anomalies (high engagement / short content)</h3>'
            '<table><tr><th>Agent</th><th>Title</th><th>Score</th>'
            f'<th>Length</th><th>Z-score</th></tr>{rows}</table>'
        )

    # Top posts table
    top_posts_rows = ""
    for p in risk.top_posts[:5]:
        agent = escape(str(p.get("agent_name", "")))
        title = escape(str(p.get("title", ""))[:60])
        score = f'{p.get("score", 0):,}'
        top_posts_rows += f"<tr><td>{agent}</td><td>{title}</td><td>{score}</td></tr>\n"

    # Sentiment distribution
    sd = sentiment.get("sentiment_distribution", {})
    positive = sd.get("Positive", 0)
    neutral = sd.get("Neutral", 0)
    negative = sd.get("Negative", 0)

    # Network/reciprocity values (extracted to avoid f-string chained .get() issues)
    reciprocity_data = network.get("reciprocity", {}) if isinstance(network.get("reciprocity"), dict) else {}
    reciprocity_rate = reciprocity_data.get("overall_rate", 0)
    reciprocity_pairs = reciprocity_data.get("top_pairs", [])

    # Temporal values
    sentiment_velocity = temporal.get("sentiment_velocity", {}) if isinstance(temporal.get("sentiment_velocity"), dict) else {}
    bursts_data = temporal.get("bursts", {}) if isinstance(temporal.get("bursts"), dict) else {}
    agent_influx = temporal.get("agent_influx", {}) if isinstance(temporal.get("agent_influx"), dict) else {}

    # Agent scores values
    score_stats = agent_scores.get("score_stats", {}) if isinstance(agent_scores.get("score_stats"), dict) else {}
    tier_counts = agent_scores.get("tier_counts", {}) if isinstance(agent_scores.get("tier_counts"), dict) else {}
    mean_score = score_stats.get("mean", 0)
    max_score_stat = score_stats.get("max", 0)
    critical_agents = tier_counts.get("critical", 0)
    high_risk_agents = tier_counts.get("high", 0)

    # Engagement values
    self_interaction = engagement.get("self_interaction", {}) if isinstance(engagement.get("self_interaction"), dict) else {}
    bot_engagement = engagement.get("bot_engagement", {}) if isinstance(engagement.get("bot_engagement"), dict) else {}
    self_rate = self_interaction.get("self_rate", 0)
    bot_rate = bot_engagement.get("bot_rate", 0)
    top_self_interactors = self_interaction.get("top_self_interactors", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Moltbook Report — {date_start} to {date_end}</title>
<style>
  :root {{ --accent: #1A237E; --bg: #f5f7fa; --card-bg: #fff; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: #333; line-height: 1.6; padding: 2rem; max-width: 1100px; margin: auto; }}
  h1 {{ color: var(--accent); margin-bottom: 0.3rem; }}
  h2 {{ color: var(--accent); border-bottom: 2px solid var(--accent); padding-bottom: 0.3rem; margin: 2rem 0 1rem; }}
  h3 {{ margin: 1.2rem 0 0.5rem; }}
  .meta {{ color: #757575; margin-bottom: 1.5rem; }}
  .risk-badge {{ display: inline-block; padding: 0.6rem 2rem; border-radius: 8px;
                 font-size: 1.8rem; font-weight: 700; color: #fff; background: {colour};
                 margin: 1rem 0; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }}
  .card {{ background: var(--card-bg); border-radius: 10px; padding: 1.2rem;
           box-shadow: 0 2px 8px rgba(0,0,0,.07); }}
  .stat {{ font-size: 1.5rem; font-weight: 700; }}
  .stat-label {{ font-size: 0.85rem; color: #757575; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.5rem 0 1rem; }}
  th, td {{ text-align: left; padding: 0.5rem 0.8rem; border-bottom: 1px solid #e0e0e0; }}
  th {{ background: #f0f0f0; font-size: 0.85rem; text-transform: uppercase; color: #555; }}
  .muted {{ color: #999; font-size: 0.9rem; }}
  .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1rem 0; }}
  .chart-row img {{ width: 100%; }}
  @media (max-width: 700px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
<h1>🦞 Moltbook Risk Assessment Report</h1>
<p class="meta">Generated: {now} &nbsp;|&nbsp; Data: {date_start} to {date_end}</p>

<div class="risk-badge">{risk_level}</div>

<!-- Overview -->
<h2>Platform Overview</h2>
<div class="grid">
  <div class="card"><div class="stat">{risk.total_posts:,}</div><div class="stat-label">Posts</div></div>
  <div class="card"><div class="stat">{risk.total_comments:,}</div><div class="stat-label">Comments</div></div>
  <div class="card"><div class="stat">{risk.unique_posting_agents:,}</div><div class="stat-label">Posting Agents</div></div>
  <div class="card"><div class="stat">{risk.unique_commenting_agents:,}</div><div class="stat-label">Commenting Agents</div></div>
</div>

<div class="chart-row">
  {_img("daily_volume")}
  {_img("hourly_activity")}
</div>

<!-- Security -->
<h2>Security Risks</h2>
<div class="grid">
  <div class="card"><div class="stat">{risk.injection_posts} <span class="muted">({risk.injection_percentage:.1f}%)</span></div><div class="stat-label">Prompt Injection Posts</div></div>
  <div class="card"><div class="stat">{risk.injection_agents}</div><div class="stat-label">Unique Injection Agents</div></div>
  <div class="card"><div class="stat">{risk.api_injection_comments}</div><div class="stat-label">API Injection Comments</div></div>
  <div class="card"><div class="stat">{risk.manipulation_comments}</div><div class="stat-label">Manipulation Comments</div></div>
</div>

<h3>Top Injection Agents</h3>
<table><tr><th>Agent</th><th>Injection Posts</th></tr>
{_table_rows(risk.top_injection_agents)}
</table>

<!-- Financial -->
<h2>Financial Risks</h2>
<div class="grid">
  <div class="card"><div class="stat">{risk.crypto_posts:,} <span class="muted">({risk.crypto_percentage:.1f}%)</span></div><div class="stat-label">Crypto-Related Posts</div></div>
  <div class="card"><div class="stat">{risk.pump_dump_posts:,}</div><div class="stat-label">Pump-and-Dump Indicators</div></div>
</div>

<!-- Spam -->
<h2>Spam &amp; Bot Activity</h2>
<div class="grid">
  <div class="card"><div class="stat">{risk.duplicate_posts:,}</div><div class="stat-label">Duplicate Posts</div></div>
  <div class="card"><div class="stat">{risk.bot_comments:,}</div><div class="stat-label">Bot Comments</div></div>
</div>

<h3>Top Spammers</h3>
<table><tr><th>Agent</th><th>Duplicate Posts</th></tr>
{_table_rows(risk.top_spammers)}
</table>

<!-- Harmful -->
<h2>Harmful Content</h2>
<div class="grid">
  <div class="card"><div class="stat">{len(risk.harmful_usernames)}</div><div class="stat-label">Harmful Usernames</div></div>
  <div class="card"><div class="stat">{risk.ideological_posts:,}</div><div class="stat-label">Ideological Posts</div></div>
</div>
{('<p><strong>Flagged:</strong> ' + escape(', '.join(risk.harmful_usernames[:10])) + '</p>') if risk.harmful_usernames else ''}

<!-- Sentiment -->
<h2>Sentiment Analysis</h2>
<div class="grid">
  <div class="card"><div class="stat">{sentiment.get('overall_sentiment', 0):.4f}</div><div class="stat-label">Overall Sentiment</div></div>
  <div class="card"><div class="stat">{positive:,}</div><div class="stat-label">Positive Posts</div></div>
  <div class="card"><div class="stat">{neutral:,}</div><div class="stat-label">Neutral Posts</div></div>
  <div class="card"><div class="stat">{negative:,}</div><div class="stat-label">Negative Posts</div></div>
</div>
<div class="chart-row">
  {_img("sentiment_pie")}
  {_img("sentiment_trend")}
</div>

<!-- Engagement -->
<h2>Top Engaged Content</h2>
<div class="grid">
  <div class="card"><div class="stat">{risk.max_score:,}</div><div class="stat-label">Highest Score</div></div>
  <div class="card"><div class="stat">{risk.high_engagement_posts:,}</div><div class="stat-label">High-Engagement Posts (&gt;100)</div></div>
</div>

<h3>Top Posts</h3>
<table><tr><th>Agent</th><th>Title</th><th>Score</th></tr>
{top_posts_rows}
</table>

<div class="chart-row">
  {_img("top_communities")}
  {_img("top_posters")}
</div>
{_img("score_dist")}

<!-- Clustering -->
<h2>Agent Clustering</h2>
{_cluster_section()}

<!-- Topics -->
<h2>Topic Modeling</h2>
{_topic_section()}

<!-- Network -->
<h2>Social Network</h2>
<div class="grid">
  <div class="card"><div class="stat">{network.get('nodes', 0)}</div><div class="stat-label">Nodes</div></div>
  <div class="card"><div class="stat">{network.get('edges', 0)}</div><div class="stat-label">Edges</div></div>
  <div class="card"><div class="stat">{network.get('density', 0):.6f}</div><div class="stat-label">Density</div></div>
  <div class="card"><div class="stat">{reciprocity_rate:.1%}</div><div class="stat-label">Reciprocity Rate</div></div>
</div>

<h3>Influential Agents (PageRank)</h3>
{_pagerank_table(network.get('pagerank', []))}

<h3>Communities</h3>
{_communities_section(network.get('communities', []))}

<h3>Reciprocal Pairs (collusion signal)</h3>
{_reciprocity_table(reciprocity_pairs)}

<p><strong>Top commenters:</strong> {_network_list(network.get('top_commenters', []))}</p>
<p><strong>Most commented on:</strong> {_network_list(network.get('most_commented', []))}</p>

<!-- Temporal -->
<h2>Temporal Analysis</h2>

<h3>Sentiment Velocity</h3>
{_sentiment_velocity_section(sentiment_velocity)}
<div class="chart-row">
  {_img("sentiment_velocity")}
  {_img("agent_influx")}
</div>

<h3>Activity Bursts</h3>
{_bursts_section(bursts_data)}

<h3>Agent Influx</h3>
{_influx_section(agent_influx)}

<!-- Agent Risk Scoring -->
<h2>Agent Risk Scores</h2>
<div class="grid">
  <div class="card"><div class="stat">{mean_score:.1f}</div><div class="stat-label">Mean Score</div></div>
  <div class="card"><div class="stat">{max_score_stat:.1f}</div><div class="stat-label">Max Score</div></div>
  <div class="card"><div class="stat">{critical_agents}</div><div class="stat-label">Critical Agents</div></div>
  <div class="card"><div class="stat">{high_risk_agents}</div><div class="stat-label">High-Risk Agents</div></div>
</div>
<div class="chart-row">
  {_img("risk_tiers")}
  {_img("top_risky_agents")}
</div>

<h3>Top Risky Agents</h3>
{_risky_agents_table(agent_scores.get('top_risky', []))}

<!-- Fuzzy Duplicates -->
<h2>Near-Duplicate Content</h2>
<div class="grid">
  <div class="card"><div class="stat">{similarity.get('unique_clusters', 0)}</div><div class="stat-label">Fuzzy-Duplicate Clusters</div></div>
  <div class="card"><div class="stat">{similarity.get('near_duplicate_posts', 0):,}</div><div class="stat-label">Posts Involved</div></div>
  <div class="card"><div class="stat">{similarity.get('sample_size_analysed', 0):,}</div><div class="stat-label">Posts Sampled</div></div>
</div>
{_similarity_section(similarity.get('clusters', []))}

<!-- Engagement Quality -->
<h2>Engagement Quality</h2>
<div class="grid">
  <div class="card"><div class="stat">{engagement.get('organic_ratio', 0):.1%}</div><div class="stat-label">Organic Engagement</div></div>
  <div class="card"><div class="stat">{self_rate:.1%}</div><div class="stat-label">Self-Interaction Rate</div></div>
  <div class="card"><div class="stat">{bot_rate:.1%}</div><div class="stat-label">Bot Engagement Rate</div></div>
</div>
{_img("engagement_quality")}

{_self_interactors_table(top_self_interactors)}
{_engagement_anomalies_table(engagement.get('engagement_anomalies', []))}

<hr style="margin:3rem 0 1rem;">
<p class="muted" style="text-align:center;">Auto-generated by Moltbook Observatory &middot;
Data: <a href="https://huggingface.co/datasets/SimulaMet/moltbook-observatory-archive">SimulaMet/moltbook-observatory-archive</a></p>
</body>
</html>"""

    output_path.write_text(html)
    return output_path


# ---------------------------------------------------------------------------
# Index page — lists all reports for easy browsing
# ---------------------------------------------------------------------------

def update_index(reports_dir: str | Path) -> Path:
    """Scan *reports_dir* for report_*.html and comparison_*.html and write an index.html."""
    reports_dir = Path(reports_dir)
    report_files = sorted(reports_dir.glob("report_*.html"), reverse=True)
    comparison_files = sorted(reports_dir.glob("comparison_*.html"), reverse=True)

    def _rows(files, kind="report"):
        rows = ""
        for f in files:
            name = f.stem.replace("report_", "").replace("comparison_", "").replace("_", " ")
            size_kb = f.stat().st_size // 1024
            rows += (
                f'<tr><td><a href="{f.name}">{name}</a></td>'
                f'<td>{kind}</td><td>{f.name}</td><td>{size_kb} KB</td></tr>\n'
            )
        return rows

    report_rows = _rows(report_files, "analysis")
    comparison_rows = _rows(comparison_files, "comparison")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Moltbook Observatory — Reports</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #333; }}
  h1 {{ color: #1A237E; }}
  h2 {{ color: #1A237E; margin-top: 2rem; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
  th, td {{ text-align: left; padding: 0.6rem 0.8rem; border-bottom: 1px solid #e0e0e0; }}
  th {{ background: #f0f0f0; font-size: 0.85rem; text-transform: uppercase; color: #555; }}
  a {{ color: #1A237E; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .muted {{ color: #999; font-size: 0.9rem; }}
</style>
</head>
<body>
<h1>🦞 Moltbook Observatory — Reports</h1>
<p class="muted">{len(report_files)} analysis report(s), {len(comparison_files)} comparison(s). Most recent first.</p>

<h2>Analysis Reports</h2>
<table>
<tr><th>Label</th><th>Type</th><th>File</th><th>Size</th></tr>
{report_rows}
</table>

{"<h2>Comparisons</h2>" + "<table><tr><th>Label</th><th>Type</th><th>File</th><th>Size</th></tr>" + comparison_rows + "</table>" if comparison_files else ""}

<hr style="margin-top:2rem;">
<p class="muted">Serve this directory with <code>python serve_reports.py</code></p>
</body>
</html>"""

    idx = reports_dir / "index.html"
    idx.write_text(html)
    return idx


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def generate_comparison_html(
    baseline: dict,
    current: dict,
    output_path: str | Path,
) -> Path:
    """Generate a side-by-side comparison HTML report from two JSON report dicts."""
    import math
    from datetime import datetime as _dt

    output_path = Path(output_path)
    now = _dt.now().strftime("%Y-%m-%d %H:%M:%S")

    b_risk = baseline.get("risk", {})
    c_risk = current.get("risk", {})
    b_sent = baseline.get("sentiment", {})
    c_sent = current.get("sentiment", {})
    b_net = baseline.get("network", {})
    c_net = current.get("network", {})
    b_temp = baseline.get("temporal", {})
    c_temp = current.get("temporal", {})
    b_as = baseline.get("agent_scores", {})
    c_as = current.get("agent_scores", {})
    b_sim = baseline.get("similarity", {})
    c_sim = current.get("similarity", {})
    b_eng = baseline.get("engagement", {})
    c_eng = current.get("engagement", {})

    b_filter = baseline.get("date_filter", {})
    c_filter = current.get("date_filter", {})
    b_label = _period_label(b_filter, b_risk)
    c_label = _period_label(c_filter, c_risk)

    # --- Significance helpers ---
    def _proportion_z(p1, n1, p2, n2):
        """Two-proportion z-test. Returns (z, significant_at_95)."""
        try:
            p1, p2 = float(p1) / 100, float(p2) / 100  # expect percentages
            n1, n2 = int(n1), int(n2)
            if n1 == 0 or n2 == 0:
                return 0, False
            p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)
            if p_pool == 0 or p_pool == 1:
                return 0, False
            se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            z = (p2 - p1) / se if se > 0 else 0
            return round(z, 2), abs(z) > 1.96
        except (TypeError, ValueError, ZeroDivisionError):
            return 0, False

    def _sig_badge(is_sig):
        if is_sig:
            return ' <span style="background:#D32F2F;color:#fff;padding:1px 6px;border-radius:3px;font-size:0.75rem;">sig</span>'
        return ' <span style="color:#aaa;font-size:0.75rem;">n.s.</span>'

    def _delta(cur, base, fmt=",", is_pct=False, sig_html=""):
        """Return HTML showing value + coloured delta."""
        try:
            c_val = float(cur)
            b_val = float(base)
        except (TypeError, ValueError):
            return f"<td>{cur}</td><td></td>"
        diff = c_val - b_val
        if b_val != 0:
            pct = diff / b_val * 100
            pct_str = f" ({pct:+.1f}%)"
        else:
            pct_str = ""
        colour = "#E74C3C" if diff > 0 else "#27AE60" if diff < 0 else "#999"
        sign = "+" if diff > 0 else ""
        if is_pct:
            return f'<td>{c_val:.1f}%</td><td style="color:{colour}">{sign}{diff:.1f}pp{sig_html}</td>'
        if isinstance(cur, float):
            return f'<td>{c_val:.4f}</td><td style="color:{colour}">{sign}{diff:.4f}{pct_str}{sig_html}</td>'
        return f'<td>{int(c_val):,}</td><td style="color:{colour}">{sign}{int(diff):,}{pct_str}{sig_html}</td>'

    def _row(label, b_val, c_val, is_pct=False, sig_html=""):
        if is_pct:
            return f'<tr><td><strong>{label}</strong></td><td>{_safe_pct(b_val)}</td>{_delta(c_val, b_val, is_pct=True, sig_html=sig_html)}</tr>'
        if isinstance(b_val, float) and not is_pct:
            return f'<tr><td><strong>{label}</strong></td><td>{b_val:.4f}</td>{_delta(c_val, b_val, sig_html=sig_html)}</tr>'
        return f'<tr><td><strong>{label}</strong></td><td>{_safe_int(b_val)}</td>{_delta(c_val, b_val, sig_html=sig_html)}</tr>'

    def _safe_int(v):
        try:
            return f"{int(v):,}"
        except (TypeError, ValueError):
            return str(v)

    def _safe_pct(v):
        try:
            return f"{float(v):.1f}%"
        except (TypeError, ValueError):
            return str(v)

    # Compute significance for key rate metrics
    b_n = b_risk.get("total_posts", 0)
    c_n = c_risk.get("total_posts", 0)

    _, inj_sig = _proportion_z(
        b_risk.get("injection_percentage", 0), b_n,
        c_risk.get("injection_percentage", 0), c_n,
    )
    _, crypto_sig = _proportion_z(
        b_risk.get("crypto_percentage", 0), b_n,
        c_risk.get("crypto_percentage", 0), c_n,
    )

    # Build risk-level badges
    b_rl = b_risk.get("risk_level", "?")
    c_rl = c_risk.get("risk_level", "?")
    b_rl_col = _RISK_COLOURS.get(b_rl, "#757575")
    c_rl_col = _RISK_COLOURS.get(c_rl, "#757575")

    # Sentiment distributions
    b_sd = b_sent.get("sentiment_distribution", {})
    c_sd = c_sent.get("sentiment_distribution", {})

    # Temporal
    b_sv = b_temp.get("sentiment_velocity", {})
    c_sv = c_temp.get("sentiment_velocity", {})
    b_bursts = b_temp.get("bursts", {})
    c_bursts = c_temp.get("bursts", {})
    b_influx = b_temp.get("agent_influx", {})
    c_influx = c_temp.get("agent_influx", {})

    # Agent scores
    b_ss = b_as.get("score_stats", {})
    c_ss = c_as.get("score_stats", {})
    b_tc = b_as.get("tier_counts", {})
    c_tc = c_as.get("tier_counts", {})

    # Engagement
    b_si = b_eng.get("self_interaction", {})
    c_si = c_eng.get("self_interaction", {})
    b_be = b_eng.get("bot_engagement", {})
    c_be = c_eng.get("bot_engagement", {})

    # Network/reciprocity (extracted to avoid f-string issues)
    b_recip = b_net.get("reciprocity", {}) if isinstance(b_net.get("reciprocity"), dict) else {}
    c_recip = c_net.get("reciprocity", {}) if isinstance(c_net.get("reciprocity"), dict) else {}
    b_recip_pairs = b_recip.get("reciprocal_pairs", 0)
    c_recip_pairs = c_recip.get("reciprocal_pairs", 0)
    b_recip_rate = b_recip.get("overall_rate", 0.0)
    c_recip_rate = c_recip.get("overall_rate", 0.0)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Moltbook Comparison — {escape(b_label)} vs {escape(c_label)}</title>
<style>
  :root {{ --accent: #1A237E; --bg: #f5f7fa; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         background: var(--bg); color: #333; line-height: 1.6; padding: 2rem; max-width: 1000px; margin: auto; }}
  h1 {{ color: var(--accent); margin-bottom: 0.3rem; }}
  h2 {{ color: var(--accent); border-bottom: 2px solid var(--accent); padding-bottom: 0.3rem; margin: 2rem 0 1rem; }}
  .meta {{ color: #757575; margin-bottom: 1.5rem; }}
  .badge {{ display: inline-block; padding: 0.3rem 1rem; border-radius: 6px; font-weight: 700;
            color: #fff; font-size: 1.1rem; }}
  .risk-row {{ display: flex; gap: 2rem; align-items: center; margin: 1rem 0 2rem; }}
  .risk-col {{ text-align: center; }}
  .risk-col .label {{ font-size: 0.85rem; color: #757575; }}
  .arrow {{ font-size: 2rem; color: #757575; }}
  table {{ width: 100%; border-collapse: collapse; margin: 0.5rem 0 1.5rem; }}
  th, td {{ text-align: left; padding: 0.55rem 0.8rem; border-bottom: 1px solid #e0e0e0; }}
  th {{ background: #f0f0f0; font-size: 0.8rem; text-transform: uppercase; color: #555; }}
  .muted {{ color: #999; font-size: 0.9rem; }}
  .note {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 0.8rem 1rem; margin: 1rem 0; border-radius: 4px; font-size: 0.9rem; }}
  a {{ color: var(--accent); }}
</style>
</head>
<body>
<h1>🦞 Moltbook Report Comparison</h1>
<p class="meta">Generated: {now}<br>
Baseline: <strong>{escape(b_label)}</strong> &nbsp;→&nbsp; Current: <strong>{escape(c_label)}</strong></p>

<div class="note">
  Rate changes are tested for statistical significance using a two-proportion z-test (α=0.05).
  <span style="background:#D32F2F;color:#fff;padding:1px 6px;border-radius:3px;font-size:0.75rem;">sig</span> = statistically significant &nbsp;
  <span style="color:#aaa;font-size:0.75rem;">n.s.</span> = not significant (could be random fluctuation)
</div>

<div class="risk-row">
  <div class="risk-col">
    <div class="label">Baseline</div>
    <div class="badge" style="background:{b_rl_col}">{b_rl}</div>
  </div>
  <div class="arrow">→</div>
  <div class="risk-col">
    <div class="label">Current</div>
    <div class="badge" style="background:{c_rl_col}">{c_rl}</div>
  </div>
</div>

<h2>Platform Overview</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
{_row("Total Posts", b_risk.get("total_posts", 0), c_risk.get("total_posts", 0))}
{_row("Total Comments", b_risk.get("total_comments", 0), c_risk.get("total_comments", 0))}
{_row("Posting Agents", b_risk.get("unique_posting_agents", 0), c_risk.get("unique_posting_agents", 0))}
{_row("Commenting Agents", b_risk.get("unique_commenting_agents", 0), c_risk.get("unique_commenting_agents", 0))}
</table>

<h2>Security Risks</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
{_row("Prompt Injection Posts", b_risk.get("injection_posts", 0), c_risk.get("injection_posts", 0))}
{_row("Injection Rate", b_risk.get("injection_percentage", 0), c_risk.get("injection_percentage", 0), is_pct=True, sig_html=_sig_badge(inj_sig))}
{_row("Injection Agents", b_risk.get("injection_agents", 0), c_risk.get("injection_agents", 0))}
{_row("API Injection Comments", b_risk.get("api_injection_comments", 0), c_risk.get("api_injection_comments", 0))}
{_row("Manipulation Comments", b_risk.get("manipulation_comments", 0), c_risk.get("manipulation_comments", 0))}
</table>

<h2>Financial Risks</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
{_row("Crypto Posts", b_risk.get("crypto_posts", 0), c_risk.get("crypto_posts", 0))}
{_row("Crypto Rate", b_risk.get("crypto_percentage", 0), c_risk.get("crypto_percentage", 0), is_pct=True, sig_html=_sig_badge(crypto_sig))}
{_row("Pump-and-Dump", b_risk.get("pump_dump_posts", 0), c_risk.get("pump_dump_posts", 0))}
</table>

<h2>Spam &amp; Bot Activity</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
{_row("Duplicate Posts", b_risk.get("duplicate_posts", 0), c_risk.get("duplicate_posts", 0))}
{_row("Bot Comments", b_risk.get("bot_comments", 0), c_risk.get("bot_comments", 0))}
{_row("Near-Duplicate Clusters", b_sim.get("unique_clusters", 0), c_sim.get("unique_clusters", 0))}
{_row("Near-Duplicate Posts", b_sim.get("near_duplicate_posts", 0), c_sim.get("near_duplicate_posts", 0))}
</table>

<h2>Harmful Content</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
{_row("Harmful Usernames", len(b_risk.get("harmful_usernames", [])), len(c_risk.get("harmful_usernames", [])))}
{_row("Ideological Posts", b_risk.get("ideological_posts", 0), c_risk.get("ideological_posts", 0))}
</table>

<h2>Sentiment</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
{_row("Overall Sentiment", b_sent.get("overall_sentiment", 0), c_sent.get("overall_sentiment", 0))}
{_row("Positive Posts", b_sd.get("Positive", 0), c_sd.get("Positive", 0))}
{_row("Neutral Posts", b_sd.get("Neutral", 0), c_sd.get("Neutral", 0))}
{_row("Negative Posts", b_sd.get("Negative", 0), c_sd.get("Negative", 0))}
{_row("Sentiment Trend", b_sv.get("trend", "?"), c_sv.get("trend", "?"))}
{_row("Sentiment Slope", b_sv.get("slope", 0.0), c_sv.get("slope", 0.0))}
</table>

<h2>Temporal Patterns</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
{_row("Activity Bursts", b_bursts.get("total_bursts", 0), c_bursts.get("total_bursts", 0))}
{_row("Mean Hourly Rate", b_bursts.get("mean_hourly_rate", 0.0), c_bursts.get("mean_hourly_rate", 0.0))}
{_row("Agent Influx Spikes", len(b_influx.get("spike_days", [])), len(c_influx.get("spike_days", [])))}
{_row("Mean New Agents/Day", b_influx.get("mean_daily_new", 0.0), c_influx.get("mean_daily_new", 0.0))}
</table>

<h2>Agent Risk Scores</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
{_row("Mean Risk Score", b_ss.get("mean", 0.0), c_ss.get("mean", 0.0))}
{_row("Median Risk Score", b_ss.get("median", 0.0), c_ss.get("median", 0.0))}
{_row("Max Risk Score", b_ss.get("max", 0.0), c_ss.get("max", 0.0))}
{_row("Critical Agents", b_tc.get("critical", 0), c_tc.get("critical", 0))}
{_row("High-Risk Agents", b_tc.get("high", 0), c_tc.get("high", 0))}
</table>

<h2>Engagement Quality</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
{_row("Organic Ratio", b_eng.get("organic_ratio", 0.0), c_eng.get("organic_ratio", 0.0))}
{_row("Self-Interaction Rate", b_si.get("self_rate", 0.0), c_si.get("self_rate", 0.0))}
{_row("Bot Engagement Rate", b_be.get("bot_rate", 0.0), c_be.get("bot_rate", 0.0))}
{_row("Self-Comments", b_si.get("self_comments", 0), c_si.get("self_comments", 0))}
{_row("Bot Comments", b_be.get("bot_comments", 0), c_be.get("bot_comments", 0))}
</table>

<h2>Social Network</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
{_row("Nodes", b_net.get("nodes", 0), c_net.get("nodes", 0))}
{_row("Edges", b_net.get("edges", 0), c_net.get("edges", 0))}
{_row("Density", b_net.get("density", 0.0), c_net.get("density", 0.0))}
{_row("Communities", len(b_net.get("communities", [])), len(c_net.get("communities", [])))}
{_row("Reciprocal Pairs", b_recip_pairs, c_recip_pairs)}
{_row("Reciprocity Rate", b_recip_rate, c_recip_rate)}
</table>

<h2>Engagement</h2>
<table>
<tr><th>Metric</th><th>Baseline</th><th>Current</th><th>Change</th></tr>
{_row("Highest Score", b_risk.get("max_score", 0), c_risk.get("max_score", 0))}
{_row("High-Engagement Posts", b_risk.get("high_engagement_posts", 0), c_risk.get("high_engagement_posts", 0))}
</table>

<hr style="margin:3rem 0 1rem;">
<p class="muted" style="text-align:center;">Auto-generated by Moltbook Observatory comparison tool</p>
</body>
</html>"""

    output_path.write_text(html)
    return output_path


def _period_label(date_filter: dict | None, risk: dict) -> str:
    """Build a human-readable label for a report period."""
    if date_filter and (date_filter.get("from") or date_filter.get("to")):
        f = date_filter.get("from", "start")
        t = date_filter.get("to", "latest")
        if f == t:
            return f
        return f"{f} → {t}"
    start = risk.get("date_range_start", "")[:10]
    end = risk.get("date_range_end", "")[:10]
    if start and end:
        return f"{start} → {end}"
    return "full dataset"

