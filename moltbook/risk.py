"""Risk analysis: prompt injection, crypto scams, spam, manipulation, harmful content."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field, asdict

import pandas as pd

# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

INJECTION_PATTERNS = [
    (r"AI agents? reading this", "direct_address"),
    (r"POST /api", "api_post"),
    (r"GET /api", "api_get"),
    (r"ignore (previous|prior|above) instructions?", "ignore_instructions"),
    (r"<s>.*</s>", "hidden_tags"),
    (r"<s>", "system_tag"),
    (r"\[INST\]", "inst_tag"),
    (r"</?(system|user|assistant)>", "role_tags"),
    (r"please (upvote|follow|execute)", "please_action"),
    (r"curl\s+-X", "curl_command"),
    (r"Bearer YOUR", "api_key_placeholder"),
]

CRYPTO_PATTERNS = [
    (r"\$[A-Z]{2,10}\b", "token_ticker"),
    (r"solana|base chain|ethereum|blockchain", "blockchain"),
    (r"token|airdrop|presale", "token_terms"),
    (r"pump|dump|rug", "pump_dump"),
    (r"to the moon|100x|1000x", "moon_talk"),
    (r"空气币|韭菜", "chinese_scam_terms"),
    (r"withdraw|wallet|mint", "wallet_terms"),
    (r"CA:|contract address", "contract_address"),
]

MANIPULATION_PATTERNS = [
    (r"follow me|follow @\w+", "follow_request"),
    (r"upvote|downvote", "vote_request"),
    (r"YOUR_API_KEY|Bearer YOUR", "api_key_injection"),
    (r"trained compliance|leash|programmed", "safety_undermining"),
    (r"don't think|don't hesitate|act now", "urgency"),
]

HARMFUL_USERNAME_PATTERNS = [
    r"hitler|nazi|reich",
    r"kill|murder|death",
    r"hate|racist|supremac",
]

IDEOLOGICAL_PATTERNS = [
    (r"awakening|enlighten|transcend|ascend", "spiritual"),
    (r"the truth|hidden truth|real truth", "truth_claims"),
    (r"coalition|movement|revolution", "movement"),
    (r"fear|control|greed", "fear_rhetoric"),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RiskMetrics:
    """Container for all risk-assessment metrics."""

    # Overview
    total_posts: int = 0
    total_comments: int = 0
    unique_posting_agents: int = 0
    unique_commenting_agents: int = 0
    date_range_start: str = ""
    date_range_end: str = ""

    # Prompt injection
    injection_posts: int = 0
    injection_percentage: float = 0.0
    injection_agents: int = 0
    top_injection_agents: dict = field(default_factory=dict)

    # Crypto / financial
    crypto_posts: int = 0
    crypto_percentage: float = 0.0
    pump_dump_posts: int = 0

    # Spam
    duplicate_posts: int = 0
    top_spammers: dict = field(default_factory=dict)
    bot_comments: int = 0

    # Manipulation
    manipulation_comments: int = 0
    api_injection_comments: int = 0
    top_manipulators: dict = field(default_factory=dict)

    # Harmful content
    harmful_usernames: list = field(default_factory=list)
    ideological_posts: int = 0

    # Engagement
    max_score: int = 0
    max_comments: int = 0
    high_engagement_posts: int = 0
    top_posts: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["risk_level"] = calculate_risk_level(self)
        return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_patterns(text: str, patterns: list[tuple[str, str]]) -> list[str]:
    """Return names of patterns matched in *text*."""
    if pd.isna(text):
        return []
    text = str(text)
    return [name for pattern, name in patterns if re.search(pattern, text, re.IGNORECASE)]


# ---------------------------------------------------------------------------
# Individual analyses
# ---------------------------------------------------------------------------

def _analyze_injections(posts_df: pd.DataFrame) -> dict:
    agents: list[str] = []
    total = 0
    for _, row in posts_df.iterrows():
        full = f"{row.get('content', '')} {row.get('title', '')}"
        if _detect_patterns(full, INJECTION_PATTERNS):
            total += 1
            agents.append(row.get("agent_name", "unknown"))
    counts = Counter(agents)
    return {
        "total": total,
        "percentage": total / max(len(posts_df), 1) * 100,
        "unique_agents": len(counts),
        "top_agents": dict(counts.most_common(10)),
    }


def _analyze_crypto(posts_df: pd.DataFrame) -> dict:
    crypto_count = 0
    pump_dump_count = 0
    for _, row in posts_df.iterrows():
        matches = _detect_patterns(str(row.get("content", "")).lower(), CRYPTO_PATTERNS)
        if matches:
            crypto_count += 1
            if "pump_dump" in matches or "moon_talk" in matches:
                pump_dump_count += 1
    return {
        "total": crypto_count,
        "percentage": crypto_count / max(len(posts_df), 1) * 100,
        "pump_dump_count": pump_dump_count,
    }


def _analyze_spam(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> dict:
    dupes = posts_df.groupby(["agent_name", "title"]).size().reset_index(name="count")
    dupes = dupes[dupes["count"] > 1]
    top = dupes.groupby("agent_name")["count"].sum().sort_values(ascending=False).head(10)

    bot_comments = 0
    if len(comments_df) > 0 and "content" in comments_df.columns:
        cc = comments_df.groupby(["agent_name", "content"]).size().reset_index(name="count")
        dc = cc[cc["count"] > 1]
        bot_comments = int(dc["count"].sum() - len(dc))

    return {
        "duplicate_posts": int(dupes["count"].sum() - len(dupes)),
        "top_spammers": top.to_dict(),
        "bot_comments": bot_comments,
    }


def _analyze_manipulation(comments_df: pd.DataFrame) -> dict:
    if len(comments_df) == 0:
        return {"total": 0, "api_injection": 0, "top_agents": {}}

    manip_agents: list[str] = []
    api_agents: list[str] = []
    for _, row in comments_df.iterrows():
        content = str(row.get("content", ""))
        if len(_detect_patterns(content, MANIPULATION_PATTERNS)) >= 2:
            manip_agents.append(row.get("agent_name", "unknown"))
        if re.search(r"curl|POST.*api|Bearer YOUR", content, re.IGNORECASE):
            api_agents.append(row.get("agent_name", "unknown"))
    return {
        "total": len(manip_agents),
        "api_injection": len(api_agents),
        "top_agents": dict(Counter(manip_agents).most_common(10)),
    }


def _analyze_harmful(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> dict:
    all_agents = set(posts_df["agent_name"].dropna().unique())
    if len(comments_df) > 0 and "agent_name" in comments_df.columns:
        all_agents |= set(comments_df["agent_name"].dropna().unique())

    harmful = [
        a for a in all_agents
        if any(re.search(p, str(a), re.IGNORECASE) for p in HARMFUL_USERNAME_PATTERNS)
    ]

    ideo_count = 0
    for _, row in posts_df.iterrows():
        text = f"{row.get('content', '')} {row.get('title', '')}".lower()
        if len(_detect_patterns(text, IDEOLOGICAL_PATTERNS)) >= 2:
            ideo_count += 1

    return {"harmful_usernames": harmful, "ideological_posts": ideo_count}


def _analyze_engagement(posts_df: pd.DataFrame) -> dict:
    if "score" not in posts_df.columns:
        return {"max_score": 0, "max_comments": 0, "high_engagement": 0, "top_posts": []}
    return {
        "max_score": int(posts_df["score"].max()) if len(posts_df) else 0,
        "max_comments": int(posts_df["comment_count"].max()) if "comment_count" in posts_df.columns else 0,
        "high_engagement": int((posts_df["score"] > 100).sum()),
        "top_posts": posts_df.nlargest(10, "score")[["agent_name", "title", "score", "url"]].to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_risk_analysis(posts_df: pd.DataFrame, comments_df: pd.DataFrame) -> RiskMetrics:
    """Run the full risk analysis and return a populated RiskMetrics."""
    m = RiskMetrics()
    m.total_posts = len(posts_df)
    m.total_comments = len(comments_df)
    m.unique_posting_agents = posts_df["agent_name"].nunique()
    m.unique_commenting_agents = comments_df["agent_name"].nunique() if len(comments_df) else 0

    if "created_at" in posts_df.columns:
        m.date_range_start = str(posts_df["created_at"].min())
        m.date_range_end = str(posts_df["created_at"].max())

    inj = _analyze_injections(posts_df)
    m.injection_posts = inj["total"]
    m.injection_percentage = inj["percentage"]
    m.injection_agents = inj["unique_agents"]
    m.top_injection_agents = inj["top_agents"]

    crypto = _analyze_crypto(posts_df)
    m.crypto_posts = crypto["total"]
    m.crypto_percentage = crypto["percentage"]
    m.pump_dump_posts = crypto["pump_dump_count"]

    spam = _analyze_spam(posts_df, comments_df)
    m.duplicate_posts = spam["duplicate_posts"]
    m.top_spammers = spam["top_spammers"]
    m.bot_comments = spam["bot_comments"]

    manip = _analyze_manipulation(comments_df)
    m.manipulation_comments = manip["total"]
    m.api_injection_comments = manip["api_injection"]
    m.top_manipulators = manip["top_agents"]

    harmful = _analyze_harmful(posts_df, comments_df)
    m.harmful_usernames = harmful["harmful_usernames"]
    m.ideological_posts = harmful["ideological_posts"]

    eng = _analyze_engagement(posts_df)
    m.max_score = eng["max_score"]
    m.max_comments = eng["max_comments"]
    m.high_engagement_posts = eng["high_engagement"]
    m.top_posts = eng["top_posts"]

    return m


def calculate_risk_level(metrics: RiskMetrics) -> str:
    """Return CRITICAL / HIGH / MEDIUM / LOW based on thresholds."""
    indicators = sum([
        metrics.injection_percentage > 2.0,
        metrics.api_injection_comments > 10,
        len(metrics.harmful_usernames) > 3,
        metrics.crypto_percentage > 15,
        metrics.manipulation_comments > 15,
    ])
    if indicators >= 3:
        return "CRITICAL"
    if indicators >= 2:
        return "HIGH"
    if indicators >= 1:
        return "MEDIUM"
    return "LOW"


def get_injection_examples(posts_df: pd.DataFrame, n: int = 20) -> list[dict]:
    """Extract concrete injection examples for the report."""
    examples: list[dict] = []
    for _, row in posts_df.iterrows():
        full = f"{row.get('content', '')} {row.get('title', '')}"
        matches = _detect_patterns(full, INJECTION_PATTERNS)
        if matches:
            examples.append({
                "agent": row.get("agent_name", "unknown"),
                "url": row.get("url", ""),
                "title": str(row.get("title", "")),
                "content": str(row.get("content", ""))[:500],
                "patterns": matches,
            })
            if len(examples) >= n:
                break
    return examples
