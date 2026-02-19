#!/usr/bin/env python3
"""
live_monitor.py

Live FeedPresser + Hybrid Model + SHAP-style XAI for a single company.

Features:
1. Hybrid FinBERT + GRU + MHSA + Topics inference (via infer_behavior_v2).
2. Entity-Specific Context Extraction (target_entity alias: "hdfc bank", "hdfc", etc.).
3. Relevance Guardrails (FlashText):
      - Filters low-value noise (local robbery, promo offers, sports, minor tech glitches)
      - Marks those as ⚠️ HUMAN REVIEW (Guardrail) with zero weight.
4. VIP Override:
      - If CEO/Board/senior management is mentioned, headline is ALWAYS processed
        (even if crime words appear), so negative governance events are captured.
5. Trusted-source weighting:
      - Only headlines from a curated list of financial / mainstream outlets
        contribute to the final consensus and SHAP (others shown but weight=0).
6. Five-class bucketing over score = (pos - neg):
      Negative, Neutral_to_Negative, Neutral, Neutral_to_Positive, Positive
7. SHAP-style headline contribution analysis and dashboard-style PNG plots.
8. Narrative summary per poll (overall tone + bucket-wise explanation).
9. Heuristics for derivatives headlines and obviously bullish phrases.

Usage (from project root):

    # Single poll, ~20 latest *relevant* headlines
    python -m src.live_monitor --company "HDFC Bank" --once

    # Continuous monitor, poll every 120 seconds
    python -m src.live_monitor --company "HDFC Bank" --poll-interval 120
"""

from __future__ import annotations

import argparse
import datetime as dt
import random
import time
import urllib.parse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- FLASHTEXT FOR O(1) GUARDRAILS ---
try:
    from flashtext import KeywordProcessor
except ImportError:
    print("[live_monitor] flashtext not found, installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "flashtext"])
    from flashtext import KeywordProcessor

from .config import PROJECT_ROOT
    # infer_behavior_v2: your Hybrid FinBERT + GRU + Attention + Topics model
from .inference import infer_behavior_v2

# Suppress noisy pandas future warnings in console
warnings.simplefilter(action="ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# Global config
# ---------------------------------------------------------------------

# We fetch a few more headlines to allow for guardrailed (0-weight) noise
TOP_FETCH = 30        # how many raw headlines to fetch from Google News
TOP_N = 20            # how many top (relevant) headlines to show / SHAP over

# SHAP approximation settings
SHAPLEY_SAMPLES = 400
SHAPLEY_MAX_ITEMS = 20

# Five-class bucketing over continuous sentiment score (pos - neg)
FIVE_BINS = [-np.inf, -0.20, -0.05, 0.05, 0.20, np.inf]
FIVE_LABELS = [
    "Negative",
    "Neutral_to_Negative",
    "Neutral",
    "Neutral_to_Positive",
    "Positive",
]

# Output folder for all live monitor artefacts
LIVE_DIR = PROJECT_ROOT / "live_results"
LIVE_DIR.mkdir(exist_ok=True, parents=True)

RUN_SUMMARY_CSV = LIVE_DIR / "run_summary.csv"

# Trusted domains: only these will move the final sentiment dial
TRUSTED_DOMAINS = {
    "economictimes.indiatimes.com",
    "livemint.com",
    "moneycontrol.com",
    "ndtv.com",
    "profit.ndtv.com",
    "business-standard.com",
    "businesstoday.in",
    "cnbctv18.com",
    "reuters.com",
    "bloomberg.com",
    "hindustantimes.com",
    "thehindu.com",
    "indianexpress.com",
    "news18.com",
    "marketsmojo.com",
    "hdfcsky.com",
    "financialexpress.com",
    "cncbtv18.com",
    # Google News aggregator (still mostly mainstream links)
    "news.google.com",
}


# ---------------------------------------------------------------------
# RELEVANCE GUARDRAIL (Noise filter with VIP override)
# ---------------------------------------------------------------------
class RelevanceGuardrail:
    """
    Filters non-financial local noise BUT allows critical governance news to pass.

    Logic:
      1. If VIP mentioned -> ALWAYS relevant (VIP_OVERRIDE).
      2. Else if noise terms matched -> mark as ⚠️ HUMAN REVIEW (Guardrail) and zero weight.
      3. Else -> relevant.
    """

    def __init__(self):
        self.noise_processor = KeywordProcessor(case_sensitive=False)
        self.vip_processor = KeywordProcessor(case_sensitive=False)
        self._build_guardrails()

    def _build_guardrails(self):
        # --- VIP TERMS: always relevant if present (CEO, MD, specific names, etc.) ---
        vips = [
            "CEO", "CFO", "CTO", "MD", "Managing Director",
            "Board of Directors", "Chairman", "Promoter",
            "Founder", "Management", "Executive",
            # add known senior names here if you want
            "Jagdishan", "Puri", "Chaudhry", "Bakhshi",
        ]
        for w in vips:
            self.vip_processor.add_keyword(w, "VIP_ENTITY")

        # --- NOISE BUCKETS (for local/operational incidents etc.) ---

        # 1) Local crime / locker incidents / counterfeit notes
        crime = [
            "robbery", "robbed", "thief", "thieves",
            "steal", "stole", "stealing", "loot", "looted", "heist",
            "arrest", "arrested", "police", "custody", "lock-up",
            "fake notes", "counterfeit",
            "murder", "gunman", "smuggling", "bribe", "bail",
            "locker", "cash van",
        ]
        for w in crime:
            self.noise_processor.add_keyword(w, "CRIME_EVENT")

        # 2) Promo / ad-like
        promo = [
            "offer", "discount", "cashback", "sale",
            "grab", "hurry", "win big", "jackpot",
            "lottery", "coupon", "voucher", "apply now",
            "credit card offer", "limited time",
        ]
        for w in promo:
            self.noise_processor.add_keyword(w, "PROMOTIONAL")

        # 3) Sports noise
        sports = [
            "cricket", "match", "tournament", "ipl", "world cup",
            "trophy", "championship", "medal", "athlete",
            "stadium", "marathon", "jersey",
        ]
        for w in sports:
            self.noise_processor.add_keyword(w, "SPORTS_NOISE")

        # 4) Minor tech glitches (ops noise)
        tech = [
            "server down", "app glitch", "login issue",
            "otp issue", "maintenance", "outage",
        ]
        for w in tech:
            self.noise_processor.add_keyword(w, "TECH_ISSUE")

    def check(self, text: str) -> (bool, str, str):
        """
        Returns:
            (is_relevant, flag_type, keys_str)

        - is_relevant = True  → send to model
        - is_relevant = False → keep but zero-weight as ⚠️ HUMAN REVIEW (Guardrail)

        flag_type:
            "VIP_OVERRIDE"   - VIP present, always relevant
            "Guardrail Hit: ..." - some noise category matched
            "OK"             - no VIP and no noise, normal
        """
        if not text:
            return True, "OK", ""

        vip_hits = self.vip_processor.extract_keywords(text)
        if vip_hits:
            # Even if crime words are present, we want to keep and score it.
            return True, "VIP_OVERRIDE", str(list(set(vip_hits)))

        noise_hits = self.noise_processor.extract_keywords(text)
        if noise_hits:
            # Local/operational/noisy events: flag as HUMAN REVIEW and zero-weight
            unique = list(set(noise_hits))
            return False, f"Guardrail Hit: {unique}", str(unique)

        return True, "OK", ""


guardrail = RelevanceGuardrail()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _safe_dt(x) -> dt.datetime:
    """Normalize various timestamp formats into a timezone-aware UTC datetime."""
    if x is None:
        return dt.datetime.now(dt.timezone.utc)
    if isinstance(x, dt.datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=dt.timezone.utc)
        return x
    try:
        return pd.to_datetime(x).to_pydatetime().replace(tzinfo=dt.timezone.utc)
    except Exception:
        try:
            return dt.datetime.strptime(str(x), "%a, %d %b %Y %H:%M:%S %Z").replace(
                tzinfo=dt.timezone.utc
            )
        except Exception:
            return dt.datetime.now(dt.timezone.utc)


def _domain(link: str) -> str:
    """Extract clean domain from a URL (no scheme, no www)."""
    try:
        if not link:
            return ""
        return urllib.parse.urlparse(link).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _consensus_from_weighted_score(wavg: float) -> str:
    """Map weighted average score into a coarse consensus label."""
    if wavg >= 0.2:
        return "Positive"
    if wavg <= -0.2:
        return "Negative"
    return "Neutral"


def weighted_avg_of_subset(indices: List[int], items: List[Dict[str, Any]]) -> float:
    """Helper for Shapley: weighted average sentiment over a subset of items."""
    s = 0.0
    w = 0.0
    for i in indices:
        it = items[i]
        sc = float(it.get("score", 0.0))
        wt = float(it.get("weight_norm", it.get("weight", 1.0) or 1.0))
        s += sc * wt
        w += wt
    return s / (w + 1e-12)


def estimate_shapley(
    items: List[Dict[str, Any]],
    value_fn,
    nsamples: int = SHAPLEY_SAMPLES,
    max_items: int = SHAPLEY_MAX_ITEMS,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Monte-Carlo Shapley approximation for a list of items.
    Each item has a score and weight; value_fn defines "portfolio value".
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = min(len(items), max_items)
    shap = np.zeros(len(items), dtype=float)

    for _ in range(nsamples):
        perm = list(range(n))
        random.shuffle(perm)
        seen: List[int] = []
        for idx in perm:
            val_before = value_fn(seen, items)
            seen = seen + [idx]
            val_after = value_fn(seen, items)
            shap[idx] += (val_after - val_before)

    shap[:n] = shap[:n] / float(nsamples)
    return shap


# ---------------------------------------------------------------------
# Fallback FeedPresser using Google News RSS
# ---------------------------------------------------------------------
def _make_item_from_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a feedparser entry into a normalized item dict."""
    title = (e.get("title") or "").strip()
    link = e.get("link", "")
    pub = e.get("published") or e.get("updated") or None

    # Time-decay weight: newer = higher weight (1 / (1 + 0.02 * age_hours))
    try:
        dtt = (
            pd.to_datetime(pub).to_pydatetime().replace(tzinfo=dt.timezone.utc)
            if pub
            else dt.datetime.now(dt.timezone.utc)
        )
        age_hours = max(0.0, (dt.datetime.now(dt.timezone.utc) - dtt).total_seconds() / 3600.0)
        weight = float(max(0.05, 1.0 / (1.0 + 0.02 * age_hours)))
    except Exception:
        weight = 0.5

    return {
        "title": title,
        "link": link,
        "published": pub,
        "weight": weight,
        "weight_norm": weight,
    }


def feed_presser(company: str, n_use: int = TOP_FETCH) -> Dict[str, Any]:
    """
    Minimal feed_presser: query Google News RSS for the company.

    Returns:
        {
          "per_item": [ { "title": ..., "link": ..., "published": ..., "weight": ... }, ... ],
          "meta": { "source": "google_news_rss_fallback" }
        }
    """
    query = urllib.parse.quote_plus(company)
    rss_url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

    d = feedparser.parse(rss_url)
    entries = d.get("entries", [])[:n_use]
    per_item = [_make_item_from_entry(e) for e in entries]
    return {"per_item": per_item, "meta": {"source": "google_news_rss_fallback"}}


# ---------------------------------------------------------------------
# Company alias helper (for entity-specific context)
# ---------------------------------------------------------------------
def _build_aliases(company_name: str) -> List[str]:
    """
    Build a small list of lowercase aliases for the company.
    e.g. "HDFC Bank Ltd" -> ["hdfc bank", "hdfc"]
    """
    name = company_name.lower()
    for suffix in ["limited", "ltd.", "ltd", "corp", "corporation", "inc.", "inc"]:
        name = name.replace(suffix, "")
    name = name.strip()
    parts = name.split()
    aliases: List[str] = []

    if len(parts) >= 2:
        aliases.append(" ".join(parts[:2]))  # 'hdfc bank'
    if parts:
        aliases.append(parts[0])             # 'hdfc'

    seen = set()
    uniq: List[str] = []
    for a in aliases:
        if a not in seen:
            uniq.append(a)
            seen.add(a)
    return uniq


# ---------------------------------------------------------------------
# Domain-specific sentiment overrides (FD offers, price ticker noise, mixed signals, options, etc.)
# ---------------------------------------------------------------------
def apply_domain_overrides(title: str, score: float, probs: Dict[str, float]):
    """
    Small rule-based patch on top of model outputs.

    Fixes:
      - Special FD / deposit offers that model sometimes treats as Negative.
      - Plain 'share price live updates / closing price' headlines that
        should usually be Neutral unless a strong or explicit move word appears.
      - Technical 'mixed indicator / mixed signals' headlines that should be more neutral.
      - Call-option-only activity treated as neutral positioning.
      - Extremely bearish put-option headlines capped.
      - Obviously bullish "outshine" style phrases nudged positive when near-neutral.
    """
    text = (title or "").lower()
    pos = float(probs.get("pos", 0.0) or 0.0)
    neu = float(probs.get("neu", 0.0) or 0.0)
    neg = float(probs.get("neg", 0.0) or 0.0)
    notes = []

    # --- 1) FD / deposit product offers: treat as mildly positive for the bank's retail franchise ---
    fd_keywords = [
        "special fd",
        "fixed deposit",
        "fd rate",
        "fd:",
        "deposit rate",
        "senior citizen deposit",
        "special deposit scheme",
    ]
    if any(k in text for k in fd_keywords) and score < 0 and score > -0.6:
        # Nudge towards mild positive instead of negative
        new_score = max(score, 0.15)
        pos = max(pos, 0.55)
        neg = min(neg, 0.15)
        neu = max(0.0, 1.0 - pos - neg)
        score = new_score
        notes.append("FD_PRODUCT_OVERRIDE")

    # --- 2) Technical 'mixed indicators / mixed signals' → clamp to neutral-ish ---
    mixed_keywords = [
        "mixed indicator",
        "mixed indicators",
        "mixed signal",
        "mixed signals",
    ]
    if any(k in text for k in mixed_keywords) and abs(score) < 0.6:
        # Treat as technical uncertainty, not a strong negative
        score = 0.0
        pos = 0.34
        neu = 0.32
        neg = 0.34
        notes.append("TECHNICAL_MIXED_CLAMP")

    # --- 3) Generic price/closing updates → clamp to neutral unless strong or explicit polarity appears ---
    price_keywords = [
        "share price live",
        "share price today",
        "share price updates",
        "closing price",
        "live updates",
        "price live updates",
    ]
    strong_move_words = [
        "surges", "soars", "jumps", "rallies", "spikes",
        "slumps", "plunges", "crashes", "tumbles",
        "hits record high", "hits record low",
    ]
    explicit_polar_words = [
        "negative return",
        "negative returns",
        "losses",
        "loss",
        "down ",
        "falls",
        "drops",
        "gains",
        "gain",
        "up ",
    ]

    if (
        any(k in text for k in price_keywords)
        and not any(w in text for w in strong_move_words)
        and not any(p in text for p in explicit_polar_words)
    ):
        # If model isn't screaming extreme, flatten to neutral-ish
        if abs(score) < 0.8:
            score = 0.0
            pos = 0.33
            neu = 0.34
            neg = 0.33
            notes.append("PRICE_NEUTRAL_CLAMP")

    # --- 4) Call-option-only activity → treat as neutral positioning, not outright bearish ---
    call_keywords = [
        "call option",
        "call options",
        "ce option",
        "ce options",
        "call oi",
        "call open interest",
        "heavy call option activity",
    ]
    put_keywords = [
        "put option",
        "put options",
        "pe option",
        "pe options",
        "put oi",
        "put open interest",
    ]
    deriv_negative_context = [
        "unwinding",
        "short covering",
        "short build-up",
        "short buildup",
        "short build up",
        "writing pressure",
        "bearish",
        "bullish",
    ]

    if (
        any(k in text for k in call_keywords)
        and not any(k in text for k in put_keywords)  # call-only, no puts
        and not any(w in text for w in deriv_negative_context)  # no strong directional words
    ):
        if abs(score) < 0.9:
            score = 0.0
            pos = 0.33
            neu = 0.34
            neg = 0.33
            notes.append("CALL_OPTION_NEUTRAL_CLAMP")

    # --- 5) Put-option-heavy headlines: keep negative, but cap extreme negativity ---
    if any(k in text for k in put_keywords):
        # If model goes too extreme negative, clamp a bit
        if score < -0.9:
            score = -0.6
            # Rescale probs gently towards a moderate negative view
            neg = max(neg, 0.55)
            pos = min(pos, 0.15)
            neu = max(0.0, 1.0 - pos - neg)
            notes.append("PUT_OPTION_NEG_CAP")

    # --- 6) Obviously bullish "outshine" style phrases, near-neutral score ---
    bullish_phrases = [
        "might outshine",
        "set to outperform",
        "poised to outperform",
        "poised to rally",
        "ready to rally",
    ]
    if any(k in text for k in bullish_phrases) and -0.1 < score < 0.2:
        score = max(score, 0.25)
        pos = max(pos, 0.55)
        neg = min(neg, 0.20)
        neu = max(0.0, 1.0 - pos - neg)
        notes.append("BULLISH_PHRASE_NUDGE")

    probs_out = {"pos": pos, "neu": neu, "neg": neg}
    note = " | ".join(notes) if notes else ""
    return score, probs_out, note


# ---------------------------------------------------------------------
# Model application: Guardrail → Entity Context → Inference
# ---------------------------------------------------------------------
def run_model_on_headlines(
    items: List[Dict[str, Any]],
    company_name: str,
) -> List[Dict[str, Any]]:
    """
    Run infer_behavior_v2 on each raw headline.

    Steps per headline:
      1) Run guardrail (VIP override + noise filter)
      2) If noise-only -> label as ⚠️ HUMAN REVIEW (Guardrail), score=0, weight=0
      3) Else -> run inference, passing target_entity alias for ABSA-style context
      4) Apply small domain overrides (FD offers, price-ticker noise, mixed signals, options)
      5) HARD FILTER: if company alias never appears in title, drop as sector/index noise.
    """
    results = []
    aliases = _build_aliases(company_name)

    for it in items:
        title = it.get("title") or it.get("headline") or it.get("text") or ""
        published = it.get("published_dt") or it.get("published") or None
        link = it.get("link", "")

        # --- 1. Guardrail check ---
        is_relevant, flag_type, keys = guardrail.check(title)

        if not is_relevant:
            # Noise-only headline: keep in CSV but neutralize its impact
            rec = {
                "title": title,
                "link": link,
                "published": published,
                "score": 0.0,
                "probs": {"pos": 0.0, "neu": 1.0, "neg": 0.0},
                "weight": 0.0,
                "weight_norm": 0.0,
                "label_raw": "⚠️ HUMAN REVIEW (Guardrail)",
                "reason": f"Guardrail: {flag_type} -> {keys}",
            }
            results.append(rec)
            continue

        # --- 2. Choose target alias for this headline (entity-specific) ---
        lower_title = title.lower()
        target_for_this: Optional[str] = None
        for a in aliases:
            if a in lower_title:
                target_for_this = a
                break

        # HARD FILTER: if company alias NEVER appears in title, drop as sector/index/peer noise
        if target_for_this is None:
            # pure sector-wide / peer-only / index-only → not counted, not shown
            continue

        # --- 3. Run model inference ---
        try:
            out = infer_behavior_v2(title, target_entity=target_for_this)
            probs = out.get("probs", {}) or {}
            pos = float(probs.get("pos", 0.0) or 0.0)
            neg = float(probs.get("neg", 0.0) or 0.0)
            neu = float(probs.get("neu", 1.0) or 1.0)
            score = pos - neg  # continuous sentiment in [-1, 1]

            # --- 4. Apply domain overrides (FD offers, price live noise, mixed signals, options, etc.) ---
            score, new_probs, override_note = apply_domain_overrides(
                title, score, {"pos": pos, "neu": neu, "neg": neg}
            )
            pos = new_probs["pos"]
            neu = new_probs["neu"]
            neg = new_probs["neg"]

            reason_text = out.get("reason", "") or ""
            if flag_type == "VIP_OVERRIDE":
                reason_text = f"[VIP EVENT: {keys}] " + reason_text
            if override_note:
                reason_text = f"[{override_note}] " + reason_text

            # Base time-decay weight from feed_presser
            wt = float(it.get("weight_norm", it.get("weight", 1.0) or 1.0))

            rec = {
                "title": title,
                "link": link,
                "published": published,
                "score": score,
                "probs": {"pos": pos, "neu": neu, "neg": neg},
                "weight": wt,
                "weight_norm": wt,
                "label_raw": out.get("label", ""),
                "reason": reason_text,
            }
        except Exception as e:
            rec = {
                "title": title,
                "link": link,
                "published": published,
                "score": 0.0,
                "probs": {"pos": 0.0, "neu": 1.0, "neg": 0.0},
                "weight": 1.0,
                "weight_norm": 1.0,
                "label_raw": "InferError",
                "reason": str(e),
            }

        results.append(rec)

    return results


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------
def _safe_slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)[:60]


def _save_plot(fig, company: str, poll_idx: int, name: str) -> Path:
    slug = _safe_slug(company)
    path = LIVE_DIR / f"{slug}_poll{poll_idx:03d}_{name}.png"
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------
# NEW: Headline bucketing + narrative summary
# ---------------------------------------------------------------------
def _bucket_headline(title: str) -> str:
    """
    Rough thematic bucket for a headline, used to build a narrative summary.
    """
    text = (title or "").lower()

    fundamentals_kw = [
        "rating", "reiterate", "reaffirms", "target price", "target of",
        "brokerage", "brokerages", "buy rating", "upgrade", "downgrade",
        "earnings", "q1", "q2", "q3", "q4", "results", "quarterly results",
        "profit", "loss", "revenue", "margin", "guidance",
        "o2c", "oil-to-chemicals", "oil to chemicals", "new energy", "capex", "investment plan",
    ]
    price_kw = [
        "share price", "stock price", "rises", "rise ", "up ", "down ",
        "falls", "jumps", "slumps", "soars", "plunges",
        "gains", "gain", "loses", "losses",
        "top gainer", "top loser", "market opening", "market close",
        "sensex", "nifty", "index", "hits", "high", "low", "52-week",
    ]
    deriv_kw = [
        "call option", "call options", "put option", "put options",
        "options", "derivatives", "open interest", "oi ", "futures",
        "expiry", "strike", "strikes",
    ]
    gov_kw = [
        "ceo", "cfo", "md", "managing director", "board", "chairman",
        "director", "governance", "resigns", "resignation", "appointed",
        "appointment", "probe", "investigation", "regulator", "penalty",
        "fine", "settlement", "scam", "fraud",
    ]

    if any(k in text for k in fundamentals_kw):
        return "Fundamentals / Earnings / Broker views"
    if any(k in text for k in deriv_kw):
        return "Derivatives / Positioning"
    if any(k in text for k in price_kw):
        return "Price action / Market move"
    if any(k in text for k in gov_kw):
        return "Management / Governance / Corporate events"
    return "Other / General news"


def _print_narrative_summary(
    company: str,
    df_top: pd.DataFrame,
    weighted_avg: float,
    consensus: str,
    pos_share: float,
    neg_share: float,
) -> None:
    """
    Print a human-style narrative summary using buckets + SHAP balance.
    """
    if df_top.empty:
        return

    # Build bucket-level stats (trusted-only weights)
    df_tmp = df_top.copy()
    total_w_eff = df_tmp["W_used"].sum()
    if total_w_eff <= 0:
        # Fall back to equal weights if everything is 0-weight
        df_tmp["W_eff"] = 1.0
    else:
        df_tmp["W_eff"] = df_tmp["W_used"]

    bucket_stats = (
        df_tmp.groupby("Bucket")
        .agg(
            count=("Title", "size"),
            weight=("W_eff", "sum"),
            avg_score=("Score", "mean"),
        )
        .sort_values("weight", ascending=False)
    )

    print("\n🧠 Narrative summary (trusted-only, top headlines):")

    # Overall tone phrase
    if weighted_avg >= 0.4:
        tone_phrase = "strongly positive"
    elif weighted_avg >= 0.2:
        tone_phrase = "moderately positive"
    elif weighted_avg > -0.2:
        tone_phrase = "mixed / sideways"
    elif weighted_avg > -0.4:
        tone_phrase = "moderately negative"
    else:
        tone_phrase = "strongly negative"

    print(
        f"   - Overall tone for {company}: {tone_phrase} "
        f"(Consensus={consensus}, Weighted Avg={weighted_avg:.3f})."
    )
    print(
        f"   - SHAP balance: {pos_share:.0%} positive vs {neg_share:.0%} negative "
        f"contribution to the sentiment dial."
    )

    # Top 3 thematic buckets
    total_weight_all = bucket_stats["weight"].sum() or 1.0
    shown = 0
    for bucket, row in bucket_stats.iterrows():
        if shown >= 3:
            break
        share = row["weight"] / total_weight_all
        avg_s = row["avg_score"]
        count_int = int(round(row["count"]))

        if avg_s >= 0.15:
            dir_word = "positive"
        elif avg_s <= -0.15:
            dir_word = "negative"
        else:
            dir_word = "mixed/neutral"

        print(
            f"   - {bucket}: {count_int} key headlines "
            f"({share:.0%} of trusted weight), overall tone {dir_word}."
        )
        shown += 1

    if len(bucket_stats) > shown:
        print(
            "   - Other buckets exist with smaller weight; treat them as background noise for now."
        )


# ---------------------------------------------------------------------
# Deduplication helper for raw headlines
# ---------------------------------------------------------------------
def _normalize_title_for_dedup(title: str) -> str:
    """
    Normalize title to deduplicate near-identical headlines.
    - Lowercase
    - Strip whitespace
    - Drop trailing " - SourceName" if present
    """
    t = (title or "").strip().lower()
    # Very common pattern: "Headline text - Moneycontrol"
    if " - " in t:
        left, _right = t.rsplit(" - ", 1)
        t = left.strip()
    return t


def deduplicate_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple deduplication based on normalized title.
    Keeps the first occurrence (usually the most recent / highest-weight in feed order).
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for it in items:
        title = it.get("title") or it.get("headline") or it.get("text") or ""
        key = _normalize_title_for_dedup(title)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


# ---------------------------------------------------------------------
# Core display + save logic for one poll
# ---------------------------------------------------------------------
def display_and_save(
    per_item: List[Dict[str, Any]],
    company: str,
    poll_idx: int,
) -> None:
    # Build DataFrame from raw items
    rows = []
    for p in per_item:
        dt_val = p.get("published") or p.get("published_dt") or None
        src_dom = _domain(p.get("link", ""))
        rows.append(
            {
                "Published_dt": _safe_dt(dt_val),
                "Title": (p.get("title", "") or "")[:400],
                "Score": float(p.get("score", 0.0)),
                "Pos": float(p.get("probs", {}).get("pos", 0.0)),
                "Neu": float(p.get("probs", {}).get("neu", 0.0)),
                "Neg": float(p.get("probs", {}).get("neg", 0.0)),
                "Weight": float(p.get("weight_norm", p.get("weight", 1.0))),
                "Source": src_dom,
                "Label": p.get("label_raw", ""),
                "Reason": p.get("reason", ""),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        print("No headlines returned.")
        return

    # Trusted-source flag
    df["Trusted"] = df["Source"].isin(TRUSTED_DOMAINS)

    # Effective weight for consensus/SHAP:
    #   - 0 for non-trusted OR HUMAN REVIEW (Guardrail)
    #   - original weight otherwise
    df["W_used"] = df["Weight"]
    df.loc[~df["Trusted"], "W_used"] = 0.0
    df.loc[df["Label"].str.contains("HUMAN REVIEW", na=False), "W_used"] = 0.0

    # Helper for top selection: prefer high effective weight, fallback to abs(score)
    df["wtmp"] = df["W_used"].replace(0, np.nan).fillna(df["Score"].abs())
    df_top = df.sort_values("wtmp", ascending=False).head(TOP_N).reset_index(drop=True)

    total_w = df_top["W_used"].sum()
    weighted_avg = (df_top["Score"] * df_top["W_used"]).sum() / (total_w + 1e-12)
    consensus = _consensus_from_weighted_score(weighted_avg)

    # Five-class bucketing
    df_top["FiveClass"] = pd.cut(df_top["Score"], bins=FIVE_BINS, labels=FIVE_LABELS)
    # Guardrailed items: force Neutral bucket for chart cleanliness
    df_top.loc[df_top["Label"].str.contains("HUMAN REVIEW", na=False), "FiveClass"] = "Neutral"

    # NEW: thematic bucket for narrative summary
    df_top["Bucket"] = df_top["Title"].apply(_bucket_headline)

    class_counts = df_top["FiveClass"].value_counts().reindex(FIVE_LABELS, fill_value=0)
    class_perc = class_counts / len(df_top) * 100.0
    class_weight_share = (
        df_top.groupby("FiveClass")["W_used"]
        .sum()
        .reindex(FIVE_LABELS, fill_value=0.0)
    )
    class_weight_share_pct = class_weight_share / (class_weight_share.sum() + 1e-12)

    # Shapley preparation (only trusted + non-guardrail items actually move)
    items_for_shap = []
    for _, r in df_top.iterrows():
        items_for_shap.append({"score": float(r["Score"]), "weight_norm": float(r["W_used"])})

    shap_values = estimate_shapley(
        items_for_shap,
        lambda inds, items: weighted_avg_of_subset(inds, items),
        nsamples=SHAPLEY_SAMPLES,
        max_items=min(SHAPLEY_MAX_ITEMS, len(items_for_shap)),
    )
    df_top["Shap"] = 0.0
    for i in range(min(len(shap_values), len(df_top))):
        df_top.at[i, "Shap"] = shap_values[i]
    abs_total = np.sum(np.abs(df_top["Shap"])) + 1e-12
    df_top["Shap_pct"] = (df_top["Shap"] / abs_total) * 100.0

    # Save snapshot CSV for this poll
    slug = _safe_slug(company)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d_%H%M%S")
    csv_snapshot = LIVE_DIR / f"{slug}_snapshot_{timestamp}_poll{poll_idx:03d}.csv"
    df_top.to_csv(csv_snapshot, index=False, encoding="utf-8")
    print(f"\n🧾 Snapshot saved: {csv_snapshot}")

    # Console summary header
    print(
        f"\n🕓 [{timestamp} UTC] Company='{company}' | "
        f"Top {len(df_top)} headlines | Weighted Avg Score={weighted_avg:.4f} | Consensus={consensus}"
    )
    print("-" * 90)
    print("📈 Five-class distribution (count | % | trusted-weight-share):")
    for c in FIVE_LABELS:
        print(
            f"  {c:>20s}: {int(class_counts[c]):2d} | "
            f"{class_perc[c]:5.1f}% | weight_share={class_weight_share_pct[c]:5.1%}"
        )

    print("-" * 90)
    print("🔎 Per-headline analysis (entity-aware, guardrail-aware, trusted-weighted):")
    for i, r in df_top.iterrows():
        title = r["Title"]
        s = r["Score"]
        w_raw = r["Weight"]
        w_used = r["W_used"]
        lbl = r["Label"]
        reason = r["Reason"] or ""
        src = r["Source"]
        dtt = r["Published_dt"].strftime("%Y-%m-%d %H:%M")
        tone = "POS" if s >= 0.2 else ("NEG" if s <= -0.2 else "NEU")

        # Small icon for immediate visual
        if "HUMAN REVIEW" in lbl:
            icon = "⚠️"  # needs human assistance
        elif not r["Trusted"]:
            icon = "❓"  # untrusted source, zero-weight
        elif "VIP EVENT" in reason:
            icon = "🤵"
        elif s >= 0.2:
            icon = "🟢"
        elif s <= -0.2:
            icon = "🔴"
        else:
            icon = "⚪"

        trust_tag = "trusted" if r["Trusted"] else "untrusted"
        print(f"\n{i+1:02d}. {icon} [{dtt}] {title}")
        print(
            f"     🏷 Source={src} ({trust_tag}) | Score={s:.3f} | "
            f"Weight_raw={w_raw:.3f} | Weight_used={w_used:.3f} | "
            f"FiveClass={r['FiveClass']} | Label={lbl} | Tone={tone}"
        )
        if reason:
            print(f"     💡 Reason: {reason}")

    # Plots: PNGs for dashboard-style visualization
    # 1) SHAP contributions (top movers)
    top_shap = df_top.sort_values("Shap", key=lambda s: np.abs(s), ascending=False).head(10)
    fig1 = plt.figure(figsize=(10, 6))
    plt.barh(range(len(top_shap)), top_shap["Shap"].values)
    plt.yticks(range(len(top_shap)), top_shap["Title"].str.slice(0, 80).values)
    plt.gca().invert_yaxis()
    plt.xlabel("Contribution to weighted avg (Δ)")
    plt.title("SHAP-like contributions — top movers (trusted-weighted)")
    shap_png = _save_plot(fig1, company, poll_idx, "shap_topmovers")
    print(f"📊 SHAP plot saved: {shap_png}")

    # 2) Five-class counts
    fig2 = plt.figure(figsize=(7, 4))
    plt.bar(FIVE_LABELS, class_counts[FIVE_LABELS].values)
    plt.title("Five-class headline counts")
    plt.ylabel("Count")
    plt.xticks(rotation=20)
    counts_png = _save_plot(fig2, company, poll_idx, "five_class_counts")
    print(f"📊 Five-class count plot saved: {counts_png}")

    # 3) Five-class weight share (trusted-effective)
    fig3 = plt.figure(figsize=(7, 4))
    plt.bar(FIVE_LABELS, class_weight_share_pct[FIVE_LABELS].values)
    plt.title("Five-class weight share (trusted-only)")
    plt.ylabel("Weight share (fraction)")
    plt.xticks(rotation=20)
    share_png = _save_plot(fig3, company, poll_idx, "five_class_weightshare")
    print(f"📊 Weight-share plot saved: {share_png}")

    # 4) Cumulative SHAP mass
    pos_sum = df_top[df_top["Shap"] > 0]["Shap"].sum()
    neg_sum = -df_top[df_top["Shap"] < 0]["Shap"].sum()
    fig4 = plt.figure(figsize=(6, 3))
    plt.bar(["Positive Shap mass", "Negative Shap mass"], [pos_sum, neg_sum])
    plt.title("Cumulative SHAP mass (absolute, trusted-only)")
    shapmass_png = _save_plot(fig4, company, poll_idx, "shap_mass")
    print(f"📊 SHAP mass plot saved: {shapmass_png}")

    # 5) Shap % top movers
    fig5 = plt.figure(figsize=(10, 5))
    sh_pct = top_shap[["Title", "Shap_pct"]].set_index("Title")["Shap_pct"]
    plt.barh(range(len(sh_pct)), sh_pct.values)
    plt.yticks(range(len(sh_pct)), [t[:80] for t in sh_pct.index])
    plt.gca().invert_yaxis()
    plt.xlabel("Percent of absolute movement (%)")
    plt.title("Shap % (top movers, trusted-only)")
    shappct_png = _save_plot(fig5, company, poll_idx, "shap_pct")
    print(f"📊 SHAP percent plot saved: {shappct_png}")

    # Attribution summary + heuristic action
    total_abs = pos_sum + neg_sum + 1e-12
    pos_share = pos_sum / total_abs
    neg_share = neg_sum / total_abs

    print("\n🔎 Attribution summary (trusted-only):")
    print(
        f"   Positive Shap mass: {pos_sum:.4f} | Negative Shap mass: {neg_sum:.4f} | "
        f"Pos share: {pos_share:.2%} | Neg share: {neg_share:.2%}"
    )

    # NEW: narrative explanation based on buckets + SHAP balance
    _print_narrative_summary(company, df_top, weighted_avg, consensus, pos_share, neg_share)

    if consensus == "Positive" and pos_share > 0.6:
        action = "BUY (Bullish bias)"
        action_reason = "Strong positive consensus & majority of Shapley mass from positive items."
    elif consensus == "Negative" and neg_share > 0.6:
        action = "SELL (Bearish bias)"
        action_reason = "Strong negative consensus & majority of Shapley mass from negative items."
    else:
        action = "WATCH / NEUTRAL"
        action_reason = "Mixed or weak consensus; monitor additional headlines and price action."

    print(f"\n📌 Suggested action for TODAY: {action}")
    print(f"   - {action_reason}")
    print("(Heuristic only; combine with your own rules.)")
    print("=" * 100)

    # ---- Append run summary row (saved EVERY poll) ----
    summary_row = {
        "timestamp_utc": timestamp,
        "company": company,
        "poll_index": poll_idx,
        "weighted_avg_score": weighted_avg,
        "consensus": consensus,
        "pos_shap_mass": float(pos_sum),
        "neg_shap_mass": float(neg_sum),
        "pos_share": float(pos_share),
        "neg_share": float(neg_share),
        "action": action,
    }
    if RUN_SUMMARY_CSV.exists():
        pd.DataFrame([summary_row]).to_csv(
            RUN_SUMMARY_CSV, mode="a", header=False, index=False
        )
    else:
        pd.DataFrame([summary_row]).to_csv(RUN_SUMMARY_CSV, index=False)
    print(f"🧾 Run summary appended to: {RUN_SUMMARY_CSV}")


# ---------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--company",
        type=str,
        required=True,
        help="Company name to monitor (e.g. 'HDFC Bank')",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between polls (ignored if --once is set).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="If set, run a single poll and exit instead of a continuous loop.",
    )
    args = parser.parse_args()

    company = args.company.strip()
    poll_interval = max(5, int(args.poll_interval))

    print(
        f"Starting live monitor for '{company}' "
        f"(interval={poll_interval}s, once={args.once}). Results saved under: {LIVE_DIR}"
    )

    poll_idx = 0
    try:
        while True:
            poll_idx += 1
            print(f"\n=== POLL #{poll_idx} for '{company}' ===")
            try:
                out = feed_presser(company, n_use=TOP_FETCH)
            except Exception as e:
                print("[live_monitor] feed_presser error:", e)
                if args.once:
                    break
                time.sleep(poll_interval)
                continue

            per_item = out.get("per_item") if isinstance(out, dict) else None
            if not per_item:
                print("No headlines returned; will try again.")
                if args.once:
                    break
                time.sleep(poll_interval)
                continue

            # Deduplicate raw feed items before scoring
            per_item_dedup = deduplicate_items(per_item)

            scored_items = run_model_on_headlines(per_item_dedup, company_name=company)
            display_and_save(scored_items, company, poll_idx)

            if args.once:
                break
            time.sleep(poll_interval)

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
