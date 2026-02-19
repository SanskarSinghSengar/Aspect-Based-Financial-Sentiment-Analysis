#!/usr/bin/env python3
"""
inference.py

Hybrid FinBERT + GRU + Multi-Head Attention + Topic model inference utilities.

- Loads trained hybrid model from disk.
- Computes topic features via CountVectorizer + LDA + StandardScaler if available.
- Applies updated behaviour logic with asymmetric thresholds + lexicon boost.
- Supports target-specific context extraction (per-company sentiment).
- Has domain rules for bond yields (yields↑ + equities under pressure).
- Main entrypoint: infer_behavior_v2(text, target_entity=None) -> dict

CLI usage:

  # Simple overall sentiment
  python -m src.inference "HDFC Bank shares rally after strong Q2 earnings"

  # Per-entity sentiment (e.g., for HDFC inside a mixed headline)
  python -m src.inference "LIC lifts stakes in SBI; cuts exposure in HDFC Bank" HDFC
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer  # via hybrid_model

from .hybrid_model import HybridFinModel, load_tokenizer_and_backbone

# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

# Project root: .../FINANCIALSENTIMENTANALYSIS
PROJ_ROOT = Path(__file__).resolve().parent.parent

# Where all artifacts live
ARTIFACTS_DIR = PROJ_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = PROJ_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model + meta paths
HYBRID_MODEL_PTH = MODELS_DIR / "best_hybrid_finbert_gru_attn.pth"
HYBRID_META_JSON = ARTIFACTS_DIR / "hybrid_meta.json"

# Topic artifacts (created by your LDA/topic pipeline)
COUNTVEC_JOBLIB = ARTIFACTS_DIR / "hybrid_countvec_best.joblib"
LDA_JOBLIB = ARTIFACTS_DIR / "hybrid_lda_best.joblib"
TOPIC_SCALER_JOBLIB = ARTIFACTS_DIR / "hybrid_topic_scaler_best.joblib"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These should match your training config
MAX_LEN = 64
GRU_HIDDEN = 256
DROPOUT = 0.3

# ---------------------------------------------------------------------
# Updated thresholds / lexicon settings (tuned based on manual audit)
# ---------------------------------------------------------------------
# Lowered from 0.10 to 0.055 to catch "marginal gains" (prob ~0.058)
POS_THRESH = 0.055

# Slightly more sensitive on the negative side
NEG_THRESH = 0.05

# Directed score deltas (how much stronger pos/neg must be vs neu)
DELTA_POS = 0.08
DELTA_NEG = 0.20

# Lexicon boosts (small additive nudges before renormalization)
LEXICON_POS_BOOST = 0.02
LEXICON_NEG_BOOST = 0.05

TEMPERATURE = 1.0
TOP_K_TOKENS = 12

# ---------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------

POS_CUES = {
    "gain", "gains", "inch", "up", "rise", "rises", "rally", "firm",
    "add", "adds", "lift", "lifts", "surge", "surges", "better", "beats",
    "beat", "soars", "jump", "jumps", "higher", "robust", "strong",
    "pickup", "inflow", "inflows", "wins", "win", "buy", "buys",
}

# Added "volatility", "clouds", "cloud", "uncertainty", "cut", "cuts", etc.
NEG_CUES = {
    "fall", "falls", "dip", "dips", "down", "sell", "selling", "pressure",
    "weak", "weakness", "slump", "slumps", "slips", "slip", "drop", "drops",
    "concern", "concerns", "caution", "cautious", "worries", "worry", "hit",
    "lower", "plunge", "plunges", "ease", "eases", "headwind", "headwinds",
    "warn", "warns", "warning", "warnings", "softening", "soften", "muted",
    "flag", "flags", "drag", "drags", "cut", "cuts", "loser", "loss",
    "volatility", "clouds", "cloud", "uncertainty",
}

# ---------------------------------------------------------------------
# Global lazy-loaded objects
# ---------------------------------------------------------------------
_tokenizer = None
_backbone = None
_model = None
_topic_cv = None
_topic_lda = None
_topic_scaler = None


# ---------------------------------------------------------------------
# Topic utilities
# ---------------------------------------------------------------------
def _load_topics_artifacts():
    """
    Lazy-load CountVectorizer, LDA and topic scaler if available.
    """
    global _topic_cv, _topic_lda, _topic_scaler
    if _topic_cv is not None and _topic_lda is not None and _topic_scaler is not None:
        return

    try:
        import joblib

        if COUNTVEC_JOBLIB.exists() and LDA_JOBLIB.exists() and TOPIC_SCALER_JOBLIB.exists():
            _topic_cv = joblib.load(COUNTVEC_JOBLIB)
            _topic_lda = joblib.load(LDA_JOBLIB)
            _topic_scaler = joblib.load(TOPIC_SCALER_JOBLIB)
            print(f"[inference] Loaded topic artifacts from {ARTIFACTS_DIR}")
        else:
            print("[inference] Topic artifacts not found; using zero topic vector fallback.")
            _topic_cv = _topic_lda = _topic_scaler = None
    except Exception as e:
        print("[inference] Failed to load topic artifacts, using fallback zeros:", e)
        _topic_cv = _topic_lda = _topic_scaler = None


def compute_topic_vector_one(text: str) -> np.ndarray:
    """
    Compute a single (1, n_topics) topic vector for the given text.
    If topic artifacts are unavailable, returns zeros of reasonable shape.
    """
    _load_topics_artifacts()

    if _topic_cv is not None and _topic_lda is not None and _topic_scaler is not None:
        try:
            vec = _topic_cv.transform([text])
            topic_raw = _topic_lda.transform(vec)
            topic_scaled = _topic_scaler.transform(topic_raw)
            return topic_scaled.astype(float)
        except Exception as e:
            print("[inference] Topic vector computation failed, using zeros:", e)

    # Fallback: zeros, try to match known topic dimension
    n_topics = 20
    if _topic_scaler is not None:
        try:
            n_topics = _topic_scaler.mean_.shape[0]
        except Exception:
            pass
    return np.zeros((1, n_topics), dtype=float)


# ---------------------------------------------------------------------
# Token cleaning for attention display
# ---------------------------------------------------------------------
def clean_token(tok: str) -> str:
    """
    Clean a token for attention display:
    - strip GPT/BPE artefacts
    - drop special tokens
    """
    if not tok:
        return ""
    t = tok.replace("Ġ", "").replace("##", "").strip()
    if not t:
        return ""
    if t.upper() in {"[CLS]", "[SEP]", "[PAD]", "<S>", "</S>", "<PAD>", "[UNK]", "<UNK>"}:
        return ""
    return t


# ---------------------------------------------------------------------
# Lexicon boost with yield-awareness
# ---------------------------------------------------------------------
def lexicon_boost(text: str) -> Tuple[float, float]:
    """
    Return (pos_boost, neg_boost) small additive values based on keyword matches.

    Includes domain-specific logic:
      - If 'yield' appears and it's described as rising (rise/jump/higher/up/surge),
        treat that as NEGATIVE for equities (higher bond yields hurt stocks).
    """
    s = "" if text is None else str(text).lower()
    pos_boost = 0.0
    neg_boost = 0.0

    # SPECIAL LOGIC: Bond Yields
    # If "yield" is present, "rise/higher/jump" is actually NEGATIVE for stocks.
    has_yield = "yield" in s

    for w in POS_CUES:
        if re.search(r"\b" + re.escape(w) + r"\b", s):
            if has_yield and w in {"rise", "rises", "jump", "jumps", "surge", "surges", "higher", "up"}:
                # Flip: yields going up is bad for equities
                neg_boost += LEXICON_NEG_BOOST
            else:
                pos_boost += LEXICON_POS_BOOST

    for w in NEG_CUES:
        if re.search(r"\b" + re.escape(w) + r"\b", s):
            neg_boost += LEXICON_NEG_BOOST

    # clamp to avoid overpowering model probabilities
    pos_boost = min(pos_boost, 0.12)
    neg_boost = min(neg_boost, 0.20)
    return pos_boost, neg_boost


# ---------------------------------------------------------------------
# Temperature scaling
# ---------------------------------------------------------------------
def apply_temperature_scale(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    if temperature is None or temperature == 1.0:
        return logits
    return logits / float(temperature)


# ---------------------------------------------------------------------
# Target-specific context extraction (entity-aware)
# ---------------------------------------------------------------------
def extract_relevant_context(text: str, target: str | None) -> str:
    """
    Try to isolate the clause in `text` that talks about `target`.

    Example:
      "LIC lifts stakes in SBI, Coal India; cuts exposure in HDFC Bank, ICICI Bank"
        -> target="HDFC"  -> "cuts exposure in HDFC Bank, ICICI Bank"

      "SBI vs HDFC vs Kotak: Q2 results reveal one clear winner (and one big loser)"
        -> target="HDFC"  -> "HDFC vs Kotak: Q2 results reveal one clear winner (and one big loser)"
    """
    if not target:
        return text

    lower = text.lower()
    target_lower = target.lower()

    idx = lower.find(target_lower)
    if idx == -1:
        # target not mentioned – keep full text (fallback)
        return text

    # Delimiters that often separate ideas in headlines
    delim_tokens = [
        ";",
        "|",
        "—",
        " - ",
        " but ",
        " however ",
        " while ",
        " whereas ",
        " vs ",
        ":",
    ]

    # --- find previous cut position (start of our slice) ---
    prev_cut = 0
    for d in delim_tokens:
        pos = lower.rfind(d, 0, idx)
        if pos != -1:
            prev_cut = max(prev_cut, pos + len(d))

    # --- find next cut position (end of our slice) ---
    next_cut = len(text)
    for d in delim_tokens:
        pos = lower.find(d, idx + len(target_lower))
        if pos != -1:
            next_cut = min(next_cut, pos)

    segment = text[prev_cut:next_cut].strip()

    # If the segment is too short (e.g. only "HDFC"),
    # expand to a character window around the target.
    if len(segment) < 20:
        window_before = 40
        window_after = 80
        start = max(0, idx - window_before)
        end = min(len(text), idx + window_after)
        segment = text[start:end].strip()

    # FINAL fallback: if for some reason we still got almost nothing, use full text
    if len(segment) < 5:
        return text

    return segment


# ---------------------------------------------------------------------
# Model & tokenizer loader
# ---------------------------------------------------------------------
def load_inference_model(device: torch.device | None = None):
    """
    Lazy-load tokenizer, backbone and HybridFinModel with trained weights.
    """
    global _tokenizer, _backbone, _model

    if device is None:
        device = DEVICE

    if _model is not None:
        return _tokenizer, _model

    # Load tokenizer + backbone using the shared util
    _tokenizer, _backbone = load_tokenizer_and_backbone(device, unfreeze_last_k=0)

    # Determine topics dimension
    _load_topics_artifacts()
    if _topic_scaler is not None:
        topics_dim = _topic_scaler.mean_.shape[0]
    else:
        topics_dim = 20  # safe default

    # Instantiate hybrid model
    hidden_size = _backbone.config.hidden_size
    _model = HybridFinModel(
        backbone=_backbone,
        hidden_size=hidden_size,
        topics_dim=topics_dim,
        num_classes=3,
        gru_hidden=GRU_HIDDEN,
        dropout=DROPOUT,
    ).to(device)

    # Load weights
    if HYBRID_MODEL_PTH.exists():
        state = torch.load(HYBRID_MODEL_PTH, map_location=device)
        _model.load_state_dict(state)
        _model.eval()
        print(f"[inference] Loaded hybrid model from {HYBRID_MODEL_PTH}")
    else:
        raise FileNotFoundError(f"Model weights not found at {HYBRID_MODEL_PTH}")

    return _tokenizer, _model


# ---------------------------------------------------------------------
# Core inference: infer_behavior_v2
# ---------------------------------------------------------------------
def infer_behavior_v2(
    text: str,
    target_entity: str | None = None,   # NEW: target-specific context
    pos_thresh: float = POS_THRESH,
    neg_thresh: float = NEG_THRESH,
    delta_pos: float = DELTA_POS,
    delta_neg: float = DELTA_NEG,
    temp: float = TEMPERATURE,
    use_lexicon: bool = True,
    top_k: int = TOP_K_TOKENS,
) -> Dict[str, Any]:
    """
    Main behaviour-aware inference function.

    Returns:
      {
        "label": "Positive behaviour" / "Negative behaviour" /
                 "Neutral_to_Positive" / "Neutral_to_Negative" /
                 "Neutral behaviour",
        "pred":  0/1/2  (hard class: neg, neu, pos),
        "probs": {"neg": float, "neu": float, "pos": float},
        "reason": str (short explanation),
        "top_tokens": [str, ...],
        "processed_text": str,
        "original_text": str,
        "target_entity": str | None
      }
    """
    original_text = text
    if target_entity:
        text = extract_relevant_context(original_text, target_entity)
    processed_text = text

    tokenizer, model = load_inference_model(DEVICE)
    model.eval()

    # Tokenize
    toks = tokenizer(
        processed_text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt",
    )
    ids = toks["input_ids"].to(DEVICE)
    mask = toks["attention_mask"].to(DEVICE)

    # Topic vector
    topic_vec = compute_topic_vector_one(processed_text)
    topic_t = torch.tensor(topic_vec, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        logits, att_w = model(ids, mask, topic_t)
        logits = apply_temperature_scale(logits, temperature=temp)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())

        # order: 0=neg, 1=neu, 2=pos  (same as training)
        neg_p, neu_p, pos_p = float(probs[0]), float(probs[1]), float(probs[2])

        # Lexicon tiny boost (with yield-aware logic)
        found_cues: List[str] = []
        if use_lexicon:
            lb_pos, lb_neg = lexicon_boost(processed_text)
            lower_txt = processed_text.lower()
            for w in POS_CUES.union(NEG_CUES):
                if w in lower_txt:
                    found_cues.append(w)

            pos_p = min(0.9999, pos_p + lb_pos)
            neg_p = min(0.9999, neg_p + lb_neg)
            s = pos_p + neu_p + neg_p
            pos_p, neu_p, neg_p = pos_p / s, neu_p / s, neg_p / s

        # Attention top tokens (for Multi-Head Attention we get [B, S, S])
        att_matrix = att_w[0].cpu().numpy()  # (S, S)
        att = att_matrix.mean(axis=0)        # per-token importance
        token_list = tokenizer.convert_ids_to_tokens(toks["input_ids"][0].cpu().numpy())
        valid_idx = [j for j, t in enumerate(token_list) if clean_token(t) != ""]
        if not valid_idx:
            valid_idx = list(range(len(token_list)))
        sorted_idx = sorted(valid_idx, key=lambda j: float(att[j]), reverse=True)[:top_k]
        top_tokens = [clean_token(token_list[j]) for j in sorted_idx if clean_token(token_list[j]) != ""]

    # -----------------------------------------------------------------
    # Behaviour label logic (5-class meta label using 3-class backbone)
    # -----------------------------------------------------------------
    label: str
    reason: str

    # Hard positive / negative dominate
    if pred == 2:
        label = "Positive behaviour"
        if found_cues:
            reason = f"Model hard-predicted Positive; cues={found_cues}"
        else:
            reason = "Model hard-predicted Positive"
    elif pred == 0:
        label = "Negative behaviour"
        if found_cues:
            reason = f"Model hard-predicted Negative; cues={found_cues}"
        else:
            reason = "Model hard-predicted Negative"
    else:
        # Neutral base class -> apply directional thresholds
        S_pos = pos_p - neu_p  # positive support vs neutral
        S_neg = neg_p - neu_p  # negative support vs neutral

        # Absolute thresholds first
        if neg_p >= neg_thresh:
            label = "Neutral_to_Negative"
            reason = f"neg_prob ({neg_p:.3f}) >= neg_thresh ({neg_thresh:.3f})"
        elif pos_p >= pos_thresh:
            label = "Neutral_to_Positive"
            reason = f"pos_prob ({pos_p:.3f}) >= pos_thresh ({pos_thresh:.3f})"
        else:
            # Directional judgement if absolute thresholds fail
            if (S_neg > 0 and (S_neg >= S_pos) and (S_neg >= -delta_neg * 0.01 or S_neg >= -delta_neg)):
                label = "Neutral_to_Negative"
                reason = f"S_neg ({S_neg:.3f}) dominated or close to neu"
            elif (S_pos > 0 and (S_pos > S_neg) and (S_pos >= -delta_pos * 0.01 or S_pos >= -delta_pos)):
                label = "Neutral_to_Positive"
                reason = f"S_pos ({S_pos:.3f}) dominated or close to neu"
            else:
                label = "Neutral behaviour"
                reason = f"Neutral: pos={pos_p:.3f}, neu={neu_p:.3f}, neg={neg_p:.3f}"

        if found_cues:
            reason += f"; cues={found_cues}"

    # -----------------------------------------------------------------
    # HARD DOMAIN OVERRIDE: “yields higher + equities under pressure”
    # -----------------------------------------------------------------
    lower_txt = processed_text.lower()
    if (
        "yield" in lower_txt
        and any(w in lower_txt for w in ["equities", "stocks", "shares"])
        and any(w in lower_txt for w in ["higher", "rise", "rises", "jump", "jumps", "surge", "surges", "up"])
        and any(w in lower_txt for w in ["pressure", "selloff", "sell-off", "outflows"])
        and neg_p >= 0.15
        and label == "Positive behaviour"
    ):
        # Override strong positive into at least Neutral_to_Negative
        label = "Neutral_to_Negative"
        reason += " [override: rising bond yields + equity pressure interpreted as headwind]"

    return {
        "label": label,
        "pred": pred,
        "probs": {
            "neg": round(neg_p, 4),
            "neu": round(neu_p, 4),
            "pos": round(pos_p, 4),
        },
        "reason": reason,
        "top_tokens": top_tokens,
        "processed_text": processed_text,
        "original_text": original_text,
        "target_entity": target_entity,
    }


# ---------------------------------------------------------------------
# Convenience: batch analysis
# ---------------------------------------------------------------------
def analyze_headlines(headlines: List[str]) -> List[Dict[str, Any]]:
    """
    Run infer_behavior_v2 over a list of headlines and return list of dicts.
    (Used by the CLI demo below.)
    """
    results = []
    for i, h in enumerate(headlines, start=1):
        out = infer_behavior_v2(h)
        results.append(
            {
                "idx": i,
                "text": h,
                "label": out["label"],
                "pred": out["pred"],
                "neg": out["probs"]["neg"],
                "neu": out["probs"]["neu"],
                "pos": out["probs"]["pos"],
                "reason": out["reason"],
                "top_tokens": " ".join(out["top_tokens"]),
            }
        )
    return results


# ---------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # First arg = full headline (use quotes in shell)
        text = sys.argv[1]
        target = sys.argv[2] if len(sys.argv) > 2 else None

        print(f"[CLI] Text: {text}")
        if target:
            print(f"[CLI] Target entity: {target}")
        print()

        out = infer_behavior_v2(text, target_entity=target)
        print(json.dumps(out, indent=2))
    else:
        # quick demo with a few sample headlines
        samples = [
            "HDFC Bank shares rally after strong Q2 earnings beat estimates",
            "Markets plunge as rate hike fears intensify and global cues weaken",
            "Sensex ends flat as investors remain cautious ahead of RBI policy",
            "Auto sales slump 20% as demand softens; companies cut production",
            "Housing starts stable; permits tick up slightly in key cities",
            "Inflation surprise pushes bond yields higher, equities under pressure",
            "Volatility clouds outlook for banking stocks",
            "LIC lifts stakes in SBI; cuts exposure in HDFC Bank",
        ]
        res = analyze_headlines(samples)
        for r in res:
            print(f"{r['idx']:02d}. {r['text']}")
            print(
                f"   -> {r['label']} | probs=(neg={r['neg']:.3f}, "
                f"neu={r['neu']:.3f}, pos={r['pos']:.3f})"
            )
            print(f"   Reason: {r['reason']}")
            print(f"   Tokens: {r['top_tokens']}\n")
