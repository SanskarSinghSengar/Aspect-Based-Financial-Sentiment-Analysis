#!/usr/bin/env python3
"""
gold_eval.py

Offline evaluation / analysis for the hybrid FinBERT + GRU + MHSA + Topic model.

Two modes:

1) FULL GOLD EVAL (with manual labels)
   - CSV must have:
       * a text column (default: 'headline')
       * a label column (default: 'label')
   - Gold labels in {neg, neu, pos} or {Negative, Neutral, Positive} (case-insensitive)
     are mapped to:
         0 = Negative
         1 = Neutral
         2 = Positive
   - Output:
       * Confusion matrix
       * Precision / Recall / F1

2) SNAPSHOT ANALYSIS (no gold labels)
   - If the specified label column is NOT present, we fall back to "analysis-only" mode.
   - This is designed to work directly on live_results snapshots from live_monitor.
   - Output:
       * Count of predicted labels (Negative / Neutral / Positive)
       * Mean probabilities for each class
       * (No confusion matrix or classification report, since no gold truth)

BONUS:
   --auto-live    → automatically pick the latest live_results/*snapshot*.csv
                    (no need to type any path)

Usage (from project root):

  # A) True evaluation with manually labelled CSV:
  python -m src.gold_eval --csv data/manual_gold_eval.csv \
      --text-col headline --label-col label

  # B) Analysis-only directly on a specific live_results snapshot:
  python -m src.gold_eval --csv live_results/HDFC_Bank_snapshot_20251119_122348_poll001.csv \
      --text-col Title

  # C) Easiest: auto-analyse the latest snapshot (no path needed!)
  python -m src.gold_eval --auto-live

  # Optional ABSA-style target entity:
  python -m src.gold_eval --auto-live --company "HDFC Bank"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from .inference import infer_behavior_v2
from .config import PROJECT_ROOT

# ---------------------------------------------------------------------
# Gold label normalisation
# ---------------------------------------------------------------------

LABEL_MAP: Dict[str, int] = {
    "neg": 0,
    "negative": 0,
    "neu": 1,
    "neutral": 1,
    "pos": 2,
    "positive": 2,
}


def _normalize_gold_label(x: str) -> int:
    s = str(x).strip().lower()
    if s not in LABEL_MAP:
        raise ValueError(f"Unknown gold label: {x!r}")
    return LABEL_MAP[s]


# ---------------------------------------------------------------------
# Mode A: Full gold evaluation (manual labels)
# ---------------------------------------------------------------------

def _run_full_gold_eval(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    target_entity: Optional[str],
) -> None:
    """Proper evaluation when gold label column is present."""
    texts: List[str] = df[text_col].astype(str).tolist()
    gold_raw = df[label_col].tolist()
    gold = np.array([_normalize_gold_label(x) for x in gold_raw], dtype=int)

    preds: List[int] = []
    for t in texts:
        out = infer_behavior_v2(t, target_entity=target_entity)
        preds.append(int(out["pred"]))
    preds_arr = np.array(preds, dtype=int)

    print("\n=== GOLD EVAL REPORT (with manual labels) ===")
    print(f"Samples: {len(texts)}")
    print("\nLabel mapping used:")
    print("  0 = Negative, 1 = Neutral, 2 = Positive\n")

    print("Confusion Matrix (rows=gold, cols=pred):\n")
    print(confusion_matrix(gold, preds_arr))

    print("\nClassification Report:\n")
    print(
        classification_report(
            gold,
            preds_arr,
            target_names=["Negative", "Neutral", "Positive"],
            digits=4,
        )
    )


# ---------------------------------------------------------------------
# Mode B: Snapshot analysis (no gold labels)
# ---------------------------------------------------------------------

def _run_snapshot_analysis(
    df: pd.DataFrame,
    text_col: str,
    target_entity: Optional[str],
) -> None:
    """
    Analysis-only when NO gold label column is present.

    - Re-runs infer_behavior_v2 on each row.
    - Prints distribution of predicted labels and average probabilities.
    - No accuracy metrics (we have no gold truth).
    """
    texts: List[str] = df[text_col].astype(str).tolist()

    preds: List[int] = []
    probs_neg: List[float] = []
    probs_neu: List[float] = []
    probs_pos: List[float] = []

    for t in texts:
        out = infer_behavior_v2(t, target_entity=target_entity)
        preds.append(int(out["pred"]))
        p = out.get("probs", {})
        probs_neg.append(float(p.get("neg", 0.0)))
        probs_neu.append(float(p.get("neu", 0.0)))
        probs_pos.append(float(p.get("pos", 0.0)))

    preds_arr = np.array(preds, dtype=int)

    label_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    counts = {k: int((preds_arr == k).sum()) for k in [0, 1, 2]}
    total = len(texts)

    print("\n=== SNAPSHOT ANALYSIS (no gold labels) ===")
    print(f"Samples: {total}")
    print(f"Text column: {text_col}")
    if target_entity:
        print(f"Target entity (ABSA context): {target_entity}")
    else:
        print("Target entity: None (headline-level sentiment)")

    print("\nPredicted label distribution:")
    for k in [0, 1, 2]:
        name = label_names[k]
        c = counts[k]
        pct = (c / total * 100.0) if total > 0 else 0.0
        print(f"  {k} = {name:8s}: {c:4d} ({pct:5.1f}%)")

    print("\nAverage predicted probabilities (over all samples):")
    print(f"  neg: {np.mean(probs_neg):.4f}")
    print(f"  neu: {np.mean(probs_neu):.4f}")
    print(f"  pos: {np.mean(probs_pos):.4f}")

    print("\nNote:")
    print("  - This mode does NOT compute accuracy/F1 because no gold labels are present.")
    print("  - It is meant for quickly analysing a live_results snapshot or any unlabeled CSV.")


# ---------------------------------------------------------------------
# Helper: auto-pick latest live_results snapshot
# ---------------------------------------------------------------------

def _resolve_csv_path(args_csv: Optional[str], auto_live: bool) -> Path:
    """
    Decide which CSV to use:
      - If auto_live is True → latest live_results/*snapshot*.csv
      - Else → args_csv (relative to PROJECT_ROOT if not absolute)
    """
    if auto_live:
        live_dir = PROJECT_ROOT / "live_results"
        if not live_dir.exists():
            raise FileNotFoundError(f"[gold_eval] live_results folder not found at: {live_dir}")

        snapshot_files = sorted(
            live_dir.glob("*_snapshot_*.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not snapshot_files:
            raise FileNotFoundError(
                f"[gold_eval] No snapshot CSVs found in {live_dir}. "
                "Run live_monitor at least once first."
            )
        latest = snapshot_files[0]
        print(f"[gold_eval] Auto-selected latest snapshot: {latest}")
        return latest

    # Manual path
    if args_csv is None:
        raise ValueError(
            "[gold_eval] Either --csv must be provided or --auto-live must be set."
        )

    csv_path = (PROJECT_ROOT / args_csv) if not Path(args_csv).is_absolute() else Path(args_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"[gold_eval] CSV not found at: {csv_path}")
    return csv_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to CSV file. Can be a manual gold file or a live_results snapshot.",
    )
    parser.add_argument(
        "--auto-live",
        action="store_true",
        help="If set, automatically use the latest live_results/*snapshot*.csv and ignore --csv.",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default="headline",
        help="Name of the text column in the CSV (e.g. 'headline' or 'Title').",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help=(
            "Name of the gold label column in the CSV. "
            "If this column is missing, the script falls back to snapshot analysis mode."
        ),
    )
    parser.add_argument(
        "--company",
        type=str,
        default=None,
        help="Optional company name; if provided, its first token is used as target_entity.",
    )
    args = parser.parse_args()

    # Decide which CSV file to use
    csv_path = _resolve_csv_path(args.csv, auto_live=args.auto_live)

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"\n[gold_eval] Loaded CSV from: {csv_path}")
    print(f"[gold_eval] Columns: {list(df.columns)}")

    # Text column handling: try user text-col, otherwise smart fallback
    text_col = args.text_col
    if text_col not in df.columns:
        # Smart fallback: try Title (for live_results) or headline
        if "Title" in df.columns:
            print(
                f"[gold_eval] text column '{text_col}' not found; "
                "falling back to 'Title'."
            )
            text_col = "Title"
        elif "headline" in df.columns:
            print(
                f"[gold_eval] text column '{text_col}' not found; "
                "falling back to 'headline'."
            )
            text_col = "headline"
        else:
            raise ValueError(
                f"Text column '{text_col}' not found and couldn't find 'Title' or 'headline'. "
                f"Available columns: {list(df.columns)}"
            )

    label_col = args.label_col

    # Optional target entity alias from company name
    target_entity = None
    if args.company:
        target_entity = args.company.split()[0]

    # Decide evaluation mode based on presence of label_col
    if label_col in df.columns:
        # FULL evaluation
        _run_full_gold_eval(df, text_col=text_col, label_col=label_col, target_entity=target_entity)
    else:
        # SNAPSHOT analysis (no gold labels)
        print(
            f"\n[gold_eval] label column '{label_col}' not found. "
            "Running SNAPSHOT ANALYSIS mode (no accuracy / F1)."
        )
        _run_snapshot_analysis(df, text_col=text_col, target_entity=target_entity)


if __name__ == "__main__":
    main()
