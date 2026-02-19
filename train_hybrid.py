#!/usr/bin/env python3
"""
tfidf_baselines.py

Train classic TF-IDF baselines on SentFiN:

  - Logistic Regression
  - Linear SVM
  - Random Forest

Saves:
  - TF-IDF vectorizer
  - models/*.joblib
  - confusion matrices (PNG + CSV)
  - classification reports (.txt)
  - baselines_summary.json / .csv
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from .config import (
    ARTIFACTS_DIR,
    TRAIN_TEST_JOBLIB,
    VECTORIZER_JOBLIB,
    LOGREG_JOBLIB,
    LINEARSVC_JOBLIB,
    RANDFOREST_JOBLIB,
    MODELS_DIR,
)


def train_baselines() -> None:
    # Ensure dirs exist
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = joblib.load(TRAIN_TEST_JOBLIB)
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    # TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=3,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    joblib.dump(vectorizer, VECTORIZER_JOBLIB)
    print(f"Saved TF-IDF vectorizer to {VECTORIZER_JOBLIB}")

    models = {
        "LogReg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "LinearSVC": LinearSVC(
            max_iter=20000,
            class_weight="balanced",
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
    }

    results = {}
    metrics_rows = []

    for name, model in models.items():
        print("=" * 60)
        print(f"Training {name}...")

        model.fit(X_train_tfidf, y_train)
        preds = model.predict(X_test_tfidf)

        acc = accuracy_score(y_test, preds)
        f1m = f1_score(y_test, preds, average="macro")
        report = classification_report(
            y_test,
            preds,
            target_names=["Neg", "Neu", "Pos"],
            digits=4,
        )

        print(f"{name}: acc={acc:.4f} macro-F1={f1m:.4f}")
        print(report)

        # Confusion matrix
        cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])
        cm_df = pd.DataFrame(
            cm,
            index=["Neg", "Neu", "Pos"],
            columns=["Neg", "Neu", "Pos"],
        )

        cm_csv_path = ARTIFACTS_DIR / f"cm_{name}.csv"
        cm_df.to_csv(cm_csv_path)
        print(f"Saved {name} confusion matrix CSV to {cm_csv_path}")

        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Neg", "Neu", "Pos"],
            yticklabels=["Neg", "Neu", "Pos"],
        )
        plt.title(f"{name} — Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_png_path = ARTIFACTS_DIR / f"cm_{name}.png"
        plt.savefig(cm_png_path)
        plt.close()
        print(f"Saved {name} confusion matrix PNG to {cm_png_path}")

        # Save classification report
        rpt_path = ARTIFACTS_DIR / f"classification_{name}.txt"
        with open(rpt_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved {name} classification report to {rpt_path}")

        # Save model
        out_path = {
            "LogReg": LOGREG_JOBLIB,
            "LinearSVC": LINEARSVC_JOBLIB,
            "RandomForest": RANDFOREST_JOBLIB,
        }[name]
        joblib.dump(model, out_path)
        print(f"Saved {name} model to {out_path}")

        results[name] = (acc, f1m)
        metrics_rows.append(
            {
                "model": name,
                "accuracy": float(acc),
                "f1_macro": float(f1m),
            }
        )

    # Summary JSON + CSV
    summary_json = ARTIFACTS_DIR / "baselines_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(metrics_rows, f, indent=2)
    print(f"Saved baselines summary JSON to {summary_json}")

    summary_csv = ARTIFACTS_DIR / "baselines_summary.csv"
    pd.DataFrame(metrics_rows).to_csv(summary_csv, index=False)
    print(f"Saved baselines summary CSV to {summary_csv}")

    print("Results:", results)


if __name__ == "__main__":
    train_baselines()
