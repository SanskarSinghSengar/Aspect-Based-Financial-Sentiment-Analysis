import os
import zipfile
from pathlib import Path
import re
import json
from collections import Counter

import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

from .config import (
    SENTFIN_ZIP, DATA_DIR, ARTIFACTS_DIR,
    DF_SENTFIN_PARQUET, TRAIN_TEST_JOBLIB
)

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z0-9\s\.\,\-\:\']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_decisions(dec_str):
    if pd.isna(dec_str):
        return {}
    try:
        parsed = json.loads(dec_str)
    except Exception:
        try:
            parsed = json.loads(str(dec_str).replace("'", '"'))
        except Exception:
            return {}
    return {str(k).strip(): str(v).strip().lower() for k, v in parsed.items()}

def map_sentiment_to_id(s: str) -> int:
    return {"negative": 0, "neutral": 1, "positive": 2}.get(str(s).lower(), -1)

def derive_label_id_from_map(dec_map) -> int:
    if not dec_map:
        return -1
    c = Counter(dec_map.values())
    pos, neu, neg = c.get("positive", 0), c.get("neutral", 0), c.get("negative", 0)
    if pos == neu == neg == 0:
        return -1
    max_c = max(pos, neu, neg)
    for p in ("positive", "neutral", "negative"):
        if c.get(p, 0) == max_c:
            return map_sentiment_to_id(p)
    return -1

def load_and_prepare_sentfin(zip_path: Path = SENTFIN_ZIP):
    assert zip_path.exists(), f"ZIP not found: {zip_path}"

    extract_folder = DATA_DIR / "sentfin_extracted"
    extract_folder.mkdir(exist_ok=True, parents=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_folder)

    csvs = [f for f in os.listdir(extract_folder) if f.lower().endswith(".csv")]
    assert csvs, "No CSV found in ZIP!"
    csv_path = extract_folder / csvs[0]

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    df["text_clean"] = df["Title"].apply(clean_text)
    df["decisions_map"] = df.get("Decisions", pd.Series([None] * len(df))).apply(parse_decisions)
    df["label_id"] = df["decisions_map"].apply(derive_label_id_from_map)
    # fallback: ambiguous -> neutral
    df.loc[df["label_id"] == -1, "label_id"] = 1

    print("Label distribution:", df["label_id"].value_counts().to_dict())

    df.to_parquet(DF_SENTFIN_PARQUET)
    print(f"Saved cleaned df to {DF_SENTFIN_PARQUET}")

    # split
    mask = ~df["text_clean"].isna() & ~df["label_id"].isna()
    df = df.loc[mask].reset_index(drop=True)
    X = df["text_clean"].values
    y = df["label_id"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    joblib.dump(
        {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        TRAIN_TEST_JOBLIB
    )
    print(f"Saved train/test arrays to {TRAIN_TEST_JOBLIB}")

if __name__ == "__main__":
    load_and_prepare_sentfin()
