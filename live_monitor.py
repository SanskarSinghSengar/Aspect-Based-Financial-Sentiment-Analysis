import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    ARTIFACTS_DIR, TRAIN_TEST_JOBLIB,
    COUNTVEC_JOBLIB, LDA_JOBLIB,
    TOPIC_TOP_WORDS_CSV, TOPIC_LABELS_JOBLIB,
    TOPIC_TERM_MATRIX_JOBLIB,
    TRAIN_TOPICS_RAW_JOBLIB, TEST_TOPICS_RAW_JOBLIB
)

def clean_token(tok: str):
    if tok is None:
        return None
    tok = tok.strip().lower()
    if re.search(r"\d", tok):
        return None
    if len(tok) <= 1:
        return None
    if re.match(r"^[^\w]+$", tok):
        return None
    return tok

def run_lda(n_topics: int = 20, n_top_words: int = 12, rep_docs_per_topic: int = 5):
    data = joblib.load(TRAIN_TEST_JOBLIB)
    X_train, X_test = np.array(data["X_train"]), np.array(data["X_test"])

    all_texts = np.concatenate([X_train, X_test])
    n_docs = len(all_texts)

    CV_MAX_FEATURES = 2000
    CV_MIN_DF = 5
    if n_docs < 200:
        CV_MIN_DF = max(1, int(CV_MIN_DF * (n_docs / 200)))
        CV_MAX_FEATURES = max(500, int(CV_MAX_FEATURES * (n_docs / 200)))

    cv = CountVectorizer(
        max_features=CV_MAX_FEATURES,
        ngram_range=(1, 2),
        min_df=CV_MIN_DF,
        stop_words="english",
    )

    print(f"Fitting CountVectorizer on {n_docs} docs...")
    cv_fit = cv.fit_transform(all_texts)
    if cv_fit.shape[1] == 0:
        raise RuntimeError("Empty vocabulary: lower min_df or check texts.")

    print(f"Fitting LDA with {n_topics} topics...")
    try:
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1)
    except TypeError:
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

    lda.fit(cv_fit)

    train_cv = cv.transform(X_train)
    test_cv = cv.transform(X_test)
    train_topics_raw = lda.transform(train_cv)
    test_topics_raw = lda.transform(test_cv)

    joblib.dump(train_topics_raw, TRAIN_TOPICS_RAW_JOBLIB)
    joblib.dump(test_topics_raw, TEST_TOPICS_RAW_JOBLIB)

    topic_strength = train_topics_raw.sum(axis=0)
    plt.figure(figsize=(12, 4))
    sns.barplot(x=np.arange(len(topic_strength)) + 1, y=topic_strength)
    plt.title("Topic strength (sum over train set)")
    plt.xlabel("Topic #"); plt.ylabel("Sum probability")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "topic_strength.png")
    plt.close()

    feature_names = cv.get_feature_names_out()
    topic_top_words = []
    for idx, topic in enumerate(lda.components_):
        top_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_feats = []
        for i in top_idx:
            tok = clean_token(feature_names[i])
            if tok:
                top_feats.append(tok)
        topic_top_words.append(top_feats)
        print(f"Topic {idx+1}: {', '.join(top_feats)}")

    top_words_df = pd.DataFrame({
        "topic": np.arange(1, len(topic_top_words) + 1),
        "top_words": [", ".join(w) for w in topic_top_words],
    })
    top_words_df.to_csv(TOPIC_TOP_WORDS_CSV, index=False)

    topic_labels = {i: " / ".join(w[:2]) if w else f"Topic-{i+1}" for i, w in enumerate(topic_top_words)}

    topic_term_matrix = lda.components_ / lda.components_.sum(axis=1)[:, None]

    joblib.dump(cv, COUNTVEC_JOBLIB)
    joblib.dump(lda, LDA_JOBLIB)
    joblib.dump(topic_top_words, ARTIFACTS_DIR / "hybrid_topic_top_words.joblib")
    joblib.dump(topic_labels, TOPIC_LABELS_JOBLIB)
    joblib.dump(topic_term_matrix, TOPIC_TERM_MATRIX_JOBLIB)

    corr = cosine_similarity(topic_term_matrix)
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, cmap="viridis",
                xticklabels=np.arange(1, corr.shape[0]+1),
                yticklabels=np.arange(1, corr.shape[0]+1))
    plt.title("Topic correlation (cosine similarity)")
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "topic_corr.png")
    plt.close()

    print("Saved LDA artifacts to", ARTIFACTS_DIR)

if __name__ == "__main__":
    run_lda()
