## 🏁 Project Setup

### 1. Clone / open project

```powershell
cd C:\Users\Pulki\FINANCIALSENTIMENTANALYSIS
```

### 2. Create + activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

> **Important**
> All commands below assume:
>
> * You are in: `C:\Users\Pulki\FINANCIALSENTIMENTANALYSIS`
> * Virtual env is active: prompt shows `(.venv)`

---

## 📦 Data & Config

* Dataset ZIP **must** be at:

```text
C:\Users\Pulki\FINANCIALSENTIMENTANALYSIS\data\SEntFiN-v1.1.csv.zip
```

* In `src/config.py`:

```python
SENTFIN_ZIP = DATA_DIR / "SEntFiN-v1.1.csv.zip"
```

Do **not** change this unless you rename the file.

---

## 1️⃣ `data_prep.py` — Prepare SentFiN dataset

**Script:** `src/data_prep.py`
**What it does:**

* Extracts the ZIP
* Cleans text (`Title` → `text_clean`)
* Parses `Decisions` JSON → `label_id` (0/1/2)
* Saves:

  * `artifacts/df_sentfin.parquet`
  * `artifacts/train_test_data.joblib`

### Run (standard)

```powershell
python -m src.data_prep
```

### When to re-run

* You change the ZIP
* You change label parsing / cleaning logic
* You want a fresh train/test split

---

## 2️⃣ `lda_topics.py` — Topic model (CountVectorizer + LDA)

**Script:** `src/lda_topics.py`
**What it does:**

* Loads `train_test_data.joblib`
* Fits CountVectorizer + LDA over all texts
* Computes train/test topic distributions
* Saves:

  * `artifacts/hybrid_countvec_best.joblib`
  * `artifacts/hybrid_lda_best.joblib`
  * `artifacts/hybrid_topic_scaler_best.joblib`
  * `artifacts/hybrid_topic_top_words.joblib`
  * `artifacts/hybrid_topic_labels.joblib`
  * `artifacts/hybrid_topic_term_matrix.joblib`
  * `artifacts/train_topics_raw.joblib`
  * `artifacts/test_topics_raw.joblib`
  * `artifacts/topic_strength.png`
  * `artifacts/topic_corr.png`

### Run

```powershell
python -m src.lda_topics
```

### When to re-run

* You changed `data_prep` logic
* You want a different number of topics (change `run_lda(n_topics=...)`)

---

## 3️⃣ `tfidf_baselines.py` — Classic baseline models

**Script:** `src/tfidf_baselines.py`
**What it does:**

* TF–IDF (1–2 grams) on `text_clean`
* Trains:

  * Logistic Regression
  * Linear SVM
  * Random Forest
* Saves:

  * `artifacts/vectorizer_sentfin_v2.joblib`
  * `models/sentfin_LogReg.joblib`
  * `models/sentfin_LinearSVC.joblib`
  * `models/sentfin_RandomForest.joblib`
  * Confusion matrices as PNG + CSV
  * Classification reports (.txt)
  * Summary JSON + CSV

### Run

```powershell
python -m src.tfidf_baselines
```

### Outputs (per model)

* Confusion matrices:

  * `artifacts/cm_LogReg.png`, `cm_LogReg.csv`
  * `artifacts/cm_LinearSVC.png`, `cm_LinearSVC.csv`
  * `artifacts/cm_RandomForest.png`, `cm_RandomForest.csv`
* Reports:

  * `artifacts/classification_LogReg.txt`
  * `artifacts/classification_LinearSVC.txt`
  * `artifacts/classification_RandomForest.txt`
* Summary:

  * `artifacts/baselines_summary.json`
  * `artifacts/baselines_summary.csv`

---

## 4️⃣ `train_hybrid.py` — Hybrid FinBERT + BiGRU + MHSA + Topics

**Script:** `src/train_hybrid.py`
**What it does:**

* Loads:

  * `train_test_data.joblib`
  * `train_topics_raw.joblib` / `test_topics_raw.joblib`
* Fits/loads StandardScaler on topics
* Tokenizes with FinBERT tokenizer
* Builds `HybridFinModel`:

  * FinBERT backbone
  * BiGRU
  * Multi-Head Self-Attention
  * Topic fusion + MLP head
* Trains with Focal Loss + class weights
* Saves:

  * `models/best_hybrid_finbert_gru_attn.pth`
  * `artifacts/enc_train_joblib.pkl`, `enc_test_joblib.pkl`
  * Final confusion matrix (PNG + CSV)
  * Classification report
  * Meta JSON with metrics

### Run (full training)

```powershell
python -m src.train_hybrid
```

### Output metrics

* Confusion matrix:

  * `artifacts/cm_hybrid_final.png`
  * `artifacts/cm_hybrid_final.csv`
* Classification report:

  * `artifacts/hybrid_classification_report.txt`
* Meta:

  * `artifacts/hybrid_training_meta.json`

The meta JSON includes (example keys):

```json
{
  "device": "cuda or cpu",
  "max_len": 64,
  "epochs": 6,
  "best_f1_macro": 0.83,
  "final_test_acc": 0.85,
  "final_test_f1_macro": 0.84,
  "train_batches_per_epoch": ...,
  "train_batch_size": 16,
  "eval_batch_size": 32,
  "total_train_time_sec": ...,
  "gru_hidden": 256,
  "dropout": 0.3,
  "unfreeze_last_k": 4
}
```

### When to re-run

* You updated topics (LDA)
* You changed model architecture or thresholds
* You want to retrain with more epochs (change `EPOCHS` constant)

---

## 5️⃣ `inference.py` — Single-headline inference (CLI)

**Script:** `src/inference.py`
**Main entrypoint:** `infer_behavior_v2`
**What it does (CLI mode):**

* Loads trained hybrid model + topic artifacts
* Runs inference on a single headline
* Optional: target-entity specific context

### Simple CLI usage (overall sentiment)

```powershell
python -m src.inference "HDFC Bank shares rally after strong Q2 earnings"
```

### Entity-specific sentiment

```powershell
python -m src.inference "LIC lifts stakes in SBI; cuts exposure in HDFC Bank" HDFC
```

This prints a JSON dict:

* `label` (e.g., `"Positive behaviour"`, `"Neutral_to_Negative"`)
* `pred` (0/1/2)
* `probs` (neg/neu/pos)
* `reason`
* `top_tokens`
* `processed_text`
* `original_text`
* `target_entity`

---

## 6️⃣ `live_monitor.py` — Live Google News monitor + SHAP-like analysis

**Script:** `src/live_monitor.py`
**What it does:**

* Fetches latest headlines from Google News RSS for a company
* Applies:

  * Guardrails (noise filter with VIP override)
  * Hybrid model + entity-context extraction
* Computes:

  * Weighted average sentiment
  * Five-class distribution
  * Approx SHAP-style contributions per headline
* Produces:

  * Console summary
  * CSV snapshot of top headlines
  * PNG plots
  * Run summary CSV

### A. Single poll (one-time snapshot)

```powershell
python -m src.live_monitor --company "HDFC Bank" --once
```

### B. Continuous monitoring (every N seconds)

```powershell
python -m src.live_monitor --company "HDFC Bank" --poll-interval 120
```

> Press `Ctrl + C` to stop.

### Outputs

* Snapshots for each poll:

  ```text
  live_results/HDFC_Bank_snapshot_YYYYMMDD_HHMMSS_pollNNN.csv
  ```

* Plots per poll:

  * `*_pollNNN_shap_topmovers.png`
  * `*_pollNNN_five_class_counts.png`
  * `*_pollNNN_five_class_weightshare.png`
  * `*_pollNNN_shap_mass.png`
  * `*_pollNNN_shap_pct.png`

* Run summary (appended every poll):

  * `live_results/run_summary.csv`

---

## 7️⃣ `gold_eval.py` — Offline evaluation & snapshot analysis

**Script:** `src/gold_eval.py`
**Two main modes:**

### Mode A — Full evaluation with manual gold labels

Input CSV must have:

* Text column (e.g. `headline` or `Title`)
* Gold label column (e.g. `label`) with values in:

  * `{neg, neu, pos}` or `{Negative, Neutral, Positive}` (case-insensitive)

Example command:

```powershell
python -m src.gold_eval `
  --csv data/manual_gold_eval.csv `
  --text-col headline `
  --label-col label `
  --company "HDFC Bank"
```

Outputs (printed):

* Confusion matrix
* Classification report

### Mode B — Snapshot analysis (no gold labels)

You can directly analyse a `live_results` snapshot or any unlabeled CSV.

Example with explicit CSV:

```powershell
python -m src.gold_eval `
  --csv live_results/HDFC_Bank_snapshot_20251119_122348_poll001.csv `
  --text-col Title `
  --company "HDFC Bank"
```

Example using `--auto-live` (auto-pick latest snapshot):

```powershell
python -m src.gold_eval --auto-live --company "HDFC Bank"
```

This prints:

* Predicted label distribution
* Average probabilities for neg/neu/pos
* No accuracy/F1 (no gold labels)

---

