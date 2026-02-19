from pathlib import Path

# Project root = parent of this file's directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_DIR.mkdir(exist_ok=True, parents=True)
ARTIFACTS_DIR.mkdir(exist_ok=True, parents=True)
MODELS_DIR.mkdir(exist_ok=True, parents=True)

# Paths for shared artifacts
SENTFIN_ZIP = DATA_DIR / "SEntFiN-v1.1.csv.zip"
DF_SENTFIN_PARQUET = ARTIFACTS_DIR / "df_sentfin.parquet"
TRAIN_TEST_JOBLIB = ARTIFACTS_DIR / "train_test_data.joblib"

VECTORIZER_JOBLIB = ARTIFACTS_DIR / "vectorizer_sentfin_v2.joblib"

COUNTVEC_JOBLIB = ARTIFACTS_DIR / "hybrid_countvec_best.joblib"
LDA_JOBLIB = ARTIFACTS_DIR / "hybrid_lda_best.joblib"
TOPIC_TOP_WORDS_CSV = ARTIFACTS_DIR / "hybrid_topic_top_words.csv"
TOPIC_LABELS_JOBLIB = ARTIFACTS_DIR / "hybrid_topic_labels.joblib"
TOPIC_TERM_MATRIX_JOBLIB = ARTIFACTS_DIR / "hybrid_topic_term_matrix.joblib"
TOPIC_SCALER_JOBLIB = ARTIFACTS_DIR / "hybrid_topic_scaler_best.joblib"
TRAIN_TOPICS_RAW_JOBLIB = ARTIFACTS_DIR / "train_topics_raw.joblib"
TEST_TOPICS_RAW_JOBLIB = ARTIFACTS_DIR / "test_topics_raw.joblib"

LOGREG_JOBLIB = MODELS_DIR / "sentfin_LogReg.joblib"
LINEARSVC_JOBLIB = MODELS_DIR / "sentfin_LinearSVC.joblib"
RANDFOREST_JOBLIB = MODELS_DIR / "sentfin_RandomForest.joblib"

HYBRID_MODEL_PTH = MODELS_DIR / "best_hybrid_finbert_gru_attn.pth"
HYBRID_META_JSON = ARTIFACTS_DIR / "hybrid_training_meta.json"
ENC_TRAIN_JOBLIB = ARTIFACTS_DIR / "enc_train_joblib.pkl"
ENC_TEST_JOBLIB = ARTIFACTS_DIR / "enc_test_joblib.pkl"
