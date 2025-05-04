import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

TRAIN_DATA_DIR = os.path.join(DATA_DIR, 'MINDsmall_train')
DEV_DATA_DIR = os.path.join(DATA_DIR, 'MINDsmall_dev')

NEWS_TSV = 'news.tsv'
BEHAVIORS_TSV = 'behaviors.tsv'

PROCESSED_NEWS_PKL = os.path.join(MODEL_DIR, 'processed_news.pkl')
USER_HISTORY_PKL = os.path.join(MODEL_DIR, 'user_history.pkl')
TFIDF_VECTORIZER_PKL = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
TFIDF_MATRIX_PKL = os.path.join(MODEL_DIR, 'tfidf_matrix.pkl')
RANKER_MODEL_PKL = os.path.join(MODEL_DIR, 'lgbm_ranker.pkl')

# --- Preprocessing ---
TEXT_COLS = ['title', 'abstract']
TARGET_TEXT_COL = 'text'
MAX_TFIDF_FEATURES = 5000

# --- Candidate Generation ---
CG_CONTENT_TOP_K = 100 # Top K similar items for content-based CG
CG_POPULARITY_TOP_M = 100 # Top M popular items for popularity-based CG
CG_FINAL_SIZE = 200 # Approximate size after combining candidates

# --- Feature Engineering ---
RANKING_NUM_NEG_SAMPLES = 4 # Number of negative samples per positive click

# --- Ranker Training ---
LGBM_PARAMS = {
    'objective': 'binary', # or 'lambdarank' if using LGBMRanker with groups
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'n_estimators': 100,
    'random_state': 42,
    'n_jobs': -1
}

# --- API ---
DEFAULT_RANKING_SIZE = 10

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)