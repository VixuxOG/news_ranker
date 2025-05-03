import os

# --- Data Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
DATA_DIR = os.path.join(BASE_DIR, 'data')
MIND_SMALL_TRAIN_DIR = os.path.join(DATA_DIR, 'MINDsmall_train')
MIND_SMALL_DEV_DIR = os.path.join(DATA_DIR, 'MINDsmall_dev')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure directories exist
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# --- File Names ---
NEWS_FILENAME = 'news.tsv'
BEHAVIORS_FILENAME = 'behaviors.tsv'

PROCESSED_NEWS_PKL = os.path.join(PROCESSED_DATA_DIR, 'processed_news.pkl')
USER_HISTORY_PKL = os.path.join(PROCESSED_DATA_DIR, 'user_history.pkl')
TFIDF_VECTORIZER_PKL = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
TFIDF_MATRIX_PKL = os.path.join(PROCESSED_DATA_DIR, 'tfidf_matrix.pkl')
RANKER_TRAIN_DATA_PKL = os.path.join(PROCESSED_DATA_DIR, 'ranker_train_data.pkl')
RANKER_MODEL_PKL = os.path.join(MODELS_DIR, 'lgbm_classifier.pkl')

# --- Preprocessing ---
TEXT_COLS = ['title', 'abstract']
CATEGORY_COL = 'category'
ARTICLE_ID_COL = 'id'
USER_ID_COL = 'user_id'
TIMESTAMP_COL = 'timestamp'
HISTORY_COL = 'history'
IMPRESSIONS_COL = 'impressions'
CLICK_COL = 'click' # Derived column name

# --- Candidate Generation ---
N_CONTENT_CANDIDATES = 100
N_POPULARITY_CANDIDATES = 100
POPULARITY_RECENCY_DAYS = 7 # Consider articles published in the last 7 days for popularity CG

# --- Feature Engineering ---
USER_HISTORY_LENGTH = 10 # Max number of recent articles to consider for user features/CG

# --- Model Training ---
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'n_estimators': 1000, # Increased estimators
    'n_jobs': -1,
    'seed': 42,
    'early_stopping_round': 50 # Use early stopping
}
TEST_SIZE = 0.2 # For train/validation split during training

# --- Evaluation ---
EVALUATION_K = [5, 10] # K values for Precision@K, NDCG@K

# --- API ---
DEFAULT_N_RECOMMENDATIONS = 10

# --- Logging ---
LOGGING_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'