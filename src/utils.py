import time
import pandas as pd
import joblib
from . import config

# Example utility function (can add more as needed)
def timer(func):
    """Decorator to time function execution."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def load_processed_data():
    """Loads all necessary processed data files for inference."""
    try:
        processed_news = joblib.load(config.PROCESSED_NEWS_PKL)
        user_history = joblib.load(config.USER_HISTORY_PKL)
        vectorizer = joblib.load(config.TFIDF_VECTORIZER_PKL)
        tfidf_matrix = joblib.load(config.TFIDF_MATRIX_PKL)
        ranker_model = joblib.load(config.RANKER_MODEL_PKL)
        # Create news lookup dict and newsid->index mapping
        news_lookup = processed_news.set_index('news_id').to_dict('index')
        newsid_to_idx = {news_id: idx for idx, news_id in enumerate(processed_news['news_id'])}

        print("Loaded all necessary data and models for inference.")
        return processed_news, user_history, vectorizer, tfidf_matrix, ranker_model, news_lookup, newsid_to_idx
    except FileNotFoundError as e:
        print(f"Error loading processed data: {e}. Please run training script first.")
        return None, None, None, None, None, None, None
    except Exception as e:
         print(f"An unexpected error occurred during data loading: {e}")
         return None, None, None, None, None, None, None