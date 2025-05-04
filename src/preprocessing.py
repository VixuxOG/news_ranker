import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from . import config
import numpy as np

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    print("NLTK stopwords not found. Downloading...")
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    """Basic text cleaning: lowercase, remove non-alphanumeric, remove stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text) # Keep spaces
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def preprocess_news(news_df):
    """Applies cleaning and combines text columns."""
    print("Preprocessing news data...")
    processed_df = news_df.copy()
    # Fill NaN values in text columns before cleaning
    for col in config.TEXT_COLS:
        processed_df[col] = processed_df[col].fillna('')

    processed_df[config.TARGET_TEXT_COL] = processed_df[config.TEXT_COLS].apply(lambda x: ' '.join(x), axis=1)
    processed_df[config.TARGET_TEXT_COL] = processed_df[config.TARGET_TEXT_COL].apply(clean_text)

    # Add a simple popularity score (using index as proxy - higher index = newer/more popular assumption)
    processed_df['popularity_score'] = processed_df.index / len(processed_df)

    print("News preprocessing complete.")
    return processed_df[['news_id', 'category', 'subcategory', config.TARGET_TEXT_COL, 'popularity_score']]

def build_tfidf(news_df, text_column=config.TARGET_TEXT_COL):
    """Builds TF-IDF vectorizer and matrix."""
    print("Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(max_features=config.MAX_TFIDF_FEATURES, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(news_df[text_column])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Save the vectorizer and matrix
    joblib.dump(vectorizer, config.TFIDF_VECTORIZER_PKL)
    joblib.dump(tfidf_matrix, config.TFIDF_MATRIX_PKL)
    print(f"TF-IDF vectorizer saved to {config.TFIDF_VECTORIZER_PKL}")
    print(f"TF-IDF matrix saved to {config.TFIDF_MATRIX_PKL}")

    return vectorizer, tfidf_matrix

def build_user_history(behaviors_df):
    """Builds a dictionary of user click history."""
    print("Building user history...")
    user_history = {}
    # Fill NaN in history before splitting
    behaviors_df['history'] = behaviors_df['history'].fillna('')

    for _, row in behaviors_df.iterrows():
        user_id = row['user_id']
        history_clicks = row['history'].split() # Articles clicked in the past
        # Also consider clicks within the current impression log for history
        impressions = row['impressions'].split()
        current_clicks = [imp.split('-')[0] for imp in impressions if imp.endswith('-1')]

        all_clicks = list(set(history_clicks + current_clicks)) # Unique clicks

        if user_id not in user_history:
            user_history[user_id] = []
        # Keep history relatively recent/manageable if needed (optional)
        user_history[user_id].extend(all_clicks)
        # Keep only unique clicks in history
        user_history[user_id] = list(set(user_history[user_id]))

    # Save user history
    joblib.dump(user_history, config.USER_HISTORY_PKL)
    print(f"User history built for {len(user_history)} users.")
    print(f"User history saved to {config.USER_HISTORY_PKL}")
    return user_history

if __name__ == '__main__':
    from .data_loader import load_news, load_behaviors
    news_df = load_news()
    behaviors_df = load_behaviors()

    if news_df is not None and behaviors_df is not None:
        processed_news = preprocess_news(news_df)
        print("\nProcessed News Head:")
        print(processed_news.head())

        vectorizer, matrix = build_tfidf(processed_news)
        print("\nTF-IDF Vectorizer Features (sample):")
        print(vectorizer.get_feature_names_out()[:20])

        history = build_user_history(behaviors_df)
        print("\nSample User History (U13):")
        print(history.get('U13', 'User not found'))