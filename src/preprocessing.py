import pandas as pd
import re
import string
import logging
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from src import config, utils, data_loader
import nltk

# Download stopwords if not already present
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """Basic text cleaning: lowercase, remove punctuation and stopwords."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in STOPWORDS])
    return text

def preprocess_news(news_df):
    """Applies text cleaning and combines title and abstract."""
    logging.info("Preprocessing news data...")
    # Combine title and abstract
    news_df['text'] = news_df[config.TEXT_COLS].agg(' '.join, axis=1)
    # Clean the combined text
    news_df['cleaned_text'] = news_df['text'].apply(clean_text)
    logging.info("Text cleaning complete.")
    return news_df[['id', 'category', 'cleaned_text']] # Keep relevant columns

def fit_tfidf_vectorizer(text_data):
    """Fits a TF-IDF vectorizer on the provided text data."""
    logging.info("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english') # Limit features
    vectorizer.fit(text_data)
    logging.info("TF-IDF fitting complete.")
    return vectorizer

def transform_tfidf(vectorizer, text_data):
    """Transforms text data using a pre-fitted TF-IDF vectorizer."""
    logging.info("Transforming text data with TF-IDF...")
    tfidf_matrix = vectorizer.transform(text_data)
    logging.info("TF-IDF transformation complete.")
    return tfidf_matrix

def run_preprocessing():
    """Main function to run the entire preprocessing pipeline."""
    utils.setup_logging()
    logging.info("--- Starting Preprocessing ---")

    # Load data
    news_df = data_loader.load_news_data(config.MIND_SMALL_TRAIN_DIR)
    behaviors_df = data_loader.load_behaviors_data(config.MIND_SMALL_TRAIN_DIR)

    # Preprocess news
    processed_news_df = preprocess_news(news_df)
    utils.save_object(processed_news_df, config.PROCESSED_NEWS_PKL)

    # Fit and transform TF-IDF
    vectorizer = fit_tfidf_vectorizer(processed_news_df['cleaned_text'])
    utils.save_object(vectorizer, config.TFIDF_VECTORIZER_PKL)
    tfidf_matrix = transform_tfidf(vectorizer, processed_news_df['cleaned_text'])
    utils.save_object(tfidf_matrix, config.TFIDF_MATRIX_PKL)

    # Build user history
    user_history = data_loader.build_user_history(behaviors_df)
    utils.save_object(user_history, config.USER_HISTORY_PKL)

    logging.info("--- Preprocessing Finished ---")

if __name__ == '__main__':
    # This allows running preprocessing directly
    # Example: python -m src.preprocessing
    run_preprocessing()