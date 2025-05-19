import sys
import os
import joblib
import pandas as pd

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_loader import load_news, load_behaviors
from src.preprocessing import preprocess_news, build_tfidf, build_user_history
from src.feature_engineering import create_ranking_dataset
from src.ranker import train_ranker
from src.utils import timer

@timer
def main():
    print("--- Starting Training Pipeline ---")

    # 1. Load Data
    print("\n1. Loading Data...")
    news_df = load_news(config.TRAIN_DATA_DIR)
    behaviors_df = load_behaviors(config.TRAIN_DATA_DIR)
    if news_df is None or behaviors_df is None:
        print("Error loading data. Exiting.")
        return

    # Save behaviors temporarily if needed by feature engineering example
    # joblib.dump(behaviors_df, 'temp_behaviors.pkl')

    # 2. Preprocess News Data
    print("\n2. Preprocessing News Data...")
    processed_news = preprocess_news(news_df)
    joblib.dump(processed_news, config.PROCESSED_NEWS_PKL) # Save processed news
    print(f"Processed news data saved to {config.PROCESSED_NEWS_PKL}")


    # 3. Build TF-IDF
    print("\n3. Building TF-IDF...")
    vectorizer, tfidf_matrix = build_tfidf(processed_news)
    # Vectorizer and matrix are saved within build_tfidf

    # 4. Build User History
    print("\n4. Building User History...")
    user_history = build_user_history(behaviors_df)
    # User history is saved within build_user_history

    # 5. Create Ranking Dataset
    print("\n5. Creating Ranking Dataset...")
    # Ensure user_history is loaded if not returned directly
    user_history = joblib.load(config.USER_HISTORY_PKL)
    features_df, labels, groups = create_ranking_dataset(
        behaviors_df, processed_news, user_history, tfidf_matrix, vectorizer
    )

    # Save features and labels if needed for debugging
    # features_df.to_pickle('models/ranking_features.pkl')
    # joblib.dump(labels, 'models/ranking_labels.pkl')
    # joblib.dump(groups, 'models/ranking_groups.pkl')


    # 6. Train Ranker
    print("\n6. Training Ranker Model...")
    # Pass groups=None if using LGBMClassifier or if groups aren't needed/reliable
    # Pass groups=groups if using LGBMRanker and objective='lambdarank'
    train_ranker(features_df, labels, groups=None) # Using Classifier setup for simplicity here
    # Model is saved within train_ranker

    print("\n--- Training Pipeline Complete ---")

if __name__ == "__main__":
    main()