import pandas as pd
import os
from . import config

def load_news(data_dir=config.TRAIN_DATA_DIR):
    """Loads the news.tsv file."""
    news_path = os.path.join(data_dir, config.NEWS_TSV)
    try:
        # Adjust column names based on MIND dataset documentation
        col_names = ['news_id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities']
        news_df = pd.read_csv(news_path, sep='\t', header=None, names=col_names)
        print(f"Loaded news data: {news_df.shape}")
        return news_df
    except FileNotFoundError:
        print(f"Error: {news_path} not found.")
        return None

def load_behaviors(data_dir=config.TRAIN_DATA_DIR):
    """Loads the behaviors.tsv file."""
    behaviors_path = os.path.join(data_dir, config.BEHAVIORS_TSV)
    try:
        # Adjust column names based on MIND dataset documentation
        col_names = ['impression_id', 'user_id', 'time', 'history', 'impressions']
        behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None, names=col_names)
        print(f"Loaded behaviors data: {behaviors_df.shape}")
        return behaviors_df
    except FileNotFoundError:
        print(f"Error: {behaviors_path} not found.")
        return None

if __name__ == '__main__':
    # Example usage:
    news = load_news()
    if news is not None:
        print("\nNews Head:")
        print(news.head())

    behaviors = load_behaviors()
    if behaviors is not None:
        print("\nBehaviors Head:")
        print(behaviors.head())