import pandas as pd
import os
import logging
from collections import defaultdict
from tqdm import tqdm
from src import config

def load_news_data(data_dir):
    """Loads news data from news.tsv."""
    news_path = os.path.join(data_dir, config.NEWS_FILENAME)
    try:
        # Define column names based on MIND dataset description
        col_names = [
            'id', 'category', 'subcategory', 'title', 'abstract', 'url',
            'title_entities', 'abstract_entities'
        ]
        news_df = pd.read_csv(news_path, sep='\t', header=None, names=col_names)
        logging.info(f"Loaded news data from {news_path}. Shape: {news_df.shape}")
        # Fill NaN in text columns
        news_df[config.TEXT_COLS] = news_df[config.TEXT_COLS].fillna('')
        news_df[config.CATEGORY_COL] = news_df[config.CATEGORY_COL].fillna('Unknown')
        return news_df
    except FileNotFoundError:
        logging.error(f"Error: News file not found at {news_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading news data: {e}")
        raise

def load_behaviors_data(data_dir):
    """Loads behaviors data from behaviors.tsv."""
    behaviors_path = os.path.join(data_dir, config.BEHAVIORS_FILENAME)
    try:
        # Define column names
        col_names = [
            'impression_id', 'user_id', 'timestamp', 'history', 'impressions'
        ]
        behaviors_df = pd.read_csv(behaviors_path, sep='\t', header=None, names=col_names)
        logging.info(f"Loaded behaviors data from {behaviors_path}. Shape: {behaviors_df.shape}")
        # Fill NaN history with empty string
        behaviors_df[config.HISTORY_COL] = behaviors_df[config.HISTORY_COL].fillna('')
        return behaviors_df
    except FileNotFoundError:
        logging.error(f"Error: Behaviors file not found at {behaviors_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading behaviors data: {e}")
        raise

def parse_impressions(impressions_str):
    """Parses the impression string into article_id and click status."""
    if not isinstance(impressions_str, str):
        return []
    impressions = []
    for imp in impressions_str.split():
        try:
            article_id, click = imp.split('-')
            impressions.append({'article_id': article_id, config.CLICK_COL: int(click)})
        except ValueError:
            logging.warning(f"Could not parse impression item: {imp}")
            continue # Skip malformed items
    return impressions

def build_user_history(behaviors_df):
    """Builds a dictionary mapping user_id to their click history."""
    user_history = defaultdict(list)
    logging.info("Building user click history...")
    for _, row in tqdm(behaviors_df.iterrows(), total=behaviors_df.shape[0], desc="Processing Behaviors"):
        user_id = row[config.USER_ID_COL]
        history_str = row[config.HISTORY_COL]
        timestamp = row[config.TIMESTAMP_COL] # Keep timestamp for potential recency features

        # Process historical clicks
        if isinstance(history_str, str) and history_str:
            clicked_articles = history_str.split()
            # Store as tuples (article_id, timestamp) - assuming history is chronological
            # Note: MIND dataset timestamp is for the *impression*, not historical clicks.
            # We'll use index as a proxy for recency within history for simplicity.
            for i, article_id in enumerate(clicked_articles):
                 # Avoid duplicates if user appears multiple times
                if article_id not in [item[0] for item in user_history[user_id]]:
                    user_history[user_id].append((article_id, -len(clicked_articles) + i)) # Use negative index as proxy timestamp

        # Process clicks from the current impression log
        impressions = parse_impressions(row[config.IMPRESSIONS_COL])
        for imp in impressions:
            if imp[config.CLICK_COL] == 1:
                 if imp['article_id'] not in [item[0] for item in user_history[user_id]]:
                    user_history[user_id].append((imp['article_id'], 0)) # 0 indicates current impression click

    # Optional: Sort history by proxy timestamp (index) if needed, though list order might suffice
    # for user_id in user_history:
    #    user_history[user_id].sort(key=lambda x: x[1])

    logging.info(f"Built history for {len(user_history)} users.")
    return dict(user_history) # Convert back to regular dict if needed