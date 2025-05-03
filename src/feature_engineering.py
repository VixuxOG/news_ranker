import pandas as pd
import numpy as np
import logging
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from src import config, utils, data_loader # Make sure data_loader is available

def get_user_features(user_id, user_history):
    """Extracts features based on user history."""
    history = user_history.get(user_id, [])
    click_count = len(history)
    # More features could be added: distinct categories clicked, avg time between clicks etc.
    return {'user_click_count': click_count}

def get_article_features(article_id, news_data, news_id_to_idx):
    """Extracts features for a given article."""
    if article_id not in news_id_to_idx:
        # Handle case where article might not be in the loaded news data (e.g., from dev set)
        return {'article_category': 'Unknown', 'article_popularity': 0} # Default values

    idx = news_id_to_idx[article_id]
    article_info = news_data.iloc[idx]
    # Simple features: category
    # More features: recency (needs publish date), global popularity (needs click counts)
    # Placeholder for popularity - needs to be calculated/passed in
    return {'article_category': article_info.get('category', 'Unknown'),
            'article_popularity': 0} # Placeholder

def get_user_item_features(user_id, article_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx):
    """Extracts features based on user-item interaction."""
    features = {}
    history = user_history.get(user_id, [])
    recent_history_ids = [item[0] for item in sorted(history, key=lambda x: x[1], reverse=True)[:config.USER_HISTORY_LENGTH]]

    # TF-IDF Similarity
    if article_id in news_id_to_idx and recent_history_ids:
        article_idx = news_id_to_idx[article_id]
        history_indices = [news_id_to_idx[h_id] for h_id in recent_history_ids if h_id in news_id_to_idx]
        if history_indices:
            user_vector = tfidf_matrix[history_indices].mean(axis=0)
            article_vector = tfidf_matrix[article_idx]
            similarity = cosine_similarity(user_vector, article_vector)[0][0]
            features['user_item_tfidf_similarity'] = similarity
        else:
            features['user_item_tfidf_similarity'] = 0.0
    else:
        features['user_item_tfidf_similarity'] = 0.0

    # Category Match (Example)
    # Needs article features calculated first or access to news_data
    # article_category = get_article_features(article_id, news_data, news_id_to_idx)['article_category']
    # history_categories = [news_data.iloc[news_id_to_idx[h_id]]['category']
    #                       for h_id in recent_history_ids if h_id in news_id_to_idx]
    # features['user_item_category_match_count'] = history_categories.count(article_category)

    return features

def extract_features(user_id, article_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx):
    """Extracts all features for a user-article pair."""
    if article_id not in news_id_to_idx:
         logging.warning(f"Article {article_id} not found in news data index. Skipping feature extraction.")
         # Return default features or handle appropriately
         # Returning None might be better to filter out these samples later
         return None

    user_feats = get_user_features(user_id, user_history)
    article_feats = get_article_features(article_id, news_data, news_id_to_idx) # Needs popularity passed
    user_item_feats = get_user_item_features(user_id, article_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx)

    # Combine all features
    all_features = {}
    all_features.update(user_feats)
    all_features.update(article_feats)
    all_features.update(user_item_feats)

    # Add identifiers
    all_features[config.USER_ID_COL] = user_id
    all_features[config.ARTICLE_ID_COL] = article_id

    return all_features


def create_ranking_data(behaviors_df, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx, negative_samples=4):
    """Creates the dataset for training the ranking model."""
    logging.info("Creating ranking training data...")
    feature_list = []
    labels = []

    processed_impressions = 0
    total_impressions = behaviors_df.shape[0]

    for _, row in tqdm(behaviors_df.iterrows(), total=total_impressions, desc="Generating Ranker Data"):
        user_id = row[config.USER_ID_COL]
        impressions = data_loader.parse_impressions(row[config.IMPRESSIONS_COL])

        clicked_articles = [imp['article_id'] for imp in impressions if imp[config.CLICK_COL] == 1]
        non_clicked_articles = [imp['article_id'] for imp in impressions if imp[config.CLICK_COL] == 0]

        # Add positive samples (clicked articles)
        for article_id in clicked_articles:
            features = extract_features(user_id, article_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx)
            if features: # Only add if features could be extracted
                feature_list.append(features)
                labels.append(1)

        # Add negative samples (non-clicked articles from the same impression)
        neg_count = 0
        for article_id in non_clicked_articles:
            if neg_count >= len(clicked_articles) * negative_samples and clicked_articles: # Balance ratio
                 break
            features = extract_features(user_id, article_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx)
            if features:
                feature_list.append(features)
                labels.append(0)
                neg_count += 1

        processed_impressions += 1
        # Optional: Log progress periodically
        # if processed_impressions % 10000 == 0:
        #     logging.info(f"Processed {processed_impressions}/{total_impressions} impressions...")


    if not feature_list:
        logging.error("No features were generated. Check data and feature extraction logic.")
        return pd.DataFrame(), np.array([])

    feature_df = pd.DataFrame(feature_list)
    # Handle categorical features (e.g., article_category) - Use One-Hot Encoding
    feature_df = pd.get_dummies(feature_df, columns=['article_category'], dummy_na=False) # Keep track of columns

    # Separate identifiers and features
    id_cols = [config.USER_ID_COL, config.ARTICLE_ID_COL]
    feature_cols = [col for col in feature_df.columns if col not in id_cols]

    logging.info(f"Created ranking data with {feature_df.shape[0]} samples and {len(feature_cols)} features.")
    return feature_df[feature_cols], np.array(labels), feature_df[id_cols], feature_cols # Return feature names