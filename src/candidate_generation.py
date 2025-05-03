import logging
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from tqdm import tqdm
from src import config, utils

def get_content_based_candidates(user_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx, n=config.N_CONTENT_CANDIDATES):
    """Generates candidates based on TF-IDF similarity to user's recent history."""
    history = user_history.get(user_id, [])
    if not history:
        return []

    # Get recent history (use proxy timestamp/index)
    history.sort(key=lambda x: x[1], reverse=True) # Sort by proxy timestamp desc
    recent_history_ids = [item[0] for item in history[:config.USER_HISTORY_LENGTH]]

    # Find indices of historical articles
    history_indices = [news_id_to_idx[article_id] for article_id in recent_history_ids if article_id in news_id_to_idx]

    if not history_indices:
        return []

    # Calculate average TF-IDF vector for user history
    user_vector = tfidf_matrix[history_indices].mean(axis=0)

    # Calculate cosine similarity with all articles
    similarities = cosine_similarity(user_vector, tfidf_matrix)[0]

    # Get top N similar articles (indices)
    # Exclude articles already in the user's *entire* history to avoid recommending seen items
    all_history_ids = set(item[0] for item in history)
    candidate_indices = np.argsort(similarities)[::-1] # Sort descending

    content_candidates = []
    for idx in candidate_indices:
        article_id = news_data.iloc[idx]['id']
        if article_id not in all_history_ids:
            content_candidates.append(article_id)
            if len(content_candidates) >= n:
                break

    return content_candidates


def get_popularity_candidates(news_data, behaviors_df, n=config.N_POPULARITY_CANDIDATES):
    """Generates candidates based on recent global popularity (clicks)."""
    logging.info("Calculating recent popularity...")
    # --- Simple Popularity Calculation (can be improved) ---
    # Count clicks per article from behavior logs
    click_counts = defaultdict(int)
    for _, row in tqdm(behaviors_df.iterrows(), total=behaviors_df.shape[0], desc="Counting Clicks"):
        impressions = data_loader.parse_impressions(row[config.IMPRESSIONS_COL])
        for imp in impressions:
            if imp[config.CLICK_COL] == 1:
                click_counts[imp['article_id']] += 1

    # Convert to DataFrame and merge with news data to get categories etc. if needed
    popularity_df = pd.DataFrame(click_counts.items(), columns=['id', 'clicks'])
    popularity_df = popularity_df.sort_values('clicks', ascending=False)

    # Simple recency: Just take top N overall popular (MINDsmall doesn't have publish dates)
    # In a real scenario, you'd filter by publication date here.
    popular_candidates = popularity_df['id'].head(n).tolist()
    logging.info(f"Generated {len(popular_candidates)} popular candidates.")
    return popular_candidates


def generate_candidates(user_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx, behaviors_df):
    """Combines candidates from different strategies."""
    logging.debug(f"Generating candidates for user {user_id}")

    # 1. Content-based candidates
    content_candidates = get_content_based_candidates(
        user_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx
    )

    # 2. Popularity-based candidates (calculated once ideally, passed in if possible)
    # For simplicity, recalculating or loading popularity might happen here in a real system
    # Here we assume behaviors_df is available for calculation if needed.
    # A better approach would precompute popularity.
    # Let's use a placeholder or simplified popularity for now if behaviors_df isn't easily passed
    # For this example structure, let's assume popularity is less dynamic and can be derived
    # from the training behaviors data (or precomputed).
    # Reusing the function for simplicity, but be aware of performance implications.
    pop_candidates = get_popularity_candidates(news_data, behaviors_df) # Pass behaviors_df

    # 3. Combine and deduplicate
    candidates = list(set(content_candidates + pop_candidates))

    # 4. Remove already clicked items from user's *entire* history
    all_history_ids = set(item[0] for item in user_history.get(user_id, []))
    final_candidates = [cand for cand in candidates if cand not in all_history_ids]

    logging.debug(f"Generated {len(final_candidates)} final candidates for user {user_id}")
    return final_candidates

# Helper function needed within this module or passed from preprocessing
from src import data_loader # Import necessary functions if needed