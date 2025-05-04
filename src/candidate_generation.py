import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from . import config

def generate_content_candidates(user_id, user_history, news_with_vectors, tfidf_matrix, top_k=config.CG_CONTENT_TOP_K):
    """Generates candidates based on content similarity to user history."""
    if user_id not in user_history or not user_history[user_id]:
        # Cold start: return empty list or fallback to popularity
        return []

    clicked_articles = user_history[user_id]
    # Get indices of clicked articles in the news_df
    try:
        # Ensure news_with_vectors index is aligned with tfidf_matrix rows
        clicked_indices = news_with_vectors[news_with_vectors['news_id'].isin(clicked_articles)].index
    except KeyError:
         print(f"Warning: Some clicked articles for user {user_id} not found in news data.")
         return [] # Or handle differently

    if len(clicked_indices) == 0:
        return []

    # Average TF-IDF vector of clicked articles
    user_profile_vector = tfidf_matrix[clicked_indices].mean(axis=0)

    # Calculate cosine similarity between user profile and all articles
    # Ensure user_profile_vector is 2D array for cosine_similarity
    if isinstance(user_profile_vector, np.matrix):
         user_profile_vector = user_profile_vector.A # Convert matrix to array if needed

    if user_profile_vector.ndim == 1:
        user_profile_vector = user_profile_vector.reshape(1, -1)


    similarities = cosine_similarity(user_profile_vector, tfidf_matrix)

    # Get top K similar articles (indices)
    # similarities[0] because user_profile_vector has 1 row
    # Use argsort to get indices, then reverse and take top K
    # Exclude already clicked articles from candidates
    similar_indices = np.argsort(similarities[0])[::-1]

    candidate_indices = [idx for idx in similar_indices if idx not in clicked_indices][:top_k]

    # Map indices back to news_ids
    candidate_news_ids = news_with_vectors.iloc[candidate_indices]['news_id'].tolist()

    return candidate_news_ids

def generate_popularity_candidates(news_df, top_m=config.CG_POPULARITY_TOP_M):
    """Generates candidates based on global popularity/recency (using index as proxy)."""
    # Sort by precomputed popularity score (higher is better)
    popular_articles = news_df.sort_values(by='popularity_score', ascending=False)
    return popular_articles['news_id'].head(top_m).tolist()

def get_candidates(user_id, user_history, news_with_vectors, tfidf_matrix, max_candidates=config.CG_FINAL_SIZE):
    """Combines candidates from different strategies."""
    # Ensure news_with_vectors has 'popularity_score'
    if 'popularity_score' not in news_with_vectors.columns:
         # Add a dummy score if missing (e.g., based on index)
         news_with_vectors['popularity_score'] = news_with_vectors.index / len(news_with_vectors)


    content_candidates = generate_content_candidates(user_id, user_history, news_with_vectors, tfidf_matrix)
    popularity_candidates = generate_popularity_candidates(news_with_vectors)

    # Combine and deduplicate
    combined_candidates = list(set(content_candidates + popularity_candidates))

    # Remove articles already clicked by the user
    clicked_articles = user_history.get(user_id, [])
    final_candidates = [cand for cand in combined_candidates if cand not in clicked_articles]

    # Limit the number of candidates (optional)
    return final_candidates[:max_candidates]

if __name__ == '__main__':
    # Example Usage (requires pre-run training script outputs)
    try:
        processed_news = joblib.load(config.PROCESSED_NEWS_PKL)
        user_history = joblib.load(config.USER_HISTORY_PKL)
        tfidf_matrix = joblib.load(config.TFIDF_MATRIX_PKL)

        # Add TF-IDF vectors to news df for easier lookup (might be memory intensive)
        # This assumes processed_news index aligns with tfidf_matrix rows
        # processed_news['vector'] = list(tfidf_matrix.toarray()) # Very memory intensive! Avoid if possible.

        test_user = 'U13' # Example user
        if test_user in user_history:
            print(f"History for {test_user}: {user_history[test_user][:10]}...") # Show first 10
            candidates = get_candidates(test_user, user_history, processed_news, tfidf_matrix)
            print(f"\nGenerated candidates for {test_user} ({len(candidates)}):")
            print(candidates[:20]) # Show first 20
        else:
            print(f"User {test_user} not found in history.")

        # Test cold start user
        cold_user = 'U_COLD_START'
        cold_candidates = get_candidates(cold_user, user_history, processed_news, tfidf_matrix)
        print(f"\nGenerated candidates for {cold_user} ({len(cold_candidates)}):")
        print(cold_candidates[:20])

    except FileNotFoundError:
        print("Error: Model/data files not found. Run scripts/run_training.py first.")