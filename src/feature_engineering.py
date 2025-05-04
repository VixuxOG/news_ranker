import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from tqdm import tqdm
from . import config

def get_ranking_features(user_id, article_id, user_history_map, news_lookup, tfidf_matrix, vectorizer, newsid_to_idx):
    """Extracts features for a given user-article pair for the ranking model."""
    features = {}

    # --- User Features ---
    user_hist = user_history_map.get(user_id, [])
    features['user_click_count'] = len(user_hist)
    # TODO: Add more user features (e.g., preferred categories, avg time between clicks)

    # --- Article Features ---
    article_data = news_lookup.get(article_id)
    if article_data:
        features['article_popularity'] = article_data.get('popularity_score', 0.0)
        # Use category as a feature (needs encoding later, e.g., one-hot)
        features['article_category'] = article_data.get('category', 'Unknown')
        # TODO: Add more article features (e.g., recency, subcategory)
    else:
        features['article_popularity'] = 0.0
        features['article_category'] = 'Unknown'


    # --- User-Item Interaction Features ---
    features['user_item_similarity'] = 0.0 # Default
    if user_hist and article_id in newsid_to_idx:
        try:
            clicked_indices = [newsid_to_idx[h] for h in user_hist if h in newsid_to_idx]
            if clicked_indices:
                user_profile_vector = tfidf_matrix[clicked_indices].mean(axis=0)
                article_idx = newsid_to_idx[article_id]
                article_vector = tfidf_matrix[article_idx]

                # Ensure vectors are 2D for cosine_similarity
                if isinstance(user_profile_vector, np.matrix): user_profile_vector = user_profile_vector.A
                if isinstance(article_vector, np.matrix): article_vector = article_vector.A
                if user_profile_vector.ndim == 1: user_profile_vector = user_profile_vector.reshape(1, -1)
                if article_vector.ndim == 1: article_vector = article_vector.reshape(1, -1)

                similarity = cosine_similarity(user_profile_vector, article_vector)
                features['user_item_similarity'] = similarity[0][0]
        except Exception as e:
            # print(f"Warning: Error calculating similarity for ({user_id}, {article_id}): {e}")
            pass # Keep default similarity 0

    # TODO: Add more interaction features (e.g., has user clicked this category before?)

    return features


def create_ranking_dataset(behaviors_df, news_df, user_history_map, tfidf_matrix, vectorizer, num_neg_samples=config.RANKING_NUM_NEG_SAMPLES):
    """Creates the dataset (features + labels) for training the ranking model."""
    print(f"Creating ranking dataset with {num_neg_samples} negative samples per positive...")

    # Precompute lookups for efficiency
    news_lookup = news_df.set_index('news_id').to_dict('index')
    newsid_to_idx = {news_id: idx for idx, news_id in enumerate(news_df['news_id'])}

    feature_list = []
    label_list = []
    group_list = [] # For LGBMRanker: indicates samples belonging to the same query/user impression

    current_group_id = 0
    # Use tqdm for progress bar
    for _, row in tqdm(behaviors_df.iterrows(), total=behaviors_df.shape[0], desc="Processing Behaviors"):
        user_id = row['user_id']
        impressions = row['impressions'].split()
        group_features = []
        group_labels = []

        positive_samples = []
        negative_samples = []

        for impression in impressions:
            try:
                article_id, click_label = impression.split('-')
                click_label = int(click_label)

                if click_label == 1:
                    positive_samples.append(article_id)
                else:
                    negative_samples.append(article_id)
            except ValueError:
                print(f"Warning: Skipping malformed impression '{impression}'")
                continue # Skip malformed impressions

        # Add positive samples
        for article_id in positive_samples:
            features = get_ranking_features(user_id, article_id, user_history_map, news_lookup, tfidf_matrix, vectorizer, newsid_to_idx)
            if features: # Ensure features were extracted
                 group_features.append(features)
                 group_labels.append(1)

        # Add negative samples (undersample if many negatives)
        sampled_negatives = np.random.choice(
            negative_samples,
            size=min(len(negative_samples), num_neg_samples * len(positive_samples)), # Sample negatives relative to positives
            replace=False
        )
        for article_id in sampled_negatives:
             features = get_ranking_features(user_id, article_id, user_history_map, news_lookup, tfidf_matrix, vectorizer, newsid_to_idx)
             if features:
                 group_features.append(features)
                 group_labels.append(0)

        if group_features: # Only add if we have features for this group
            feature_list.extend(group_features)
            label_list.extend(group_labels)
            group_list.extend([current_group_id] * len(group_features))
            current_group_id += 1


    print(f"Generated {len(feature_list)} samples for ranking.")

    # Convert list of dicts to DataFrame
    features_df = pd.DataFrame(feature_list)

    # --- Feature Encoding ---
    # One-Hot Encode categorical features like 'article_category'
    # Use dummy variables, handle unknown categories if necessary during inference
    features_df = pd.get_dummies(features_df, columns=['article_category'], dummy_na=False)

    # Fill any remaining NaNs (e.g., from failed feature calculations)
    features_df = features_df.fillna(0)

    print("Ranking dataset creation complete.")
    return features_df, np.array(label_list), np.array(group_list)


if __name__ == '__main__':
     # Example Usage (requires pre-run training script outputs)
    try:
        processed_news = joblib.load(config.PROCESSED_NEWS_PKL)
        user_history = joblib.load(config.USER_HISTORY_PKL)
        tfidf_matrix = joblib.load(config.TFIDF_MATRIX_PKL)
        vectorizer = joblib.load(config.TFIDF_VECTORIZER_PKL)
        behaviors = joblib.load('temp_behaviors.pkl') # Assumes behaviors saved temporarily

        features_df, labels, groups = create_ranking_dataset(behaviors, processed_news, user_history, tfidf_matrix, vectorizer)

        print("\nRanking Features DataFrame Head:")
        print(features_df.head())
        print("\nLabels (sample):", labels[:10])
        print("\nGroup IDs (sample):", groups[:10])
        print("\nFeatures shape:", features_df.shape)
        print("Labels shape:", labels.shape)
        print("Groups shape:", groups.shape)

    except FileNotFoundError:
        print("Error: Model/data files not found. Run scripts/run_training.py first.")
    except Exception as e:
        print(f"An error occurred: {e}")