import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
from sklearn.metrics import precision_score # Precision@K needs custom implementation or careful use

from src import config, utils, data_loader, candidate_generation, feature_engineering, ranker

def dcg_score(y_true, y_score, k=10):
    """Discounted Cumulative Gain."""
    order = np.argsort(y_score)[::-1] # Sort scores descending
    y_true_sorted = np.take(y_true, order[:k]) # Take top k true labels
    gain = 2**y_true_sorted - 1
    discounts = np.log2(np.arange(len(y_true_sorted)) + 2)
    return np.sum(gain / discounts)

def ndcg_score(y_true, y_score, k=10):
    """Normalized Discounted Cumulative Gain."""
    best_dcg = dcg_score(y_true, y_true, k=k) # Ideal ranking
    actual_dcg = dcg_score(y_true, y_score, k=k)
    if best_dcg == 0:
        return 0.0
    return actual_dcg / best_dcg

def precision_at_k(y_true, y_score, k=10):
    """Precision at K."""
    order = np.argsort(y_score)[::-1] # Sort scores descending
    y_true_sorted = np.take(y_true, order[:k]) # Take top k true labels
    if k == 0:
        return 0.0
    return np.sum(y_true_sorted) / k


def evaluate_pipeline():
    """Evaluates the full recommendation pipeline on the dev set."""
    utils.setup_logging()
    logging.info("--- Starting Evaluation ---")

    # Load evaluation data (dev set)
    logging.info("Loading dev set data...")
    behaviors_dev_df = data_loader.load_behaviors_data(config.MIND_SMALL_DEV_DIR)
    # Load necessary objects created during training/preprocessing
    user_history = utils.load_object(config.USER_HISTORY_PKL) # Use history built from train set
    news_data = utils.load_object(config.PROCESSED_NEWS_PKL)
    tfidf_matrix = utils.load_object(config.TFIDF_MATRIX_PKL)
    vectorizer = utils.load_object(config.TFIDF_VECTORIZER_PKL)
    model = utils.load_object(config.RANKER_MODEL_PKL)
    feature_names = utils.load_object(config.RANKER_MODEL_PKL + ".features") # Load feature names

    news_id_to_idx = {id: i for i, id in enumerate(news_data['id'])}

    ndcg_scores = defaultdict(list)
    precision_scores = defaultdict(list)

    logging.info("Evaluating pipeline on dev set impressions...")
    for _, row in tqdm(behaviors_dev_df.iterrows(), total=behaviors_dev_df.shape[0], desc="Evaluating"):
        user_id = row[config.USER_ID_COL]
        impressions = data_loader.parse_impressions(row[config.IMPRESSIONS_COL])
        true_clicks = {imp['article_id'] for imp in impressions if imp[config.CLICK_COL] == 1}

        if not true_clicks: # Skip impressions with no clicks for evaluation
            continue

        # 1. Candidate Generation
        # Pass train behaviors_df for popularity calculation if needed by CG
        train_behaviors_df = data_loader.load_behaviors_data(config.MIND_SMALL_TRAIN_DIR) # Load if needed
        candidates = candidate_generation.generate_candidates(
            user_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx, train_behaviors_df
        )

        # Ensure true clicked items are among candidates for fair evaluation
        # (In real-time, they might not be if CG misses them)
        eval_candidates = list(set(candidates + list(true_clicks)))

        if not eval_candidates:
            continue

        # 2. Feature Engineering for candidates
        candidate_features_list = []
        valid_candidates = []
        for article_id in eval_candidates:
             features = feature_engineering.extract_features(
                 user_id, article_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx
             )
             if features: # Only consider candidates for which features can be extracted
                 candidate_features_list.append(features)
                 valid_candidates.append(article_id)

        if not candidate_features_list:
            continue

        features_df = pd.DataFrame(candidate_features_list)
        # Apply one-hot encoding consistent with training
        features_df = pd.get_dummies(features_df, columns=['article_category'], dummy_na=False)


        # 3. Prediction
        scores = ranker.predict_scores(model, features_df, feature_names)

        # 4. Evaluation Metrics
        # Create true labels for the *valid candidates*
        candidate_true_labels = np.array([1 if cand_id in true_clicks else 0 for cand_id in valid_candidates])

        if len(scores) != len(candidate_true_labels):
             logging.warning(f"Mismatch in scores ({len(scores)}) and labels ({len(candidate_true_labels)}) for user {user_id}. Skipping impression.")
             continue


        for k in config.EVALUATION_K:
            ndcg = ndcg_score(candidate_true_labels, scores, k=k)
            precision = precision_at_k(candidate_true_labels, scores, k=k)
            ndcg_scores[k].append(ndcg)
            precision_scores[k].append(precision)

    # Calculate average metrics
    logging.info("--- Evaluation Results ---")
    for k in config.EVALUATION_K:
        avg_ndcg = np.mean(ndcg_scores[k]) if ndcg_scores[k] else 0
        avg_precision = np.mean(precision_scores[k]) if precision_scores[k] else 0
        logging.info(f"NDCG@{k}: {avg_ndcg:.4f}")
        logging.info(f"Precision@{k}: {avg_precision:.4f}")

    logging.info("--- Evaluation Finished ---")
    return ndcg_scores, precision_scores


if __name__ == '__main__':
    # Example: python -m src.evaluation
    from collections import defaultdict # Add import here if running directly
    evaluate_pipeline()