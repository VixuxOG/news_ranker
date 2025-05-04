import sys
import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.data_loader import load_behaviors # Only need behaviors for eval
from src.candidate_generation import get_candidates
from src.feature_engineering import get_ranking_features
from src.ranker import load_model, predict_scores
from src.utils import timer, load_processed_data

# Basic evaluation: Calculate rank of clicked item
def evaluate_ranking(test_behaviors, user_history, processed_news, tfidf_matrix, vectorizer, ranker_model, news_lookup, newsid_to_idx):
    print("Starting evaluation...")
    ranks = []

    # Get training columns for consistency during feature extraction
    # This is a HACK - ideally save/load columns during training
    try:
        # Attempt to infer columns from a dummy feature extraction
        dummy_features = get_ranking_features('dummy_user', processed_news['news_id'].iloc[0], {}, news_lookup, tfidf_matrix, vectorizer, newsid_to_idx)
        train_cols_df = pd.DataFrame([dummy_features])
        train_cols_df = pd.get_dummies(train_cols_df, columns=['article_category'], dummy_na=False)
        training_columns = train_cols_df.columns
        print(f"Inferred training columns ({len(training_columns)}): {training_columns.tolist()}")
    except Exception as e:
        print(f"Could not infer training columns: {e}. Evaluation might fail.")
        training_columns = None


    for _, row in tqdm(test_behaviors.iterrows(), total=test_behaviors.shape[0], desc="Evaluating"):
        user_id = row['user_id']
        impressions = row['impressions'].split()
        clicked_articles = {imp.split('-')[0] for imp in impressions if imp.endswith('-1')}

        if not clicked_articles:
            continue # Skip impressions with no clicks for this basic evaluation

        # 1. Generate Candidates
        candidates = get_candidates(user_id, user_history, processed_news, tfidf_matrix)
        if not candidates:
            continue # Skip if no candidates generated

        # Ensure clicked articles are included in candidates for evaluation fairness
        candidates_set = set(candidates)
        for clicked in clicked_articles:
            if clicked not in candidates_set:
                candidates.append(clicked)

        # 2. Extract Features for Candidates
        candidate_features_list = []
        valid_candidates = []
        for article_id in candidates:
             features = get_ranking_features(user_id, article_id, user_history, news_lookup, tfidf_matrix, vectorizer, newsid_to_idx)
             if features:
                 candidate_features_list.append(features)
                 valid_candidates.append(article_id) # Keep track of candidates for which features were extracted

        if not candidate_features_list:
            continue

        features_df = pd.DataFrame(candidate_features_list)
        # Apply one-hot encoding consistent with training
        features_df = pd.get_dummies(features_df, columns=['article_category'], dummy_na=False)

        # Align columns with training data
        if training_columns is not None:
            features_df = features_df.reindex(columns=training_columns, fill_value=0)
        else:
             print("Warning: Cannot align columns, results may be inaccurate.")


        # 3. Predict Scores
        scores = predict_scores(ranker_model, features_df)

        # 4. Rank Candidates
        ranked_results = sorted(zip(scores, valid_candidates), key=lambda x: x[0], reverse=True)
        ranked_ids = [article_id for score, article_id in ranked_results]

        # 5. Find Rank of Clicked Item(s)
        for clicked_article_id in clicked_articles:
            try:
                rank = ranked_ids.index(clicked_article_id) + 1
                ranks.append(rank)
            except ValueError:
                ranks.append(np.inf) # Clicked item not found in ranked list

    return ranks

@timer
def main():
    print("--- Starting Evaluation Pipeline ---")

    # 1. Load Processed Data & Model
    print("\n1. Loading Processed Data & Model...")
    processed_news, user_history, vectorizer, tfidf_matrix, ranker_model, news_lookup, newsid_to_idx = load_processed_data()
    if ranker_model is None:
        print("Exiting.")
        return

    # 2. Load Test Behaviors
    print("\n2. Loading Test Behaviors...")
    test_behaviors = load_behaviors(config.DEV_DATA_DIR)
    if test_behaviors is None:
        print("Error loading test behaviors. Exiting.")
        return

    # 3. Run Evaluation
    print("\n3. Running Evaluation...")
    # Limit evaluation size for speed if needed
    # test_behaviors_sample = test_behaviors.sample(n=1000, random_state=42)
    ranks = evaluate_ranking(test_behaviors, user_history, processed_news, tfidf_matrix, vectorizer, ranker_model, news_lookup, newsid_to_idx)

    # 4. Report Basic Metrics
    print("\n--- Evaluation Results ---")
    if ranks:
        ranks = np.array(ranks)
        finite_ranks = ranks[np.isfinite(ranks)]
        if len(finite_ranks) > 0:
            print(f"Average Rank of Clicked Item: {np.mean(finite_ranks):.2f}")
            print(f"Median Rank of Clicked Item: {np.median(finite_ranks):.2f}")
            print(f"Percentage Clicked Item in Top 5: {np.mean(finite_ranks <= 5) * 100:.2f}%")
            print(f"Percentage Clicked Item in Top 10: {np.mean(finite_ranks <= 10) * 100:.2f}%")
        else:
            print("No clicked items were found in the ranked lists.")
        print(f"Percentage Clicked Item Not Found: {np.mean(np.isinf(ranks)) * 100:.2f}%")
    else:
        print("No ranks were calculated.")

    print("\n--- Evaluation Pipeline Complete ---")


if __name__ == "__main__":
    main()