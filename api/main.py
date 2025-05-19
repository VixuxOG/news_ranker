import sys
import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config
from src.candidate_generation import get_candidates
from src.feature_engineering import get_ranking_features
from src.ranker import predict_scores
from src.utils import load_processed_data # Use utility to load everything

app = Flask(__name__)

# --- Load Models and Data ONCE at startup ---
print("Loading models and data for API...")
(
    processed_news,
    user_history,
    vectorizer,
    tfidf_matrix,
    ranker_model,
    news_lookup,
    newsid_to_idx
) = load_processed_data()

# Infer training columns (HACK - same as in evaluation)
training_columns = None
if ranker_model is not None:
    try:
        dummy_features = get_ranking_features('dummy_user', processed_news['news_id'].iloc[0], {}, news_lookup, tfidf_matrix, vectorizer, newsid_to_idx)
        train_cols_df = pd.DataFrame([dummy_features])
        train_cols_df = pd.get_dummies(train_cols_df, columns=['article_category'], dummy_na=False)
        training_columns = train_cols_df.columns
        print(f"API: Inferred training columns ({len(training_columns)})")
    except Exception as e:
        print(f"API Warning: Could not infer training columns: {e}")

if ranker_model is None:
    print("API Error: Ranker model not loaded. API may not function correctly.")
# ---------------------------------------------

@app.route('/rank/<user_id>', methods=['GET'])
def rank_articles(user_id):
    """API endpoint to get ranked articles for a user."""
    if ranker_model is None or training_columns is None:
        return jsonify({"error": "Model or configuration not loaded properly"}), 500

    try:
        n = int(request.args.get('n', config.DEFAULT_RANKING_SIZE))
    except ValueError:
        return jsonify({"error": "Invalid value for 'n' parameter"}), 400

    print(f"Received request for user: {user_id}, n={n}")

    # 1. Generate Candidates
    candidates = get_candidates(user_id, user_history, processed_news, tfidf_matrix)
    if not candidates:
        print(f"No candidates generated for user {user_id}")
        return jsonify({"user_id": user_id, "ranked_articles": []})

    # 2. Extract Features
    candidate_features_list = []
    valid_candidates = []
    for article_id in candidates:
        features = get_ranking_features(user_id, article_id, user_history, news_lookup, tfidf_matrix, vectorizer, newsid_to_idx)
        if features:
            candidate_features_list.append(features)
            valid_candidates.append(article_id)

    if not candidate_features_list:
         print(f"No valid features extracted for candidates of user {user_id}")
         return jsonify({"user_id": user_id, "ranked_articles": []})

    features_df = pd.DataFrame(candidate_features_list)
    features_df = pd.get_dummies(features_df, columns=['article_category'], dummy_na=False)
    features_df = features_df.reindex(columns=training_columns, fill_value=0) # Align columns

    # 3. Predict Scores
    scores = predict_scores(ranker_model, features_df)

    # 4. Rank and Return Top N
    ranked_results = sorted(zip(scores, valid_candidates), key=lambda x: x[0], reverse=True)
    top_n_articles = [article_id for score, article_id in ranked_results[:n]]

    print(f"Returning {len(top_n_articles)} ranked articles for user {user_id}")
    return jsonify({
        "user_id": user_id,
        "ranked_articles": top_n_articles
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) # Set debug=False for production