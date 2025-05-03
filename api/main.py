import logging
import pandas as pd
from flask import Flask, request, jsonify

# Important: Adjust imports relative to the project root if running api/main.py directly
# Or structure imports assuming you run Flask from the project root (e.g., flask run)
try:
    from src import config, utils, candidate_generation, feature_engineering, ranker, data_loader
except ImportError:
    # If running directly from api/, adjust path (less ideal)
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config, utils, candidate_generation, feature_engineering, ranker, data_loader


app = Flask(__name__)
utils.setup_logging()

# --- Load necessary objects on startup ---
logging.info("Loading models and data for API...")
try:
    user_history = utils.load_object(config.USER_HISTORY_PKL)
    news_data = utils.load_object(config.PROCESSED_NEWS_PKL)
    tfidf_matrix = utils.load_object(config.TFIDF_MATRIX_PKL)
    vectorizer = utils.load_object(config.TFIDF_VECTORIZER_PKL)
    model = utils.load_object(config.RANKER_MODEL_PKL)
    feature_names = utils.load_object(config.RANKER_MODEL_PKL + ".features")
    news_id_to_idx = {id: i for i, id in enumerate(news_data['id'])}
    # Load train behaviors only if needed by CG (popularity calculation)
    train_behaviors_df = data_loader.load_behaviors_data(config.MIND_SMALL_TRAIN_DIR)
    logging.info("API Ready: Models and data loaded.")
except Exception as e:
    logging.error(f"FATAL: Could not load models/data for API: {e}", exc_info=True)
    # Optionally exit or disable the endpoint if loading fails
    model = None # Indicate failure


@app.route('/rank/<user_id>', methods=['GET'])
def get_recommendations(user_id):
    """Endpoint to get ranked article recommendations for a user."""
    if model is None:
         return jsonify({"error": "Model not loaded, API unavailable"}), 503

    logging.info(f"Received request for user: {user_id}")
    try:
        n_recs = int(request.args.get('n', config.DEFAULT_N_RECOMMENDATIONS))
    except ValueError:
        return jsonify({"error": "Invalid value for 'n'"}), 400

    if user_id not in user_history:
        logging.warning(f"User {user_id} not found in history. Returning popular items (or handle cold start).")
        # Basic cold start: return popular candidates (needs popularity logic here)
        # For now, return empty or a fixed popular list
        pop_candidates = candidate_generation.get_popularity_candidates(news_data, train_behaviors_df, n=n_recs)
        return jsonify({"user_id": user_id, "ranked_articles": pop_candidates})


    try:
        # 1. Candidate Generation
        candidates = candidate_generation.generate_candidates(
            user_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx, train_behaviors_df
        )

        if not candidates:
            logging.info(f"No candidates generated for user {user_id}.")
            return jsonify({"user_id": user_id, "ranked_articles": []})

        # 2. Feature Engineering
        candidate_features_list = []
        valid_candidates = []
        for article_id in candidates:
            features = feature_engineering.extract_features(
                user_id, article_id, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx
            )
            if features:
                candidate_features_list.append(features)
                valid_candidates.append(article_id)

        if not candidate_features_list:
             logging.info(f"No valid features for candidates of user {user_id}.")
             return jsonify({"user_id": user_id, "ranked_articles": []})

        features_df = pd.DataFrame(candidate_features_list)
        features_df = pd.get_dummies(features_df, columns=['article_category'], dummy_na=False)

        # 3. Prediction
        scores = ranker.predict_scores(model, features_df, feature_names)

        # 4. Ranking
        scored_candidates = sorted(zip(valid_candidates, scores), key=lambda x: x[1], reverse=True)

        # 5. Return top N
        ranked_articles = [article_id for article_id, score in scored_candidates[:n_recs]]

        logging.info(f"Returning {len(ranked_articles)} recommendations for user {user_id}")
        return jsonify({"user_id": user_id, "ranked_articles": ranked_articles})

    except Exception as e:
        logging.error(f"Error processing request for user {user_id}: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Run the Flask app
    # Example: python api/main.py
    # Access: http://127.0.0.1:5000/rank/U1?n=5
    app.run(debug=True, host='0.0.0.0') # Use debug=False in production