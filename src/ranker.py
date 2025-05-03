import lightgbm as lgb
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src import config, utils, feature_engineering, data_loader # Need data loaders for training data gen

def train_ranker(use_validation=True):
    """Trains the LightGBM classifier model."""
    utils.setup_logging()
    logging.info("--- Starting Ranker Training ---")

    # Load necessary data for feature generation
    logging.info("Loading preprocessed data for feature generation...")
    behaviors_df = data_loader.load_behaviors_data(config.MIND_SMALL_TRAIN_DIR)
    user_history = utils.load_object(config.USER_HISTORY_PKL)
    news_data = utils.load_object(config.PROCESSED_NEWS_PKL)
    tfidf_matrix = utils.load_object(config.TFIDF_MATRIX_PKL)
    vectorizer = utils.load_object(config.TFIDF_VECTORIZER_PKL)

    # Create news_id to index mapping
    news_id_to_idx = {id: i for i, id in enumerate(news_data['id'])}

    # Create ranking data
    X, y, _, feature_names = feature_engineering.create_ranking_data(
        behaviors_df, user_history, news_data, tfidf_matrix, vectorizer, news_id_to_idx
    )

    if X.empty:
        logging.error("Feature matrix is empty. Cannot train model.")
        return

    logging.info(f"Training data shape: X={X.shape}, y={y.shape}")
    logging.info(f"Feature names: {feature_names}")


    # Train/Validation Split
    if use_validation:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config.TEST_SIZE, random_state=42, stratify=y
        )
        logging.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
        logging.info(f"Validation shapes: X={X_val.shape}, y={y_val.shape}")

        # Prepare dataset for LightGBM
        lgb_train = lgb.Dataset(X_train, y_train, feature_name=feature_names)
        lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=feature_names)

        logging.info("Training LightGBM Classifier with validation...")
        model = lgb.train(
            config.LGBM_PARAMS,
            lgb_train,
            valid_sets=lgb_eval,
            # callbacks=[lgb.early_stopping(config.LGBM_PARAMS['early_stopping_round'], verbose=True)] # Use callbacks in newer versions
            early_stopping_rounds=config.LGBM_PARAMS.get('early_stopping_round', 50) # Older syntax
        )

    else:
         # Train on all data if no validation
        lgb_train = lgb.Dataset(X, y, feature_name=feature_names)
        logging.info("Training LightGBM Classifier on all data (no validation)...")
        model = lgb.train(
            config.LGBM_PARAMS,
            lgb_train
            # No validation set or early stopping here
        )


    # Save the trained model
    utils.save_object(model, config.RANKER_MODEL_PKL)
    # Save feature names used for training (important for prediction)
    utils.save_object(feature_names, config.RANKER_MODEL_PKL + ".features")


    logging.info("--- Ranker Training Finished ---")
    return model

def predict_scores(model, features_df, expected_feature_names):
    """Predicts scores using the trained LightGBM model."""
    logging.debug(f"Predicting scores for {features_df.shape[0]} samples.")

    # Ensure columns match training features
    # Add missing columns with 0 (e.g., categories not seen in this batch)
    for col in expected_feature_names:
        if col not in features_df.columns:
            features_df[col] = 0
    # Reorder columns to match training order
    features_df = features_df[expected_feature_names]

    scores = model.predict(features_df)
    return scores

if __name__ == '__main__':
    # Example: python -m src.ranker
    train_ranker()