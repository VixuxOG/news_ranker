import lightgbm as lgbm
import joblib
import pandas as pd
import numpy as np
from . import config

def train_ranker(features_df, labels, groups=None):
    """Trains the LightGBM ranking model."""
    print("Training LightGBM model...")

    # Ensure features are in numeric format suitable for LightGBM
    # This should be handled after pd.get_dummies in feature engineering
    X = features_df.astype(np.float32)
    y = labels

    # Use LGBMClassifier if not using group info
    # model = lgbm.LGBMClassifier(**config.LGBM_PARAMS)
    # model.fit(X, y)

    # Use LGBMRanker if group info is available and objective is 'lambdarank'
    # Need to calculate group counts for LGBMRanker's fit method
    if groups is not None and config.LGBM_PARAMS.get('objective') == 'lambdarank':
         model = lgbm.LGBMRanker(**config.LGBM_PARAMS)
         group_counts = pd.Series(groups).value_counts().sort_index().tolist()
         model.fit(X, y, group=group_counts)
         print("Trained LGBMRanker.")
    else:
         # Fallback to classifier if groups missing or objective is binary/regression
         print("Warning: Training LGBMClassifier (groups not provided or objective not rank-based).")
         model = lgbm.LGBMClassifier(**config.LGBM_PARAMS)
         model.fit(X, y)
         print("Trained LGBMClassifier.")


    # Save the trained model
    save_model(model, config.RANKER_MODEL_PKL)

    # Print feature importances
    if hasattr(model, 'feature_importances_'):
        feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, features_df.columns)), columns=['Value','Feature'])
        print("\nFeature Importances:")
        print(feature_imp.sort_values(by='Value', ascending=False).head(15))


    return model

def predict_scores(model, features_df):
    """Predicts relevance scores using the trained model."""
    # Ensure features_df columns match training columns (order and presence)
    # This might require saving/loading training columns
    # For now, assume columns are aligned correctly after get_dummies
    X = features_df.astype(np.float32)
    scores = model.predict_proba(X)[:, 1] # Predict probability of class 1 (click) for classifier
    # If using LGBMRanker, use model.predict(X) directly for scores
    # scores = model.predict(X)
    return scores

def save_model(model, path):
    """Saves the model to disk."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path=config.RANKER_MODEL_PKL):
    """Loads the model from disk."""
    try:
        model = joblib.load(path)
        print(f"Model loaded from {path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}")
        return None

if __name__ == '__main__':
    # Example: Load a saved model (if training was run)
    model = load_model()
    if model:
        print("\nModel loaded successfully.")
        print(model)