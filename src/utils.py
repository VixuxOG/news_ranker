import joblib
import logging
import sys
from src import config

def setup_logging(level=logging.INFO):
    """Configures basic logging."""
    logging.basicConfig(level=level, format=config.LOGGING_FORMAT, stream=sys.stdout)

def save_object(obj, filepath):
    """Saves a Python object to a file using joblib."""
    try:
        joblib.dump(obj, filepath)
        logging.info(f"Object saved successfully to {filepath}")
    except Exception as e:
        logging.error(f"Error saving object to {filepath}: {e}")
        raise

def load_object(filepath):
    """Loads a Python object from a file using joblib."""
    try:
        obj = joblib.load(filepath)
        logging.info(f"Object loaded successfully from {filepath}")
        return obj
    except FileNotFoundError:
        logging.error(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        logging.error(f"Error loading object from {filepath}: {e}")
        raise