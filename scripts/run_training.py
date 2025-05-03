import sys
import os
# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import ranker

if __name__ == "__main__":
    ranker.train_ranker(use_validation=True) # Set to False to train on all data