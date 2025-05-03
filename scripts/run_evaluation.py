import sys
import os
# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import evaluation
from collections import defaultdict # Need this import if running script directly

if __name__ == "__main__":
    evaluation.evaluate_pipeline()