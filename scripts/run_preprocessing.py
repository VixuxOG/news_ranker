import sys
import os
# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import preprocessing

if __name__ == "__main__":
    preprocessing.run_preprocessing()