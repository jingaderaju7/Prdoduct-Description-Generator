import pickle
import os
import traceback

script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'models')

try:
    print("Loading vectorizer...")
    with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    print("Vectorizer loaded OK")
except Exception as e:
    print(f"Error loading vectorizer: {e}")
    traceback.print_exc()

try:
    print("\nLoading model...")
    with open(os.path.join(models_dir, 'nearest_neighbors_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    print("Model loaded OK")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()

try:
    print("\nLoading data...")
    import pandas as pd
    df = pd.read_pickle(os.path.join(models_dir, 'processed_data.pkl'))
    print(f"Data loaded OK: {len(df)} rows")
except Exception as e:
    print(f"Error loading data: {e}")
    traceback.print_exc()
