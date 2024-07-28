# src/models/train_model.py

from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd

def train_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    """
    Train a RandomForest model.
    """
    rfc = RandomForestClassifier(n_estimators=100, max_depth=10)
    rfc.fit(X, y)
    return rfc

def save_model(model: RandomForestClassifier, filepath) -> None:
    """
    Save the trained model to a file.
    """
    joblib.dump(model, filepath)

def load_model(filepath) -> RandomForestClassifier:
    """
    Load a trained model from a file.
    """
    return joblib.load(filepath)
