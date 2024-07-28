# src/models/predict_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def predict(model: RandomForestClassifier, X: pd.DataFrame) -> pd.Series:
    """
    Make predictions using the trained model.
    """
    preds = model.predict_proba(X)[:, 1]
    return preds

def evaluate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Evaluate the model using F1 score.
    """
    return f1_score(y_true, y_pred.round())
