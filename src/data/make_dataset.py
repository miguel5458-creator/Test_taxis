# src/data/make_dataset.py

import pandas as pd

def load_data(url: str) -> pd.DataFrame:
    """
    Load dataset from a given URL.
    """
    return pd.read_parquet(url)
