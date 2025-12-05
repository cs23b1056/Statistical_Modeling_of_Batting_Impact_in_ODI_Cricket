"""Preprocessing helpers for ODI analysis.

Functions:
- load_data(path): loads the CSV and returns a DataFrame
- compute_basic_metrics(df): computes SR, balls faced, runs, and phase flags
"""

import os
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Load CSV into pandas DataFrame.

    Args:
        path: path to CSV file

    Returns:
        pd.DataFrame
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def compute_basic_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute strike rate and normalize common batting metrics.

    Adds columns: 'strike_rate', 'balls_faced'
    This is a lightweight helper; adapt to your dataset schema.
    """
    df = df.copy()
    # Expected columns: 'runs', 'balls' or similar — adapt if different
    if 'runs' in df.columns and 'balls' in df.columns:
        df['balls_faced'] = df['balls']
        df['strike_rate'] = df['runs'] / df['balls_faced'] * 100
    else:
        # Don't modify; user will adapt depending on dataset layout
        pass
    return df
