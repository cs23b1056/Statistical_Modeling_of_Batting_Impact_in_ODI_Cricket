"""Clustering helpers for classifying batsmen: anchors, attackers, finishers."""

import pandas as pd
from sklearn.cluster import KMeans


def cluster_batters(df: pd.DataFrame, features, n_clusters=3):
    """Run KMeans on provided features and return labels.

    Args:
        df: DataFrame with features
        features: list of column names to use
        n_clusters: number of clusters

    Returns:
        array of cluster labels
    """
    X = df[features].dropna()
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X)
    return labels, model
