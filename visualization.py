"""Visualization helpers (matplotlib / seaborn wrappers) for the ODI project."""

import matplotlib.pyplot as plt
import seaborn as sns


def sr_vs_win_scatter(df, sr_col='strike_rate', win_col='win'):
    plt.figure(figsize=(8,6))
    sns.scatterplot(data=df, x=sr_col, y=win_col, alpha=0.6)
    plt.xlabel('Strike Rate')
    plt.ylabel('Win (0/1 or probability)')
    plt.title('Strike Rate vs Win')
    return plt.gcf()
