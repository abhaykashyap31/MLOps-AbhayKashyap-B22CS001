from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="ticks", font_scale=1.2)


def _build_confusion_df(test_labels, predicted_labels, exclude_diagonal=False):
    d = defaultdict(int)
    for true_l, pred_l in zip(test_labels, predicted_labels):
        if exclude_diagonal and true_l == pred_l:
            continue
        d[(true_l, pred_l)] += 1
    rows = [
        {"True Genre": t, "Predicted Genre": p, "Number of Classifications": c}
        for (t, p), c in d.items()
    ]
    return pd.DataFrame(rows)


def plot_confusion_heatmap(test_labels, predicted_labels, figsize=(9, 7)):
    df = _build_confusion_df(test_labels, predicted_labels, exclude_diagonal=False)
    if df.empty:
        return
    df_wide = df.pivot_table(
        index="True Genre",
        columns="Predicted Genre",
        values="Number of Classifications",
    )
    plt.figure(figsize=figsize)
    sns.heatmap(df_wide, linewidths=1, cmap="Purples")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_misclassification_heatmap(test_labels, predicted_labels, figsize=(9, 7)):
    df = _build_confusion_df(test_labels, predicted_labels, exclude_diagonal=True)
    if df.empty:
        return
    df_wide = df.pivot_table(
        index="True Genre",
        columns="Predicted Genre",
        values="Number of Classifications",
    )
    plt.figure(figsize=figsize)
    sns.heatmap(df_wide, linewidths=1, cmap="Purples")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
