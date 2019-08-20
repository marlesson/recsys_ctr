from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

def plot_history(history_df: pd.DataFrame) -> Figure:
    metrics = [column for column in history_df.columns if column != "epoch" and "val_" not in column]

    fig = plt.figure(figsize=(8*len(metrics), 5))

    for i, metric in enumerate(metrics):
        ax = fig.add_subplot(1, len(metrics), i+1)
        if metric == "loss":
            ax.set_yscale('log')
        ax.plot(history_df[metric], label="train")
        ax.plot(history_df[f"val_{metric}"], label="validation")
        ax.set_title(metric)
        ax.set_xlabel('epoch')
        ax.set_ylabel(metric)
        ax.legend()

    fig.tight_layout()

    return fig


def plot_loss_per_lr(learning_rates: List[float], loss_values: List[float]) -> Figure:
    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale('log')
    ax.plot(learning_rates, loss_values)
    ax.set_xlabel('learning rate (log scale)')
    ax.set_ylabel('loss')

    fig.tight_layout()

    return fig


def plot_loss_derivatives_per_lr(learning_rates: List[float], loss_derivatives: List[float]) -> Figure:
    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(1, 1, 1)
    ax.set_xscale('log')
    ax.plot(learning_rates, loss_derivatives)
    ax.set_xlabel('learning rate (log scale)')
    ax.set_ylabel('d/loss')

    fig.tight_layout()

    return fig


def plot_confusion_matrix(confusion_matrix: np.ndarray, class_names, figsize=(10, 7), fontsize=14) -> Figure:
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    heatmap = sns.heatmap(df_cm, annot=True, ax=ax, fmt="d", cmap="Blues")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig