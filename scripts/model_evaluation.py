import numpy as np
import tensorflow as tf
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt


def find_optimal_threshold(labels, raw_predictions):
    """
    Finds the optimal threshold using the ROC curve based on the Youden's J Statistic metric

    Parameters:
        labels (array-like): True binary labels.
        raw_predictions (array-like): Predicted scores or probabilities.

    Returns:
        dict: Optimal thresholds and corresponding performance metrics.
    """
    print("Computing ROC curve...")
    fpr, tpr, thresholds = roc_curve(labels, raw_predictions)

    # Youden's J Statistic metric
    print("Calculating Youden's J statistic...")
    j_scores = tpr - fpr
    ix_j = np.argmax(j_scores)
    optimal_j_threshold = thresholds[ix_j]

    print(f"Optimal threshold based on Youden's J: {optimal_j_threshold:.4f}")

    return optimal_j_threshold
