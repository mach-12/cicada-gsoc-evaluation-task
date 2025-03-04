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


def plot_confusion_matrix(
    labels, predicted_classes, optimal_threshold=0.5, cmap="Blues"
):
    """
    Plots a confusion matrix using Matplotlib.

    Parameters:
        labels (array-like): True labels.
        predicted_classes (array-like): Predicted labels.
        optimal_threshold (float, optional): Threshold value to display in the title. Default is 0.5.
        cmap (str, optional): Colormap for the heatmap. Default is 'Blues'.
    """
    cm = confusion_matrix(labels, predicted_classes)

    cm_normalized = cm.astype("float") / cm.max()

    # Create the figure
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.colorbar()

    # Add labels and title
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix (threshold={optimal_threshold:.4f})")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_normalized[i, j] > 0.5 else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    # Show the plot
    plt.show()


def plot_roc_curve(labels, predictions, optimal_threshold):
    """
    Plot the Receiver Operating Characteristic (ROC) curve.

    Parameters:
    labels (array-like): True binary labels.
    predictions (array-like): Predicted scores or probabilities.
    optimal_threshold (float): The optimal threshold for classification.
    ix (int): Index of the optimal threshold in the FPR/TPR arrays.
    """

    fpr, tpr, thresholds = roc_curve(labels, predictions)
    ix = np.where(thresholds == optimal_threshold)[0]

    if ix.size > 0:
        pass
    else:
        print("Optimal threshold not found in thresholds_")

    plt.figure(figsize=(6, 4))
    plt.plot(
        fpr, tpr, "b-", linewidth=2, label=f"ROC curve (AUC = {auc(fpr, tpr):.4f})"
    )
    plt.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random Guess")
    plt.plot(
        fpr[ix],
        tpr[ix],
        "ro",
        markersize=10,
        label=f"Optimal threshold = {optimal_threshold:.4f}",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.show()


def plot_precision_recall_curve(labels, predictions):
    """
    Calculate and plot the Precision-Recall curve.


    Parameters:
    labels (array-like): True binary labels.
    predictions (array-like): Predicted scores or probabilities.
    """
    precision, recall, _ = precision_recall_curve(labels, predictions)
    average_precision = average_precision_score(labels, predictions)

    plt.figure(figsize=(6, 4))
    plt.plot(
        recall,
        precision,
        "b-",
        linewidth=2,
        label=f"Precision-Recall curve (AP = {average_precision:.4f})",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.show()
