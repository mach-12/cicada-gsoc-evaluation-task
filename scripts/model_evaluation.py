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


def plot_sample_predictions(
    images,
    true_labels,
    scaled_predictions=None,
    raw_predictions=None,
    threshold=0.5,
    num_samples=25,
):
    # Now NumPy uses JAX like method for setting RNG keys!
    rng = np.random.default_rng(200)

    if scaled_predictions is None and raw_predictions is None:
        print("No predictions provided.")
        return

    # Determine predictions
    predictions = (
        scaled_predictions if scaled_predictions is not None else raw_predictions
    )
    pred_classes = (predictions >= threshold).astype(int)

    # Get confusion matrix components
    tn, fp, fn, tp = confusion_matrix(true_labels, pred_classes).ravel()

    # Get indices of different types
    false_positives = np.where((pred_classes == 1) & (true_labels == 0))[0]
    false_negatives = np.where((pred_classes == 0) & (true_labels == 1))[0]
    true_positives = np.where((pred_classes == 1) & (true_labels == 1))[0]
    true_negatives = np.where((pred_classes == 0) & (true_labels == 0))[0]

    # Prioritize false positives & false negatives
    selected_indices = (
        np.random.choice(
            false_positives, min(len(false_positives), num_samples // 3), replace=False
        ).tolist()
        + np.random.choice(
            false_negatives, min(len(false_negatives), num_samples // 3), replace=False
        ).tolist()
    )

    # Fill remaining slots with TP & TN
    other_indices = np.concatenate((true_positives, true_negatives))
    remaining_needed = num_samples - len(selected_indices)
    if len(other_indices) > 0:
        selected_indices += np.random.choice(
            other_indices, min(len(other_indices), remaining_needed), replace=False
        ).tolist()
    rng.shuffle(selected_indices)  # Shuffle selection

    categories = {
        "True Positive": "green",
        "False Positive": "red",
        "True Negative": "blue",
        "False Negative": "orange",
    }

    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(selected_indices):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[idx], cmap="gray")

        # Determine category
        pred_label, true_label = pred_classes[idx], true_labels[idx]
        category = (
            "True Positive"
            if (pred_label, true_label) == (1, 1)
            else (
                "False Positive"
                if (pred_label, true_label) == (1, 0)
                else (
                    "True Negative"
                    if (pred_label, true_label) == (0, 0)
                    else "False Negative"
                )
            )
        )

        title_text = f"True: {true_label}\nPred: {pred_label} ({category})"

        if scaled_predictions is not None:
            title_text += f"\nScaled Value: {scaled_predictions[idx]:.3f}"

        if raw_predictions is not None:
            title_text += f"\nPred Value: {raw_predictions[idx]:.3f}"

        plt.title(title_text, color=categories[category], fontdict={"fontsize": 10})

        plt.axis("off")

        # Adding information of the number so we know what we are talking about!
        plt.text(
            0.5,
            -0.15,
            f"#{i+1}",
            fontsize=10,
            ha="center",
            va="top",
            transform=plt.gca().transAxes,
        )
    plt.tight_layout()

    # Add legend
    legend_patches = [
        mpatches.Patch(color=color, label=label) for label, color in categories.items()
    ]
    plt.figlegend(
        handles=legend_patches, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05)
    )

    plt.show()


def compare_sum_model_results(loaded_model, images, num_of_samples_per_class=5):
    """
    Compares the absolute difference between the sum of pixel values in an image and the output of a trained model.
    This function tests whether the model is accurately approximating the sum of pixel intensities.

    Parameters:
    loaded_model (tf.keras.Model): The pre-trained model used for prediction.
    images (numpy.ndarray): Array of images from the dataset.
    num_of_samples_per_class (int, optional): Number of samples per class to test. Default is 5.

    Returns:
    pandas.DataFrame: A DataFrame containing the sum of pixel values, model outputs, and their differences.
    """

    rng = np.random.default_rng(200)

    random_arrays = rng.random((num_of_samples_per_class, 16, 16))

    selected_images = images[:num_of_samples_per_class]

    all_images = np.vstack([random_arrays, selected_images])

    # Compute sum of pixel values
    sum_values = np.sum(all_images, axis=(1, 2))

    # Model Predictions
    test_model_outputs = loaded_model.predict(all_images)

    if test_model_outputs.ndim > 1:
        test_model_outputs = test_model_outputs.flatten()

    differences = np.abs(sum_values - test_model_outputs)

    image_labels = [f"Random Image #{i+1}" for i in range(num_of_samples_per_class)] + [
        f"Dataset Image #{i+1}" for i in range(num_of_samples_per_class)
    ]

    df = pd.DataFrame(
        {
            "Input Image": image_labels,
            "Sum of Pixels": sum_values,
            "Model Output": test_model_outputs,
            "Difference (Sum - Model)": differences,
        }
    )

    return df


def plot_sum_vs_model_output_plots(df):
    """
    Plots the comparison between the sum of pixel values and the model's predicted output.

    Parameters:
    df (pandas.DataFrame): DataFrame containing sum of pixel values, model outputs, and differences.

    Returns:
    None
    """
    plt.figure(figsize=(8, 4))
    plt.scatter(
        df["Sum of Pixels"], df["Model Output"], label="Model Predictions", color="blue"
    )
    plt.plot(
        df["Sum of Pixels"],
        df["Sum of Pixels"],
        linestyle="--",
        color="red",
        label="Ideal Prediction (y=x)",
    )

    plt.xlabel("Sum of Pixels")
    plt.ylabel("Model Output")
    plt.title("Comparison of Sum of Pixels and Model Output")
    plt.legend()
    plt.grid(True)

    plt.show()
