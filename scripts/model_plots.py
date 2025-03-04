import matplotlib.pyplot as plt


def plot_model_prediction_histogram(predictions, threshold):
    plt.figure(figsize=(12, 6))
    plt.hist(predictions, bins=50, alpha=0.7)
    plt.axvline(
        x=threshold, color="r", linestyle="--", label=f"Default threshold ({threshold})"
    )
    plt.xlabel("Prediction Value")
    plt.ylabel("Count")
    plt.title("Distribution of Model Predictions")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
