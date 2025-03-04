import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def plot_pca_2d(images, labels, label_mapping={0: "Label 0", 1: "Label 1"}):
    """
    Plots a 2D PCA visualization of images.

    Parameters:
    - images: numpy array of shape (n_samples, ...) representing your images.
    - labels: numpy array of shape (n_samples,) with numeric labels.
    - label_mapping: dictionary mapping numeric labels to descriptive names.
    """
    # Reshape images for PCA
    X = images.reshape(images.shape[0], -1)

    # Apply PCA to reduce to 2 components
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Create the 2D scatter plot
    plt.figure(figsize=(6, 4))
    for label in np.unique(labels):
        idx = labels == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], alpha=0.6, label=label_mapping[label])
    plt.title("PCA Visualization of Images (2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.show()


def plot_pca_3d(images, labels, label_mapping={0: "Label 0", 1: "Label 1"}):
    """
    Plots a 3D PCA visualization of images.

    Parameters:
    - images: numpy array of shape (n_samples, ...) representing your images.
    - labels: numpy array of shape (n_samples,) with numeric labels.
    - label_mapping: dictionary mapping numeric labels to descriptive names.
    """
    # Reshape images for PCA
    X = images.reshape(images.shape[0], -1)

    # Apply PCA to reduce to 3 components
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    # Create the 3D scatter plot
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    for label in np.unique(labels):
        idx = labels == label
        ax.scatter(
            X_pca[idx, 0],
            X_pca[idx, 1],
            X_pca[idx, 2],
            alpha=0.6,
            label=label_mapping[label],
        )

    ax.set_title("PCA Visualization of Images (3D)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.legend()
    plt.show()


def plot_pixel_label_patterns(images, labels):
    """
    This function computes:
      - The average image for each label (0 and 1),
      - The difference map (avg_label1 - avg_label0) with pixel intensities capped at ±1,
      - The pixel-wise Pearson correlation between each pixel's intensity and the label.
    It then plots both the difference map and the correlation map side by side.

    Parameters:
      images: np.array of shape (N, 16, 16) representing grayscale images.
      labels: np.array of shape (N,) containing labels {0, 1}.
    """

    # Compute the average image for each label
    images_label0 = images[labels == 0]
    avg_label0 = np.mean(images_label0, axis=0)

    images_label1 = images[labels == 1]
    avg_label1 = np.mean(images_label1, axis=0)

    # Compute the difference map and clip its values to the range [-1, 1]
    # Cap pixel intensities so they don't exceed > 1 (for uniformity sake as the difference map can exceed this!)
    diff_map = avg_label1 - avg_label0
    diff_map = np.clip(diff_map, -1, 1)

    # Compute pixel-wise correlation with label
    N, width, height = images.shape
    images_flat = images.reshape(N, width * height)
    corr_values = np.zeros(width * height)

    # Compute Pearson correlation for each pixel (I was today years old that I found out that numpy has a one-line function for this!)
    for i in range(width * height):
        pixel_intensities = images_flat[:, i]
        corr = np.corrcoef(pixel_intensities, labels)[0, 1]
        corr_values[i] = corr

    # Reshape back to 16x16 for plotting
    corr_map = corr_values.reshape((width, height))

    # Create side-by-side subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot difference map
    im0 = axs[0].imshow(diff_map, cmap="bwr", vmin=-1, vmax=1)
    axs[0].set_title("Difference Map (avg_label1 - avg_label0)")
    cbar0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
    cbar0.set_label("Pixel Intensity Difference")

    # Plot correlation map
    im1 = axs[1].imshow(corr_map, cmap="bwr", vmin=-1, vmax=1)
    axs[1].set_title("Pixel-wise Correlation with Label")
    cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
    cbar1.set_label("Correlation")

    plt.tight_layout()
    plt.show()


def plot_combined_image(images, labels, label_target: int = 1):
    """
    Plots a combined image out of the

    Parameters:
    - images: numpy array of shape (n_samples, ...) representing your images.
    - labels: numpy array of shape (n_samples,) with numeric labels.
    - label_mapping: dictionary mapping numeric labels to descriptive names.
    """

    indices_label1 = np.where(labels == label_target)[0]

    picked_indices = indices_label1[:5]
    picked_images = images[picked_indices]

    # Result Image, Clipped to max value of 1 again!
    sum_image = np.sum(picked_images, axis=0)
    np.clip(sum_image, 0, 1)

    # Let's plot the input images and the result
    fig, axes = plt.subplots(nrows=2, ncols=9, figsize=(8, 4))
    fig.suptitle(
        f"Image combination for Label {label_target}",
        fontsize=14,
        fontweight="bold",
    )

    # We are adding a small '+' for visual sake
    for i in range(5):
        axes[0, 2 * i].imshow(picked_images[i], cmap="gray")
        axes[0, 2 * i].set_title(f"Image {i+1}", fontsize=10)
        axes[0, 2 * i].axis("off")

        if i < 4:
            axes[0, 2 * i + 1].text(
                0.5, 0.5, "+", fontsize=20, ha="center", va="center", fontweight="bold"
            )
            axes[0, 2 * i + 1].axis("off")

    for c in range(9):
        axes[1, c].axis("off")

    axes[1, 4].imshow(sum_image, cmap="gray")
    axes[1, 4].set_title("Combined Image", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()
