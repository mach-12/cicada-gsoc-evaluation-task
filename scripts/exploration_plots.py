import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


def plot_pca_2d(images, labels, label_mapping):
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


def plot_pca_3d(images, labels, label_mapping):
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
    fig = plt.figure(figsize=(10, 8))
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
