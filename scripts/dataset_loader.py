import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.data import Dataset
from typing import Tuple


def load_dataset(batch_size=32):
    images = np.load("data/images.npy")
    labels = np.load("data/labels.npy")
    images = images.astype("float32")

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    # Adding Batch Size
    dataset = dataset.batch(batch_size)
    return dataset, images, labels


class TrainingDatasetLoader:
    def __init__(self, batch_size: int = 32, test_size: float = 0.2, seed: int = 200):
        self.batch_size = batch_size
        self.test_size = test_size
        self.seed = seed

    def load_data(self) -> tuple:
        # Set random seeds for reproducibility
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        # Load and convert the images and labels
        images = np.load("data/images.npy").astype("float32")
        labels = np.load("data/labels.npy")
        return images, labels

    def split_data(self, images: np.ndarray, labels: np.ndarray) -> tuple:
        images_train, images_test, labels_train, labels_test = train_test_split(
            images,
            labels,
            test_size=self.test_size,
            random_state=self.seed,
            stratify=labels,
        )
        return images_train, images_test, labels_train, labels_test

    def get_tf_datasets(
        self,
        images_train: np.ndarray,
        images_test: np.ndarray,
        labels_train: np.ndarray,
        labels_test: np.ndarray,
    ) -> tuple:
        train_dataset = (
            tf.data.Dataset.from_tensor_slices((images_train, labels_train))
            .shuffle(len(images_train), self.seed)
            .batch(self.batch_size)
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (images_test, labels_test)
        ).batch(self.batch_size)
        return train_dataset, test_dataset

    def load_training_dataset(
        self,
    ) -> tuple:
        images, labels = self.load_data()
        images_train, images_test, labels_train, labels_test = self.split_data(
            images, labels
        )
        train_dataset, test_dataset = self.get_tf_datasets(
            images_train, images_test, labels_train, labels_test
        )
        return (
            train_dataset,
            test_dataset,
            images_train,
            images_test,
            labels_train,
            labels_test,
        )


if __name__ == "__main__":
    dataset, images, labels = load_dataset()

    print("Labels")
    print(labels.shape)

    print("Unique labels")
    print(set(labels))

    print("Single Image Labels")
    print(labels[0])
    print(labels[800])

    print()
    print("Images")
    print(images.shape)

    print("Single Image")
    print(images[0, :, :])
