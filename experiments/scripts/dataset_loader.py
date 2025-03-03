import numpy as np
import tensorflow as tf


def load_dataset():

    images = np.load("../data/images.npy")
    labels = np.load("../data/labels.npy")
    images = images.astype("float32")
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset, images, labels


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
