import tensorflow as tf
from keras import Model, Input
from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    Reshape,
)


class AutoencoderModel:
    def __init__(
        self,
        input_shape=(16, 16, 1),
        filters=32,
        latent_dim=40,
        kernel_size=3,
        dropout_rate=0.1,
    ):
        """
        input_shape: shape of the input (H, W, C)
        filters: number of filters for the first Conv2D layer
        latent_dim: dimensionality of the latent vector
        kernel_size: kernel size for Conv2D layers
        dropout_rate: dropout probability
        """
        self.input_shape = input_shape
        self.filters = filters
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate

    def get_model(self):
        # ----- Encoder -----
        inputs = Input(shape=self.input_shape, name="encoder_input")

        # First convolutional block
        x = Conv2D(
            self.filters,
            (self.kernel_size, self.kernel_size),
            padding="same",
            activation="relu",
        )(inputs)

        # Second convolutional block with doubled filters
        x = Conv2D(
            self.filters * 2,
            (self.kernel_size, self.kernel_size),
            padding="same",
            activation="relu",
        )(x)

        # Downsample with MaxPooling
        x = MaxPooling2D((2, 2))(x)

        # Flatten and apply dropout
        x = Flatten()(x)
        x = Dropout(self.dropout_rate)(x)

        # Latent space representation
        latent = Dense(self.latent_dim, activation="relu", name="latent")(x)

        # ----- Decoder -----
        # Expand latent space to a 2D feature map
        x = Dense(8 * 8 * (self.filters * 2), activation="relu")(latent)
        x = Reshape((8, 8, self.filters * 2))(x)

        # Upsample using Conv2DTranspose
        x = Conv2DTranspose(
            self.filters * 2,
            (self.kernel_size, self.kernel_size),
            strides=2,
            padding="same",
            activation="relu",
        )(x)

        # Final reconstruction layer
        outputs = Conv2D(
            1, (3, 3), activation="sigmoid", padding="same", name="decoder_output"
        )(x)

        return Model(inputs, outputs, name="autoencoder")


class ClassifierModel(Model):
    def __init__(self, encoder, num_classes):
        super(ClassifierModel, self).__init__()

        self.encoder = encoder
        self.encoder.trainable = False  # Optionally freeze encoder weights

        # Define additional dense layers
        self.dense1 = Dense(8, activation="relu")
        self.output_layer = Dense(num_classes, activation="softmax")

    def call(self, inputs, training=False):
        # Freezing the encoder weights
        x = self.encoder(inputs, training=False)
        x = self.dense1(x)
        return self.output_layer(x)
