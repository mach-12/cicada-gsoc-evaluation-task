import matplotlib.pyplot as plt
import visualkeras


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


# Source: https://github.com/paulgavrikov/visualkeras/?tab=readme-ov-file#usage
def text_callable(layer_index, layer):
    # Every other piece of text is drawn above the layer, the first one below
    above = bool(layer_index % 2)

    # Get the output shape of the layer
    output_shape = [x for x in list(layer.output_shape) if x is not None]

    # If the output shape is a list of tuples, we only take the first one
    if isinstance(output_shape[0], tuple):
        output_shape = list(output_shape[0])
        output_shape = [x for x in output_shape if x is not None]

    # Variable to store text which will be drawn
    output_shape_txt = ""

    # Create a string representation of the output shape
    for ii in range(len(output_shape)):
        output_shape_txt += str(output_shape[ii])
        if ii < len(output_shape) - 2:  # Add an x between dimensions, e.g. 3x3
            output_shape_txt += "x"
        if (
            ii == len(output_shape) - 2
        ):  # Add a newline between the last two dimensions, e.g. 3x3 \n 64
            output_shape_txt += "\n"

    # Add the name of the layer to the text, as a new line
    output_shape_txt += f"\n{layer.name}"

    # Return the text value and if it should be drawn above the layer
    return output_shape_txt, above


def create_neural_network_visualization(model, name):

    visualkeras.layered_view(
        model,
        legend=True,
        text_callable=text_callable,
        scale_z=0.05,
        to_file=f"./plots/{name}_architecture.png",
    )
    print(f"Model Architecture Image saved to './plots/{name}_architecture.png'")


def plot_training_loss(loss, val_loss):

    plt.figure()
    plt.plot(loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_training_accuracy(accuracy, val_accuracy):

    plt.figure()
    plt.plot(accuracy, label="Train Accuracy")
    plt.plot(val_accuracy, label="Validation Accuracy")
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
