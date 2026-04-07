import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def make_gradcam_heatmap(model, img_array):
    """
    Generates a Grad-CAM heatmap for the given input.

    Grad-CAM works by:
    1. Finding the last Conv2D layer (highest-level features)
    2. Computing gradients of the predicted class score
       with respect to that layer's output
    3. Averaging those gradients to get feature importance weights
    4. Multiplying weights by the feature maps to get the heatmap

    The result shows WHICH parts of the spectrogram the model
    focused on most when making its prediction.
    """

    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer found in model.")

    last_conv_idx = None
    for i, layer in enumerate(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_idx = i

    inp = tf.keras.Input(shape=model.input_shape[1:])
    x = inp
    for layer in model.layers[:last_conv_idx + 1]:
        x = layer(x)
    feature_extractor = tf.keras.Model(inputs=inp, outputs=x)

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(img_array)
        tape.watch(conv_outputs)

        x = conv_outputs
        for layer in model.layers[last_conv_idx + 1:]:
            x = layer(x)
        predictions = x

        predicted_class = tf.argmax(predictions[0])
        loss = predictions[:, predicted_class]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)

    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def show_heatmap(heatmap, title="Grad-CAM Heatmap"):
    """Standalone heatmap viewer."""
    plt.figure(figsize=(6, 5))
    plt.imshow(heatmap, cmap="jet")
    plt.colorbar(label="Activation Intensity")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()