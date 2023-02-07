'''
Module for VGG19 model.
'''
from io import BytesIO
import logging

from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19

from .utils import (
    content_loss,
    style_loss,
    total_variation_loss,
    preprocess_image,
    deprocess_image
)

# Build a VGG19 model loaded with pre-trained ImageNet weights
model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = {layer.name: layer.output for layer in model.layers}

# Set up a model that returns the activation values for every layer in
# VGG19 (as a dict).
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

# List of layers to use for the style loss.
style_layer_names = [
    "block1_conv1",
    "block2_conv1",
    "block3_conv1",
    "block4_conv1",
    "block5_conv1",
]
# The layer to use for the content loss.
CONTENT_LAYER_NAME = "block5_conv2"

# Weights of the different loss components
TOTAL_VARIATION_WEIGHT = 1e-6
STYLE_WEIGHT = 1e-6
CONTENT_WEIGHT = 2.5e-8


def nst_generator(image_file: bytes, style_file: bytes):
    '''
    Generate Neural style transfer images.
    '''
    img = Image.open(BytesIO(image_file))
    width, height = img.size
    img_nrows = 400
    img_ncols = int(width * img_nrows / height)

    base_image = preprocess_image(image_file, img_nrows, img_ncols)
    style_reference_image = preprocess_image(style_file, img_nrows, img_ncols)
    combination_image = tf.Variable(preprocess_image(image_file,
                                                     img_nrows,
                                                     img_ncols))

    combination_image = tf.Variable(base_image)

    iterations = 100
    for i in range(1, iterations + 1):
        loss, grads = compute_loss_and_grads(
            combination_image,
            base_image,
            style_reference_image,
            img_nrows,
            img_ncols
        )
        logging.info('Loss at iteration %s is %s', i, loss)
        optimizer.apply_gradients([(grads, combination_image)])
        img = deprocess_image(combination_image.numpy(), img_nrows, img_ncols)
        yield img


@tf.function
def compute_loss_and_grads(combination_image,
                           base_image,
                           style_reference_image,
                           img_nrows,
                           img_ncols):
    '''
    Computer loss & grads.
    '''
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image,
                            base_image,
                            style_reference_image,
                            img_nrows,
                            img_ncols)
    grads = tape.gradient(loss, combination_image)
    return loss, grads


def compute_loss(combination_image,
                 base_image,
                 style_reference_image,
                 img_nrows,
                 img_ncols):
    '''
    Compute loss.
    '''
    input_tensor = tf.concat(
        values=[base_image, style_reference_image, combination_image], axis=0
    )
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[CONTENT_LAYER_NAME]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + CONTENT_WEIGHT * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        style_loss_val = style_loss(style_reference_features,
                                    combination_features,
                                    img_nrows,
                                    img_ncols)
        loss += (STYLE_WEIGHT / len(style_layer_names)) * style_loss_val

    # Add total variation loss
    loss += TOTAL_VARIATION_WEIGHT * total_variation_loss(combination_image,
                                                          img_nrows,
                                                          img_ncols)
    return loss
