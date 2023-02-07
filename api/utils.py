'''
Utitlity Functions Module
'''
from io import BytesIO
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19


def preprocess_image(image, img_nrows, img_ncols):
    '''
    Util function to open, resize and format pictures into appropriate tensors
    '''
    img = Image.open(BytesIO(image))
    img = img.resize((img_ncols, img_nrows), Image.NEAREST)
    img = keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img)


def deprocess_image(img, img_nrows, img_ncols):
    '''
    Util function to convert a tensor into a valid image
    '''
    img = img.reshape((img_nrows, img_ncols, 3))
    # Remove zero-center by mean pixel
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def gram_matrix(img):
    '''
    The gram matrix of an image tensor (feature-wise outer product)
    '''
    img = tf.transpose(img, (2, 0, 1))
    features = tf.reshape(img, (tf.shape(img)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram


def style_loss(style, combination, img_nrows, img_ncols):
    '''
    The "style loss" is designed to maintain
    the style of the reference image in the generated image.
    It is based on the gram matrices (which capture style) of
    feature maps from the style reference image
    and from the generated image
    '''
    gram_matrix_s = gram_matrix(style)
    gram_matrix_c = gram_matrix(combination)
    channels = 3
    size = img_nrows * img_ncols
    return (tf.reduce_sum(tf.square(gram_matrix_s - gram_matrix_c)) /
            (4.0 * (channels**2) * (size**2)))


def content_loss(base, combination):
    '''An auxiliary loss function
    designed to maintain the "content" of the
    base image in the generated image
    '''
    return tf.reduce_sum(tf.square(combination - base))


def total_variation_loss(img, img_nrows, img_ncols):
    '''The 3rd loss function, total variation loss,
    designed to keep the generated image locally coherent
    '''
    square_a = tf.square(
        img[:, : img_nrows - 1, : img_ncols - 1, :] -
        img[:, 1:, : img_ncols - 1, :]
    )
    square_b = tf.square(
        img[:, : img_nrows - 1, : img_ncols - 1, :] -
        img[:, : img_nrows - 1, 1:, :]
    )
    return tf.reduce_sum(tf.pow(square_a + square_b, 1.25))


def array_to_bytes(np_arr: np.ndarray) -> bytes:
    '''
    Transform numpy array to bytes.
    '''
    np_bytes = BytesIO()
    np.save(np_bytes, np_arr, allow_pickle=True)
    return np_bytes.getvalue()
