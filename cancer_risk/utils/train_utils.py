import os
import tempfile

import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, ReLU


@tf.function
def aug_color(x, random_seed=6111):
    x = tf.image.random_brightness(x, 0.05, seed=random_seed)
    x = tf.image.random_contrast(x, 0.7, 1.3, seed=random_seed)
    return x


def convert(image, label):
    image = tf.image.convert_image_dtype(
        image, tf.float32
    )  # Cast and normalize the image to [0,1]
    return image, label


@tf.function
def augment_train(*inputs, img_height, img_width, random_seed):
    image, label = inputs
    image, label = convert(image, label)

    # random cropping
    image = tf.image.resize_with_crop_or_pad(
        image, int(img_height * 1.5), int(img_width * 1.5)
    )
    image = tf.image.random_crop(
        image, size=[img_height, img_width, 1], seed=random_seed
    )

    # random flipping
    image = tf.image.random_flip_left_right(image, seed=random_seed)

    # random color
    image = aug_color(image)
    image = tf.tile(image, [1, 1, 3])
    return image, label


@tf.function
def augment_val(*inputs, img_height, img_width, random_seed):
    image, label = inputs
    image, label = convert(image, label)
    image = tf.image.convert_image_dtype(
        image, tf.float32
    )  # Cast and normalize the image to [0,1]
    # add padding
    image = tf.image.resize_with_crop_or_pad(
        image, int(img_height * 1), int(img_width * 1)
    )
    # random cropping
    image = tf.image.random_crop(
        image, size=[img_height, img_width, 1], seed=random_seed
    )
    image = tf.tile(image, [1, 1, 3])

    return image, label


def prepare_for_training_resampling_step1(
    ds,
    random_seed,
    cache=False,
):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=100, seed=random_seed)
    # Repeat forever
    ds = ds.repeat()
    return ds


def prepare_for_training_resampling_step2(
    ds,
    batch_size,
    buffer_size,
):
    # Data augmentation
    ds = ds.map(
        augment_train,
        num_parallel_calls=buffer_size,
    )
    ds = ds.batch(batch_size)

    return ds


def prepare_for_validation(
    ds,
    batch_size,
    buffer_size,
    cache=False,
):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.map(augment_val, num_parallel_calls=buffer_size)
    ds = ds.batch(batch_size)

    return ds


def decode_img(img, img_height, img_width):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [img_height, img_width])


def get_label_classification(file_path, training, class_names=["pos", "neg"]):
    # convert the path to a list of path components
    parts = tf.strings.split(file_path, os.path.sep)
    if training == False:
        label = parts[-2] == class_names
    elif training == True:
        label = tf.strings.split(parts[-2], "_")[1] == class_names

    return tf.cast(label, tf.float32)


def process_path_classification(file_path, img_height, img_width, training=True):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img, img_height, img_width)
    label = get_label_classification(file_path, training)
    return img, label


# Conv-BatchNorm-ReLU block
def conv_batchnorm_relu(x, filters, kernel_size, strides=1):
    x = Conv2D(
        filters=filters, kernel_size=kernel_size, strides=strides, padding="same"
    )(x)
    #    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


# Identity block
def identity_block(tensor, filters):
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=1)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4 * filters, kernel_size=1, strides=1)(x)
    x = Add()([tensor, x])  # skip connection
    x = ReLU()(x)
    return x


# Projection block
def projection_block(tensor, filters, strides):
    # left stream
    x = conv_batchnorm_relu(tensor, filters=filters, kernel_size=1, strides=strides)
    x = conv_batchnorm_relu(x, filters=filters, kernel_size=3, strides=1)
    x = Conv2D(filters=4 * filters, kernel_size=1, strides=1)(x)
    # right stream
    shortcut = Conv2D(filters=4 * filters, kernel_size=1, strides=strides)(tensor)
    #     shortcut = BatchNormalization()(shortcut)
    x = Add()([shortcut, x])  # skip connection
    x = ReLU()(x)
    return x


# Resnet block
def resnet_block(x, filters, reps, strides):
    x = projection_block(x, filters, strides)
    for _ in range(reps - 1):
        x = identity_block(x, filters)
    return x


def add_l1l2_regularizer(model, l1=0.0, l2=0.0, reg_attributes=None):
    # Add L1L2 regularization to the whole model.
    # NOTE: This will save and reload the model. Do not call this function inplace but with
    # model = add_l1l2_regularizer(model, ...)

    if not reg_attributes:
        reg_attributes = ["kernel_regularizer", "bias_regularizer"]
    if isinstance(reg_attributes, str):
        reg_attributes = [reg_attributes]

    regularizer = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)

    for layer in model.layers:
        for attr in reg_attributes:
            if hasattr(layer, attr):
                setattr(layer, attr, regularizer)

    # So far, the regularizers only exist in the model config. We need to
    # reload the model so that Keras adds them to each layer's losses.
    model_json = model.to_json()

    # Save the weights before reloading the model.
    tmp_weights_path = os.path.join(tempfile.gettempdir(), "tmp_weights.h5")
    model.save_weights(tmp_weights_path)

    # Reload the model
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights(tmp_weights_path, by_name=True)

    return model
