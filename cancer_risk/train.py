import argparse
import os
import random

import efficientnet.tfkeras as efn
import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.train_utils import (
    prepare_for_training_resampling_step1,
    prepare_for_training_resampling_step2,
    prepare_for_validation,
    process_path_classification,
    resnet_block,
)

SEED = 6111
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["TF_CUDNN_DETERMINISTIC"] = "true"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
AUTOTUNE = tf.data.experimental.AUTOTUNE


def prepare_dataset(args):
    # training set
    train_dirs = [
        d
        for d in os.listdir(args.data_folder + "/train/")
        if os.path.isdir(os.path.join(args.data_folder + "/train/", d))
    ]
    train_ds_list = []
    for train_dir in train_dirs:
        traindata_ds = tf.data.Dataset.list_files(
            args.data_folder + "/train/" + str(train_dir) + "/*", seed=SEED
        )
        train_labeled_ds = traindata_ds.map(
            lambda path: process_path_classification(
                path,
                img_height=args.img_height,
                img_width=args.img_width,
                training=True,
            ),
            num_parallel_calls=AUTOTUNE,
        )
        train_ds = prepare_for_training_resampling_step1(
            train_labeled_ds, random_seed=SEED, cache=False
        )
        train_ds_list.append(train_ds)

    train_ds = tf.data.experimental.sample_from_datasets(
        train_ds_list, weights=[1 / len(train_dirs)] * len(train_dirs)
    )
    train_ds = prepare_for_training_resampling_step2(
        train_ds, batch_size=args.batch_size, buffer_size=AUTOTUNE
    )

    # validation set
    valdata_ds = tf.data.Dataset.list_files(args.data_folder + "/val/*/*", seed=SEED)
    val_labeled_ds = valdata_ds.map(
        lambda path: process_path_classification(
            path, img_height=args.img_height, img_width=args.img_width, training=False
        ),
        num_parallel_calls=AUTOTUNE,
    )
    val_ds = prepare_for_validation(
        val_labeled_ds, batch_size=args.batch_size, buffer_size=AUTOTUNE
    )
    return train_ds, val_ds


def create_model(args, model_type):
    input_shape = (args.img_height, args.img_width, 3)
    base_model = efn.EfficientNetB3(
        input_shape=input_shape, weights=args.checkpoint_option, include_top=False
    )
    drop_layers = [
        "block1b_drop",
        "block2b_drop",
        "block2c_drop",
        "block3b_drop",
        "block3c_drop",
        "block4b_drop",
        "block4c_drop",
        "block4d_drop",
        "block4e_drop",
        "block5b_drop",
        "block5c_drop",
        "block5d_drop",
        "block5e_drop",
        "block6b_drop",
        "block6c_drop",
        "block6d_drop",
        "block6e_drop",
        "block6f_drop",
        "block7b_drop",
    ]
    for drop in drop_layers:
        base_model.get_layer(drop).seed = SEED
    base_model.trainable = True

    # Create a new model
    inputs = keras.Input(shape=(args.img_height, args.img_width, 3))

    # The base model contains batchnorm layers. We want to keep them in inference mode
    # when we unfreeze the base model for fine-tuning, so we make sure that the
    # base_model is running in inference mode here.
    x = base_model(inputs, training=False)
    if model_type == "cancer":
        x = resnet_block(x, filters=512, reps=3, strides=2)
        x = resnet_block(x, filters=512, reps=3, strides=2)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(2)(x)
    outputs = tf.keras.layers.Softmax()(x)
    model = keras.Model(inputs, outputs)
    model.summary()
    return model


def train(train_ds, val_ds, args, model_type):
    strategy = tf.distribute.MirroredStrategy()

    # create and compile the model
    with strategy.scope():
        model = create_model(args, model_type)
        optimizer = tf.keras.optimizers.SGD(momentum=0.9)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                from_logits=False, label_smoothing=0
            ),
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.CategoricalCrossentropy(
                    name="categorical_crossentropy", dtype=None, from_logits=False
                ),
            ],
        )

    file_writer = tf.summary.create_file_writer(args.model_dir + "/metrics")
    file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=args.model_dir, histogram_freq=1
    )
    checkpoint_path = os.path.join(args.model_dir, "cp-{epoch:04d}.ckpt")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, verbose=1, save_weights_only=True, period=1
    )
    model.save_weights(checkpoint_path.format(epoch=0))

    # fit the model
    model.fit(
        train_ds,
        epochs=args.num_epoch,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=val_ds,
        callbacks=[tensorboard_callback, cp_callback],
    )


# create a main function
def main(args):
    #  prepare the dataset
    train_ds, val_ds = prepare_dataset(args)

    # start training
    model_type = args.model_type
    train(train_ds, val_ds, args, model_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancer or Risk Predictor")
    parser.add_argument(
        "--data_folder", type=str, required=True, help="Path to the data folder"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Directory to save models and logs"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["cancer", "risk"],
        help="Type of model to train: 'cancer' or 'risk'",
    )
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--decay_epoch", type=int, default=50)
    parser.add_argument("--img_height", type=int, default=1024)
    parser.add_argument("--img_width", type=int, default=832)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--checkpoint_option", type=str, default="noisy-student")
    parser.add_argument("--train_pos_num", type=int, default=2135)
    parser.add_argument("--steps_per_epoch", type=int, default=1068)
    parser.add_argument("--weight_decay", type=float, default=0)

    args = parser.parse_args()

    main(args)
