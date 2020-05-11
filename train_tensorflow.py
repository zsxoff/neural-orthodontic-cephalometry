#!/usr/bin/env python3

"""
Train TensorFlow model script.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Copyright (c) 2020 Konstantin Dobratulin

This software is released under the MIT License.
https://opensource.org/licenses/MIT

"""

import pathlib

import numpy as np
import pandas as pd
import sklearn.model_selection
import tensorflow as tf
from tabulate import tabulate

from constants.classes import CLASSES

# from models.cnn import CNN
from models.unet import UNet

# -----------------------------------------------------------------------------
# Customizable hyperparameters.

# TODO Remove hard-coded dataset path
# TODO Move hyperparameters to YAML.

# Data paths.
DATA_PWD = pathlib.Path("../neural-orthodontic-dataset")

EXPERT_NAME = "expert_1"

PWD_IMAGES = DATA_PWD / "images"
PWD_MARKUP = DATA_PWD / "coords-npz" / EXPERT_NAME
PWD_COORDS = DATA_PWD / "coords-txt" / EXPERT_NAME

# Model config.
BATCH_SIZE = 2
EPOCHS = 80
VALID_SIZE = 10
MODEL_LOAD_WEIGHTS = False
MODEL_RUN_TRAIN = True
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Constant hyperparameters.

# ! Please do not modify this parameters if you are not 100% sure
# ! that you know why they are needed.

IMAGE_SRC_W = 2400
IMAGE_SRC_H = 2000
IMAGE_NET_W = 512
IMAGE_NET_H = 432
CLASSES_COUNT = len(CLASSES)
RANDOM_SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

tf.keras.backend.clear_session()
tf.random.set_seed(RANDOM_SEED)
# -----------------------------------------------------------------------------


def _get_sorted_paths(path, ext):
    """
    Return sorted names of files in directory with extension.

    Args:
        path (str): Path to directory.
        ext (str): Files extension.

    Returns:
        list: List of sorted files paths.

    """
    return list(
        map(
            lambda x: x.as_posix(), sorted(pathlib.Path(path).glob(f"*.{ext}"))
        )
    )


def get_images_paths(path):
    """
    Get sorted images paths.

    Args:
        path (str): Images directory.

    Returns:
        list: Sorted images paths.

    """
    return _get_sorted_paths(path, "jpg")


def get_markup_paths(path):
    """
    Get sorted NPZ archives paths.

    Args:
        path (str): Archives directory.

    Returns:
        list: Sorted NPZ archives paths.

    """
    return _get_sorted_paths(path, "npz")


def rescale_coords(coords, src_size, out_size):
    """
    Rescale coordinates from src_size to out_size.

    Args:
        coords (tuple): Coordinates array of pixel in original image.
        src_size (tuple): Source image size.
        out_size (tuple): Target image size.

    Returns:
        numpy.ndarray: Array of scaled coordinates.

    """
    scaled_coords = coords * (np.asarray(out_size) / np.asarray(src_size))
    return np.round(scaled_coords).astype(np.int)


@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def process_image(path_image):
    """
    Load grayscale JPEG image file.

    Args:
        path_image (tensorflow.string): File name for image file.

    Returns:
        tensorflow.Tensor: Grayscale image (size of WxHx1).

    """
    # Load raw data.
    img = tf.io.read_file(path_image)

    # Convert compressed string to a tensor.
    img = tf.image.decode_jpeg(img, channels=1)

    # Use `convert_image_dtype` to convert to int32 in the [0, MAX] range.
    img = tf.image.convert_image_dtype(img, tf.int32)

    # Resize image.
    return tf.image.resize(img, [IMAGE_NET_W, IMAGE_NET_H])


@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def process_masks(path_masks):
    """
    Read NPZ archive with masks.

    Args:
        path_masks (tensorflow.string): File name for NPZ archive.

    Returns:
        tensorflow.Tensor: Masks tensor with size (WxHx27x1).

    """
    # Read matrix from NPZ.
    masks = tf.numpy_function(
        lambda x: np.load(x)["arr_0"], [path_masks], tf.float32
    )

    # Set matrix default shape.
    masks.set_shape([IMAGE_NET_W, IMAGE_NET_H, CLASSES_COUNT, 1])
    return masks


def load_data(path_image, path_masks):
    """
    Load data map function for TensorFlow.

    Args:
        path_image (tensorflow.string): File name for image.
        path_masks (tensorflow.string): File name for coordinates masks.

    Returns:
        tuple: Tuple of TensorFlow tensors for image and masks.

    """
    return process_image(path_image), process_masks(path_masks)


def normalize(image, masks):
    """
    Normalize data from dataset.

    Args:
        image (tensorflow.Tensor): Image tensor object.
        masks (tensorflow.Tensor): Masks tensor object.

    Returns:
        tuple: Tuple of tensorflow.Tensor with image and masks tensors.

    """
    image = tf.image.per_image_standardization(image)
    return image, masks


def train_model(X_train, X_test, y_train, y_test, save_dir):
    """
    Run train model process.

    Args:
        X_train (list): List of strings X_train files patches.
        X_test (list): List of strings X_test files patches.
        y_train (list): List of strings y_train files patches.
        y_test (list): List of strings y_test files patches.
        save_dir (pathlib.PosixPath): Path for save results.

    """
    len_train, len_test = len(X_train), len(X_test)

    # Create datasets from data.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(
        load_data, num_parallel_calls=AUTOTUNE
    )

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).map(
        load_data, num_parallel_calls=AUTOTUNE
    )

    # Prepare data for training.
    train_dataset = train_dataset.map(normalize, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.shuffle(len_train * 2)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    test_dataset = test_dataset.map(normalize, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # TRAIN MODEL.

    # Init paths.
    save_dir.mkdir(parents=True, exist_ok=True)

    path_weights = (save_dir / "weights.hdf5").as_posix()
    path_history = (save_dir / "training.csv").as_posix()
    path_model = (save_dir / "model").as_posix()

    path_predict_fulls = (save_dir / "predict_fulls.csv").as_posix()
    path_predict_means = (save_dir / "predict_means.csv").as_posix()

    # Init callbacks.
    callback_save_best_weights = tf.keras.callbacks.ModelCheckpoint(
        filepath=path_weights,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )

    callback_csv_logger = tf.keras.callbacks.CSVLogger(
        filename=path_history, separator=",", append=False,
    )

    # Init optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Init model.
    model = UNet(
        input_shape=(IMAGE_NET_W, IMAGE_NET_H, CLASSES_COUNT),
        out_channels=CLASSES_COUNT,
        filters=[16, 32, 64, 128, 256],
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MSE,
        metrics=[tf.keras.metrics.Accuracy()],
    )

    # Fake predict for build model and summary.
    model.predict(next(iter(test_dataset))[0])
    model.summary()

    # Load weights.
    if MODEL_LOAD_WEIGHTS:
        model.load_weights(path_weights)

    # Train model.
    if MODEL_RUN_TRAIN:
        model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=EPOCHS,
            steps_per_epoch=len_train // BATCH_SIZE,
            validation_steps=len_test // BATCH_SIZE,
            callbacks=[callback_save_best_weights, callback_csv_logger],
        )

    # Evaluate model.
    evaluate = model.evaluate(test_dataset)
    print(tabulate(zip(model.metrics_names, evaluate), tablefmt="fancy_grid"))

    # Make prediction.
    predict_dataset = model.predict(test_dataset)

    # Save model.
    model.save(path_model)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # COMPARE WITH REAL COORDINATES.

    X_test_names = [pathlib.Path(x).stem for x in X_test]
    result_array = np.zeros((len_test, CLASSES_COUNT))

    for k, predict_array in enumerate(predict_dataset):
        for channel_type, channel_number in zip(CLASSES, range(CLASSES_COUNT)):
            # Get channel.
            predict_channel = predict_array[:, :, channel_number]

            # Get predicted coordinates with max value.
            maxes = np.argwhere(predict_channel.max() == predict_channel)
            x_p, y_p = np.mean(maxes, axis=0).astype(np.int)

            # Rescale predicted coordinates.
            x_p, y_p = rescale_coords(
                (x_p, y_p),
                src_size=(IMAGE_NET_W, IMAGE_NET_H),
                out_size=(IMAGE_SRC_W, IMAGE_SRC_H),
            )

            # Load text coordinates.
            filename = PWD_COORDS / channel_type / f"{X_test_names[k]}.txt"
            x_t, y_t = np.fromfile(filename, sep=",")

            # Compute distance.
            # ! NOTE: in this dataset, 1 cm == 11 pix.
            distance = np.sqrt((x_t - x_p) ** 2 + (y_t - y_p) ** 2) / 11.0

            # Save result.
            result_array[k][channel_number] = distance

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # DUMP RESULTS.

    # Dump full results to CSV.
    results = pd.DataFrame(columns=["filename", *CLASSES])

    for n, filename in enumerate(X_test_names):

        # Write rows of matrix to CSV.
        named_results = dict(zip(CLASSES, list(result_array[n])))
        named_results["filename"] = filename

        results = results.append(named_results, ignore_index=True)

    results.to_csv(path_predict_fulls, index=False, sep=";")

    # Dump mean results to CSV.
    results = pd.DataFrame(columns=["class", "mean", "max"])

    means = np.mean(result_array, axis=0)
    maxes = np.max(result_array, axis=0)

    for class_label, val_mean, val_max in zip(CLASSES, means, maxes):
        results = results.append(
            {"class": class_label, "mean": val_mean, "max": val_max},
            ignore_index=True,
        )

    results.to_csv(path_predict_means, index=False, sep=";")
    print(results)


def main():
    """Start training model process."""
    paths_images = get_images_paths(PWD_IMAGES)
    paths_markup = get_markup_paths(PWD_MARKUP)

    kf = sklearn.model_selection.KFold(
        n_splits=5, shuffle=False, random_state=None
    )

    kfold_current_number = 0

    for train_index, test_index in kf.split(paths_images, paths_markup):

        # Init results path.
        kfold_current_number += 1
        save_dir = pathlib.Path("results") / f"kfold_{kfold_current_number}"

        # Split data.
        X_train = [paths_images[x] for x in train_index]
        y_train = [paths_markup[x] for x in train_index]

        X_test = [paths_images[x] for x in test_index]
        y_test = [paths_markup[x] for x in test_index]

        # Run model training.
        train_model(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            save_dir=save_dir,
        )


if __name__ == "__main__":
    main()
