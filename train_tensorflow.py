#!/usr/bin/env python3

"""
Train TensorFlow model script.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Copyright (c) 2020 Konstantin Dobratulin

This software is released under the MIT License.
https://opensource.org/licenses/MIT

"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tabulate import tabulate

from models.unet import UNet

# -----------------------------------------------------------------------------
# Customizable hyperparameters.

# TODO Remove hard-coded dataset path
# TODO Move hyperparameters to YAML.
# TODO Reference type list?

DATA_PWD = "/home/zsxoff/neural-orthodontic-dataset"
EXPERT_NAME = "expert_1"
REFERENCE_TYPE = "A"

BATCH_SIZE = 8
EPOCHS = 64

IMAGE_NET_W = 512
IMAGE_NET_H = 512

REFERENCE_POINT_SIZE = 128
TRAIN_SIZE_PERCENTS = 50

LOAD_WEIGHTS = False
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Constant hyperparameters.

# ! Please do not modify this parameters if you are not 100% sure
# ! that you know why they are needed.

IMAGE_SRC_W = 2400
IMAGE_SRC_H = 2000

PROJECT_NAME = f"result_{EXPERT_NAME}_{REFERENCE_TYPE}"

RANDOM_SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE
tf.keras.backend.clear_session()
tf.random.set_seed(RANDOM_SEED)
# -----------------------------------------------------------------------------


def images_coords_paths(images_dir, coords_dir):
    """
    Return all images and all coordinates files paths sorted in lexicographic order.

    Args:
        images_dir (str): Base images directory of dataset.
        coords_dir (str): Base coordinates directory of dataset.

    Returns:
        tuple: Tuple contains arrays of paths of images and relevant coordinates.

    """
    images_paths = list(
        map(lambda x: x.as_posix(), sorted(Path(images_dir).glob("*.jpg")))
    )

    coords_paths = list(
        map(lambda x: x.as_posix(), sorted(Path(coords_dir).glob("*.txt")))
    )

    return images_paths, coords_paths


def read_coords_file(file_path):
    """
    Read file_path and convert coordinates paths to rescaled coordinates pairs.

    Args:
        file_path (str): File path.

    Returns:
        tuple: Tuple of coordinates (x, y).

    """
    return rescale_coords(
        coords=np.fromfile(file_path, sep=","),
        src_size=(IMAGE_SRC_W, IMAGE_SRC_H),
        out_size=(IMAGE_NET_W, IMAGE_NET_H),
    )


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


def mask_from_coord(coords):
    """
    Create zeros image with normal-distributed spot.

    Args:
        coords (list): Coordinates array of pixel in original image.

    Returns:
        numpy.ndarray: Zeroes matrix with normal distributed spot.

    """
    spot_size = REFERENCE_POINT_SIZE

    x_meshgrid, y_meshgrid = np.meshgrid(
        np.linspace(-spot_size, spot_size, spot_size),
        np.linspace(-spot_size, spot_size, spot_size),
    )

    z_meshgrid = np.sqrt(x_meshgrid ** 2 + y_meshgrid ** 2)

    # Create normal distribution.
    # Constants 3.5 and 4096 are selected empirically.
    # It is used for resize spot and limit max value from 0.0 to 1.0.
    var = spot_size // 3.5

    distribution = (
        (1 / (var * np.sqrt(2 * np.pi)))
        * np.exp(-(z_meshgrid ** 2) / (2 * var ** 2))
        * 4096
    )

    distribution[distribution < 0] = 0.0

    # Get and resize coordinates to new image size.
    img_w, img_h = (IMAGE_NET_W, IMAGE_NET_H)

    x_coord = int(coords[0])
    y_coord = int(coords[1])

    # Plot normal distributed spot in black image.
    image = np.zeros((img_w, img_h), dtype=np.float)

    height = spot_size // 2

    # Check borders.
    r_w = x_coord + height if x_coord + height < img_w else img_w
    r_h = y_coord + height if y_coord + height < img_h else img_h
    l_w = x_coord - height if x_coord - height > 0 else 0
    l_h = y_coord - height if y_coord - height > 0 else 0

    # Plot distribution.
    image[l_w:r_w, l_h:r_h] = distribution[: r_w - l_w, : r_h - l_h]

    # Reshape for TensorFlow.
    image = np.reshape(image, (image.shape[0], image.shape[1], 1))

    # Convert to float32.
    return image.astype(np.float32)


@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def process_image(file_path):
    """
    Load grayscale JPEG image file.

    Args:
        file_path (tensorflow.string): File name for image file.

    Returns:
        tensorflow.Tensor: Grayscale image (size of WxHx1).

    """
    # Load raw data.
    img = tf.io.read_file(file_path)

    # Convert compressed string to a tensor.
    img = tf.image.decode_jpeg(img, channels=1)

    # Use `convert_image_dtype` to convert to int32 in the [0, MAX] range.
    img = tf.image.convert_image_dtype(img, tf.int32)

    # Resize image.
    return tf.image.resize(img, [IMAGE_NET_W, IMAGE_NET_H])


@tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
def process_coord(file_path):
    """
    Load coordinates from file and generate mask for network.

    Args:
        file_path (tensorflow.string): File name for coordinates file.

    Returns:
        tensorflow.Tensor: Generated mask image (size of WxHx1).

    """
    # Get coordinates from file.
    crd = tf.numpy_function(read_coords_file, [file_path], tf.int64)

    # Generate mask image by coordinates.
    img = tf.numpy_function(mask_from_coord, [crd], tf.float32)

    # Restore shape for NumPy function conversion.
    img.set_shape([IMAGE_NET_W, IMAGE_NET_H, 1])

    # Use `convert_image_dtype` to convert to floats in the [0, 1] range.
    return tf.image.convert_image_dtype(img, tf.float32)


def load_data(x_png_file, y_png_file):
    """
    Load data map function for TensorFlow.

    Args:
        x_png_file (tensorflow.string): File name for image.
        y_png_file (tensorflow.string): File name for coordinates file.

    Returns:
        tuple: Tuple of TensorFlow tensors for image and mask (size of WxHx1).

    """
    return process_image(x_png_file), process_coord(y_png_file)


# TODO Move visualization to file.

# def show_batch(image_batch, label_batch):
#     plt.figure(figsize=(14, 10))
#     rows = 2
#     cols = 4
#
#     it_b = iter(image_batch)
#     it_l = iter(label_batch)
#
#     for n in range(cols * rows):
#         plt.subplot(rows, cols, n + 1)
#         plt.imshow(next(it_b)[..., 0] - next(it_l)[..., 0])
#
#     plt.show()


def mse_argmaxs(y_true, y_pred):
    """
    Return MSE of indices of the maximum values along an axis of vectors.

    Args:
        y_true (tensorflow.Tensor): True values vector.
        y_pred (tensorflow.Tensor): Predicted values vector.

    Returns:
        tensorflow.Tensor: A tensor with the mean of elements.

    """
    return tf.keras.losses.MSE(tf.math.argmax(y_true), tf.math.argmax(y_pred))


def main():
    """Start training model process."""
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # LOAD DATA.

    # Init data dirs.
    images_pwd = Path(DATA_PWD) / "images"
    coords_pwd = Path(DATA_PWD) / "coords-txt" / EXPERT_NAME / REFERENCE_TYPE

    # Load images and coordinates of references arrays.
    images_paths, coords_paths = images_coords_paths(images_pwd, coords_pwd)

    # Split paths by train and test.
    len_train = int(len(images_paths) / 100 * TRAIN_SIZE_PERCENTS)
    X_train, X_test = images_paths[:len_train], images_paths[len_train:]
    y_train, y_test = coords_paths[:len_train], coords_paths[len_train:]
    len_train, len_test = len(X_train), len(X_test)

    # Load train and test.
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(
        load_data, num_parallel_calls=AUTOTUNE
    )

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).map(
        load_data, num_parallel_calls=AUTOTUNE
    )

    # Prepare data for training.
    train_dataset = train_dataset.shuffle(len_train * 2)
    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    test_dataset = test_dataset.shuffle(len_test * 2)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # TRAIN MODEL.

    # Create results directory.
    (Path("results") / PROJECT_NAME).mkdir(parents=True, exist_ok=True)

    # Callback - save best weights.
    path_weights = Path("results") / PROJECT_NAME / "weights.hdf5"
    path_weights = path_weights.as_posix()

    callback_save_best_weights = tf.keras.callbacks.ModelCheckpoint(
        filepath=path_weights,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=0,
    )

    # Callback - save history as CSV.
    path_history_csv = Path("results") / PROJECT_NAME / "training.hdf5"
    path_history_csv = path_history_csv.as_posix()

    callback_csv_logger = tf.keras.callbacks.CSVLogger(
        filename=path_history_csv, separator=",", append=False,
    )

    # Callback - reduce LR.
    callback_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.01,
        patience=10,
        verbose=1,
        mode="min",
        cooldown=0,
        min_lr=1e-5,
    )

    # Init optimizer.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Train model.
    if not LOAD_WEIGHTS:
        model = UNet(input_shape=(IMAGE_NET_W, IMAGE_NET_H, 1), out_channels=1)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.MSE,
            metrics=[tf.keras.metrics.Accuracy(), mse_argmaxs],
        )

        model.build(input_shape=(None, IMAGE_NET_W, IMAGE_NET_H, 1))

        model.summary()

        model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=EPOCHS,
            steps_per_epoch=len_train // BATCH_SIZE,
            callbacks=[
                callback_save_best_weights,
                callback_csv_logger,
                callback_reduce_lr,
            ],
        )

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # GET RESULTS.

    # Re-init model with best weights.
    model = UNet(input_shape=(IMAGE_NET_W, IMAGE_NET_H, 1), out_channels=1)

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MSE,
        metrics=[tf.keras.metrics.Accuracy(), mse_argmaxs],
    )

    model.build(input_shape=(None, IMAGE_NET_W, IMAGE_NET_H, 1))

    model.load_weights(path_weights)

    # Evaluate model.
    evaluate = model.evaluate(test_dataset)
    print(tabulate(zip(model.metrics_names, evaluate), tablefmt="fancy_grid"))

    # Make prediction.
    # predict_dataset = model.predict(test_dataset)

    model.predict(test_dataset)

    # TODO Visualize prediction.

    # test_image_batch, test_label_batch = next(iter(test_dataset))
    # show_batch(predict_dataset, test_label_batch)


if __name__ == "__main__":
    main()
