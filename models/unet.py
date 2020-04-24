"""
U-Net: Convolutional Networks for Biomedical Image Segmentation.

Modified implementation of the original architecture
by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.
See more at: https://arxiv.org/pdf/1505.04597.pdf

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Copyright (c) 2020 Konstantin Dobratulin

This software is released under the MIT License.
https://opensource.org/licenses/MIT

"""

import tensorflow as tf


class UNet(tf.keras.Model):
    """U-Net: Convolutional Networks for Biomedical Image Segmentation."""

    def __init__(
        self, input_shape, out_channels, filters=None, final_dropout=0.20,
    ):
        """
        Init class object.

        Args:
            input_shape (tuple): Shape of input data with channel, like (512, 512, 3).
            out_channels (int): Count of input channels.
            filters (list): Count of filters at each level of the model.
            final_dropout (float): Dropout for final result.

        Returns:
            tensorflow.keras.Model: U-Net TensorFlow implementation.

        """
        super().__init__()

        # Init count of filters for convolution layers.
        if filters is None:
            filters = [8, 16, 32, 64, 128]

        # Base input layer.
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)

        # Convolution and max-pooling.
        self.maxpooling = tf.keras.layers.MaxPool2D()

        self.conv_d_1 = UNetConvDown(filters[0])
        self.conv_d_2 = UNetConvDown(filters[1])
        self.conv_d_3 = UNetConvDown(filters[2])
        self.conv_d_4 = UNetConvDown(filters[3])
        self.conv_d_5 = UNetConvDown(filters[4])

        # Transpose convolution, up-sampling and concatenation.
        self.upsample = tf.keras.layers.UpSampling2D(interpolation="bilinear")

        self.conv_u_1 = UNetConvUp(filters[3])
        self.concat_1 = tf.keras.layers.Concatenate()
        self.conv_u_2 = UNetConvUp(filters[2])
        self.concat_2 = tf.keras.layers.Concatenate()
        self.conv_u_3 = UNetConvUp(filters[1])
        self.concat_3 = tf.keras.layers.Concatenate()
        self.conv_u_4 = UNetConvUp(filters[0])
        self.concat_4 = tf.keras.layers.Concatenate()

        # Final layer.
        self.final_conv = tf.keras.layers.Conv2D(out_channels, 1)
        self.final_actv = tf.keras.layers.ReLU()
        self.final_drop = tf.keras.layers.Dropout(final_dropout)

    def call(self, inputs):
        """
        Implement the model's forward pass.

        Args:
            inputs (tensorflow.Tensor): Input tensor.

        Returns:
            tensorflow.Tensor: Output tensor.

        """
        inputs = self.input_layer(inputs)

        # Downsample and max-pooling.
        down_conv_1 = self.conv_d_1(inputs)
        inputs = self.maxpooling(down_conv_1)

        down_conv_2 = self.conv_d_2(inputs)
        inputs = self.maxpooling(down_conv_2)

        down_conv_3 = self.conv_d_3(inputs)
        inputs = self.maxpooling(down_conv_3)

        down_conv_4 = self.conv_d_4(inputs)
        inputs = self.maxpooling(down_conv_4)

        inputs = self.conv_d_5(inputs)

        # Upsample and concatenation.
        inputs = self.upsample(inputs)
        inputs = self.concat_1([inputs, down_conv_4])
        inputs = self.conv_u_1(inputs)

        inputs = self.upsample(inputs)
        inputs = self.concat_2([inputs, down_conv_3])
        inputs = self.conv_u_2(inputs)

        inputs = self.upsample(inputs)
        inputs = self.concat_3([inputs, down_conv_2])
        inputs = self.conv_u_3(inputs)

        inputs = self.upsample(inputs)
        inputs = self.concat_4([inputs, down_conv_1])
        inputs = self.conv_u_4(inputs)

        # Final layer.
        inputs = self.final_conv(inputs)
        inputs = self.final_actv(inputs)
        inputs = self.final_drop(inputs)

        return inputs


class UNetConvDown(tf.keras.Model):
    """U-Net Double Convolution 2D module."""

    def __init__(self, filters, convs_dropout=0.0, final_dropout=0.0):
        """
        Init U-Net Double Convolution Down model.

        Args:
            filters (int): Count of filters in convolutions.
            convs_dropout (float): Dropout value between convolutions blocks.
            final_dropout (float): Dropout value at final result.

        Returns:
            tensorflow.keras.Model: U-Net Double Convolution Down model.

        """
        super().__init__()

        padding = [[0, 0], [0, 0], [0, 0], [0, 0]]

        self.c_1 = tf.keras.layers.Conv2D(filters, 3, 1, padding=padding)
        self.a_1 = tf.keras.layers.PReLU()
        self.n_1 = tf.keras.layers.BatchNormalization()
        self.d_1 = tf.keras.layers.Dropout(convs_dropout)

        self.c_2 = tf.keras.layers.Conv2D(filters, 3, 1, padding=padding)
        self.a_2 = tf.keras.layers.PReLU()
        self.n_2 = tf.keras.layers.BatchNormalization()
        self.d_2 = tf.keras.layers.Dropout(final_dropout)

    def call(self, inputs):
        """
        Implement the model's forward pass.

        Args:
            inputs (tensorflow.Tensor): Input tensor.

        Returns:
            tensorflow.Tensor: Output tensor.

        """
        padding = [[0, 0], [1, 1], [1, 1], [0, 0]]

        inputs = tf.pad(inputs, tf.constant(padding), "SYMMETRIC")
        inputs = self.a_1(self.c_1(inputs))
        inputs = self.d_1(self.n_1(inputs))

        inputs = tf.pad(inputs, tf.constant(padding), "SYMMETRIC")
        inputs = self.a_2(self.c_2(inputs))
        inputs = self.d_2(self.n_2(inputs))

        return inputs


class UNetConvUp(tf.keras.Model):
    """U-Net Double Convolution Transpose 2D module."""

    def __init__(self, filters, convs_dropout=0.0, final_dropout=0.0):
        """
        Init U-Net Double Convolution Up model.

        Args:
            filters (int): Count of filters in convolutions.
            convs_dropout (float): Dropout value between convolutions blocks.
            final_dropout (float): Dropout value at final result.

        Returns:
            tensorflow.keras.Model: U-Net Double Convolution Up model.

        """
        super().__init__()

        self.c_1 = tf.keras.layers.Conv2DTranspose(filters, 3, 1, "SAME")
        self.a_1 = tf.keras.layers.PReLU()
        self.n_1 = tf.keras.layers.BatchNormalization()
        self.d_1 = tf.keras.layers.Dropout(convs_dropout)
        self.c_2 = tf.keras.layers.Conv2DTranspose(filters, 3, 1, "SAME")
        self.a_2 = tf.keras.layers.PReLU()
        self.n_2 = tf.keras.layers.BatchNormalization()
        self.d_2 = tf.keras.layers.Dropout(final_dropout)

    def call(self, inputs):
        """
        Implement the model's forward pass.

        Args:
            inputs (tensorflow.Tensor): Input tensor.

        Returns:
            tensorflow.Tensor: Output tensor.

        """
        inputs = self.a_1(self.c_1(inputs))
        inputs = self.d_1(self.n_1(inputs))
        inputs = self.a_2(self.c_2(inputs))
        inputs = self.d_2(self.n_2(inputs))

        return inputs
