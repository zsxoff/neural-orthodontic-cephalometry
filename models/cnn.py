"""
CNN: Convolutional Network for Biomedical Image Segmentation.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

Copyright (c) 2020 Konstantin Dobratulin

This software is released under the MIT License.
https://opensource.org/licenses/MIT

"""

import tensorflow as tf


class CNN(tf.keras.Model):
    """CNN: Convolutional Network for Biomedical Image Segmentation."""

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
            filters = [16, 32, 64, 128, 256]

        # Base input layer.
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)

        # Downsample.
        self.maxpooling = tf.keras.layers.MaxPool2D()

        self.conv_d_1 = CNNConvDown(filters[0])
        self.conv_d_2 = CNNConvDown(filters[1])
        self.conv_d_3 = CNNConvDown(filters[2])
        self.conv_d_4 = CNNConvDown(filters[3])
        self.conv_d_5 = CNNConvDown(filters[4])

        self.up_1 = tf.keras.layers.Conv2DTranspose(filters[3], 3, 2, "SAME")
        self.up_1_act = tf.keras.layers.ReLU()

        self.up_2 = tf.keras.layers.Conv2DTranspose(filters[2], 3, 2, "SAME")
        self.up_2_act = tf.keras.layers.ReLU()

        self.up_3 = tf.keras.layers.Conv2DTranspose(filters[1], 3, 2, "SAME")
        self.up_3_act = tf.keras.layers.ReLU()

        self.up_4 = tf.keras.layers.Conv2DTranspose(filters[0], 3, 2, "SAME")
        self.up_4_act = tf.keras.layers.ReLU()

        self.conv_u_1 = CNNConvUp(filters[3])
        self.conv_u_2 = CNNConvUp(filters[2])
        self.conv_u_3 = CNNConvUp(filters[1])
        self.conv_u_4 = CNNConvUp(filters[0])

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

        # Downsample.

        inputs = self.conv_d_1(inputs)
        inputs = self.maxpooling(inputs)
        inputs = self.conv_d_2(inputs)
        inputs = self.maxpooling(inputs)
        inputs = self.conv_d_3(inputs)
        inputs = self.maxpooling(inputs)
        inputs = self.conv_d_4(inputs)
        inputs = self.maxpooling(inputs)
        inputs = self.conv_d_5(inputs)

        # Upsample.

        inputs = self.up_1(inputs)
        inputs = self.up_1_act(inputs)
        inputs = self.conv_u_1(inputs)

        inputs = self.up_2(inputs)
        inputs = self.up_2_act(inputs)
        inputs = self.conv_u_2(inputs)

        inputs = self.up_3(inputs)
        inputs = self.up_3_act(inputs)
        inputs = self.conv_u_3(inputs)

        inputs = self.up_4(inputs)
        inputs = self.up_4_act(inputs)
        inputs = self.conv_u_4(inputs)

        # Final layer.
        inputs = self.final_conv(inputs)
        inputs = self.final_actv(inputs)
        inputs = self.final_drop(inputs)

        return inputs


class CNNConvDown(tf.keras.Model):
    """CNN Double Convolution 2D module."""

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

        self.c_1 = tf.keras.layers.Conv2D(filters, 3, 1, padding="SAME")
        self.a_1 = tf.keras.layers.PReLU()
        self.n_1 = tf.keras.layers.BatchNormalization()
        self.d_1 = tf.keras.layers.Dropout(convs_dropout)

        self.c_2 = tf.keras.layers.Conv2D(filters, 3, 1, padding="SAME")
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


class CNNConvUp(tf.keras.Model):
    """CNN Double Convolution Transpose 2D module."""

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
