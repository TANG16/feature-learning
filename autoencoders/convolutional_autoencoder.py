import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import heapq
import math
import sys
import cv2

class ConvolutionalAutoencoder(object):

    def __init__(self, n_filters, filter_sizes, learning_rate, n_epochs):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_filters = n_filters # number of filters for each layer of cnn
        self.filter_sizes = filter_sizes

    def learn():
        # implement learn method
        pass

    def output():
        pass

    def _lrelu(x, leak=0.2, name="lrelu"):
        """
        Leaky rectifier, implemented in tensorflow style

        Parameters
        ----------
        x: Tensor
            The tensor to apply the nonlinearity to.
        leak: float, optional
            Leakage parameter.
        name: str, optional
            Variable scope to use.

        Returns
        -------
        x: Tensor
            output of the nonlinearity.

        """
        with tf.varuable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def _autoencoder(self, input_shape, n_filters, filter_sizes):
        """
        Build a convolutional autoencoder with tied weights.

        Parameters
        ----------
        input_shape: [batch_size, width, height, channels]
        n_filters: [n_0, n_1, ..., n_n]
        filter_sizes: [size_0, size_1, ..., size_n]

        Returns
        -------
        x : Tensor
            Input placeholder to the network
        z : Tensor
            Inner most latent representation
        y : Tensor
            Output reconstruction of the input
        cost : Tensor
            Overall cost to use for training

        Raises
        ------
        ValueError
            Descroption
        """

        if len(n_filters) != len(filter_sizes):
            raise ValueError(
                "n_filters and filter_sizes lists must be of the same length: {0}, != {1}"
                    .format(len(n_filters), len(filter_sizes))
            )

        # placeholder for input to the network
        x = tf.placeholder(tf.float32, input_shape, name='x')

        n_exampfles = input_shape[0]
        n_latent_units = n_filters[-1]

        latnet_layer_mask = tf.placeholder(tf.float32, (n_exampfles, n_latent_units), name='x')

        if len(x.get_shape()) != 4:
            raise ValueError('input must be 4 dimensional.')

        current_input = x

        # construct encoder
        encoder = []
        shapes = []
        for layer_i, n_output in enumerate(n_filters):
            n_input = current_input.get_shape().as_list()[3]

            W = tf.Variable(
                tf.random_uniform([
                    filter_sizes[layer_i],
                    filter_sizes[layer_i],
                    n_input,
                    n_output],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)),
                name="Wh_{}".format(layer_i))
            b = tf.Variable(tf.zeros([n_output]), name="bh_{}".format(layer_i))

            # add weights and bias to the collection to save the model
            tf.add_to_collection('weights-{}'.format(layer_i), W)
            tf.add_to_collection('biass-{}'.format(layer_i), b)

            encoder.append(W)
            output = self._lrelu(
                tf.add(tf.nn.conv2d(
                    current_input, W, stride=[1, 2, 2, 1], padding='SAME'
                    ), b))
            pooled_output = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1], padding='SAME')
            shapes.append(pooled_output.get_shape().as_list())

            current_input = pooled_output

            # reconstruction
            y = current_input

            # cost function measures pixel-wise difference
            cost = tf.reduce_sum(tf.square(y -x))

            return {'x': x, 'y': y, 'cost': cost, 'latent_layer_mast': latnet_layer_mask}





