import math

import numpy as np
import tensorflow as tf


class SimilaritySpaceAutoencoder(object):

    def __init__(self, n_features, learning_rate):
        self.num_features = n_features
        self.learning_rate = learning_rate

    def learn(self, training_data, visualize_result=False):

        input_scaling = {
            "max": training_data.max(axis=0),
            "min": training_data.min(axis=0)
        }

        ae = self._autoencoder(
            training_data=training_data
        )

        return self._activation(training_data, ae, input_scaling, visualize_result)

    def _autoencoder(self, training_data):

        input = self._scale_to_minus_plus_one(self._add_noise(training_data, 0.05))
        output = self._scale_to_minus_plus_one(training_data)

        # Autoencoder with 1 hidden layer
        n_samp, n_input = training_data.shape
        n_hidden = self.num_features
        batch_size = min(100, n_samp)

        x = tf.placeholder("float", [None, n_input])
        # Weights and biases to hidden layer
        Wh = tf.Variable(tf.random_uniform((n_input, n_hidden), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
        bh = tf.Variable(tf.zeros([n_hidden]))
        h = tf.nn.tanh(tf.matmul(x, Wh) + bh)
        # Weights and biases to hidden layer
        Wo = tf.transpose(Wh)  # tied weights
        bo = tf.Variable(tf.zeros([n_input]))
        y = tf.nn.tanh(tf.matmul(h, Wo) + bo)
        # Objective functions
        y_ = tf.placeholder("float", [None, n_input])
        reconstruction_error = tf.reduce_mean(tf.square(y_-y))

        h_means = tf.reduce_mean(h, 0)

        h_minus_means = tf.sub(h, h_means)
        covariance = tf.mul(tf.matmul(tf.transpose(h_minus_means), h_minus_means), 1.0 / batch_size)
        V = tf.reduce_mean(tf.square(h_minus_means), 0)
        std = tf.sqrt(V)
        std_matrix = tf.reshape(std, [n_hidden, 1])
        std_products = tf.matmul(std_matrix, tf.transpose(std_matrix))

        correlation = covariance / std_products

        non_diagonal_mask = tf.sub(tf.ones([n_hidden]), tf.diag(tf.ones([n_hidden])))
        correlation_non_diag = tf.mul(non_diagonal_mask, correlation)

        correlation_score = tf.reduce_mean(tf.abs(correlation_non_diag))
        correlation_peak = tf.reduce_max(tf.abs(correlation_non_diag))
        standard_deviation_score = tf.reduce_min(std)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

        cost_function = \
            1 * reconstruction_error + \
            0.1 * correlation_score + \
            0.05 * correlation_peak + \
            -0.25 * standard_deviation_score

        train_step = optimizer.minimize(cost_function)

        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)

        n_rounds = 5000
        for i in range(n_rounds):
            sample = np.random.randint(n_samp, size=batch_size)
            batch_xs = input[sample][:]
            batch_ys = output[sample][:]

            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

            if i % 1000 == 0:
                print (i, sess.run(cost_function, feed_dict={x: batch_xs, y_: batch_ys}))

        return {
            "Wh": np.matrix(sess.run(Wh)),
            "bh": np.matrix(sess.run(bh)),
            "bo": np.matrix(sess.run(bo))
        }

    def _activation(self, test_data, autoencoder, input_scaling, visualize_result):

        training_input_max = input_scaling["max"]
        training_input_min = input_scaling["min"]

        x = self._scale_to_minus_plus_one(test_data, max=training_input_max, min=training_input_min)

        Wh = autoencoder["Wh"]
        bh = autoencoder["bh"]
        bo = autoencoder["bo"]

        h = np.tanh(x * Wh + bh)

        if visualize_result:
            from visualize import visualize
            visualize(Wh, bh, bo, h, test_data)

        return h

    def _scale_to_minus_plus_one(self, input, max=None, min=None):
        scaled_zero_to_one = self._min_max_normalize(input, max=max, min=min)
        return (scaled_zero_to_one * 2) - 1

    @staticmethod
    def _min_max_normalize(input, max=None, min=None):
        input_max_values = max if max is not None else input.max(axis=0)
        input_min_values = min if min is not None else input.min(axis=0)
        return (input - input_min_values) / (input_max_values - input_min_values)

    @staticmethod
    def _add_noise(input, noise_level):
        noise_scalar = np.array(noise_level * input.mean(axis=0))
        noise = np.array(np.random.random_sample(input.shape) - 0.5)
        for idx, _ in enumerate(noise):
            noise[idx] = noise[idx] * noise_scalar
        return input + noise
