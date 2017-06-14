
import tensorflow as tf
import numpy as np
import math


class ConvolutionalFeatureFinder(object):

    def __init__(self, n_filters, filter_sizes, learning_rate, num_epochs):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.n_filters = n_filters
        self.filter_sizes = filter_sizes

    def learn(self, training_data, visualize_result=False):

        images = training_data

        width, height, num_channels = images[0].shape

        batch_size = 100
        n_epochs = self.num_epochs
        n_examples = len(images)
        n_features = self.n_filters[-1]

        if n_examples < batch_size:
            raise ValueError("Not enough training examples. Minimum {}, got {}".format(batch_size, n_examples))

        mean_img = np.mean(images, axis=0)

        ae = self._autoencoder(
            input_shape=[None, width, height, num_channels],
            n_filters=self.n_filters,
            filter_sizes=self.filter_sizes
        )

        learning_rate = self.learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        no_op_latent_layer_mask = np.ones((1, n_features))

        # Fit all training data
        for epoch_i in range(n_epochs):
            batches = self._random_batch_generator(images, batch_size)
            for batch_i in range(n_examples // batch_size):
                batch_xs = batches.next()
                train = np.array([img - mean_img for img in batch_xs])
                sess.run(optimizer, feed_dict={ae['x']: train, ae['latent_layer_mask']: no_op_latent_layer_mask})
            print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train, ae['latent_layer_mask']: no_op_latent_layer_mask}))

        def output(latent_layer_mask):

            activations = []
            reconstructions = []
            n_examples = len(images)

            def from_to(start_index, end_index):
                batch_xs = images[start_index:end_index]
                test = np.array([img - mean_img for img in batch_xs])

                recon, z = sess.run([ae['y'], ae['z']], feed_dict={ae['x']: test, ae['latent_layer_mask']: latent_layer_mask})
                reconstructions.extend(recon)
                activations.extend(z)

            current_end_index = 0
            for batch_i in range(n_examples // batch_size):
                start_index = batch_i * batch_size
                end_index = start_index + batch_size
                from_to(start_index, end_index)
                current_end_index = end_index

            remains = n_examples % batch_size
            if remains > 0:
                from_to(
                    start_index=current_end_index,
                    end_index=current_end_index + remains
                )

            return np.array(reconstructions), np.array(activations)

        original_reconstructions, original_latent_activations = output(no_op_latent_layer_mask)

        if visualize_result:
            from visualize import visualize
            visualize(images, original_latent_activations, original_reconstructions, output, mean_img)

        all_activations_reordered = np.rollaxis(original_latent_activations, 3, 1)

        return all_activations_reordered

    @staticmethod
    def _lrelu(x, leak=0.2, name="lrelu"):
        """Leaky rectifier.

        Parameters
        ----------
        x : Tensor
            The tensor to apply the nonlinearity to.
        leak : float, optional
            Leakage parameter.
        name : str, optional
            Variable scope to use.

        Returns
        -------
        x : Tensor
            Output of the nonlinearity.
        """
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * abs(x)

    def _autoencoder(
            self,
            input_shape,
            n_filters,
            filter_sizes
    ):
        """Build a deep convolutional autoencoder w/ tied weights.

        Parameters
        ----------
        input_shape : [batch_size, width, height, channels]
            Description
        n_filters : [n_0, n_1, ..., n_n]
            Description
        filter_sizes : [size_0, size_1, ..., size_n]
            Description

        Returns
        -------
        x : Tensor
            Input placeholder to the network
        z : Tensor
            Inner-most latent representation
        y : Tensor
            Output reconstruction of the input
        cost : Tensor
            Overall cost to use for training

        Raises
        ------
        ValueError
            Description
        """

        if len(n_filters) != len(filter_sizes):
            raise ValueError(
                "n_filters and filter_sizes lists must be of same length: {} != {}"
                .format(len(n_filters), len(filter_sizes))
            )

        # input to the network
        x = tf.placeholder(
            tf.float32, input_shape, name='x')

        n_examples = input_shape[0]
        n_latent_units = n_filters[-1]

        latent_layer_mask = tf.placeholder(
            tf.float32, (n_examples, n_latent_units), name='x')

        if len(x.get_shape()) != 4:
            raise ValueError('Unsupported input dimensions')

        current_input = x

        # Build the encoder
        encoder = []
        shapes = []
        for layer_i, n_output in enumerate(n_filters):

            n_input = current_input.get_shape().as_list()[3]

            W = tf.Variable(
                tf.random_uniform([
                    filter_sizes[layer_i],
                    filter_sizes[layer_i],
                    n_input, n_output],
                    -1.0 / math.sqrt(n_input),
                    1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([n_output]))

            encoder.append(W)
            output = self._lrelu(
                tf.add(tf.nn.conv2d(
                    current_input, W, strides=[1, 1, 1, 1], padding='SAME'), b))

            output_pooled = tf.nn.max_pool(output, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            shapes.append(output_pooled.get_shape().as_list())

            current_input = output_pooled

        # store the latent convolved representation
        z = current_input
        current_input = tf.mul(current_input, latent_layer_mask)
        encoder.reverse()
        shapes.reverse()

        # Build the decoder using the same weights
        for layer_i, shape in enumerate(shapes):
            W = encoder[layer_i]
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))

            new_height = shape[1] * 2
            new_width = shape[2] * 2
            input_unpooled = tf.image.resize_images(current_input, new_height, new_width)

            batch_size = tf.shape(current_input)[0]
            n_channels = W.get_shape().as_list()[2]

            output = self._lrelu(tf.add(
                tf.nn.conv2d_transpose(
                    input_unpooled, W,
                    tf.pack([batch_size, new_height, new_width, n_channels]),
                    strides=[1, 1, 1, 1], padding='SAME'), b))

            current_input = output

        # reconstruction after the network
        y = current_input

        # cost function measures pixel-wise difference
        cost = tf.reduce_sum(tf.square(y - x))

        return {'x': x, 'z': z, 'y': y, 'cost': cost, 'latent_layer_mask': latent_layer_mask}

    @staticmethod
    def _random_batch_generator(images, batch_size=100):

        # Shuffle each epoch
        current_permutation = np.random.permutation(range(len(images)))
        epoch_images = images[current_permutation, ...]

        current_batch_idx = 0
        while current_batch_idx < len(images):
            end_idx = min(
                current_batch_idx + batch_size, len(images))
            subset = epoch_images[current_batch_idx:end_idx]
            current_batch_idx += batch_size
            yield subset

    @staticmethod
    def _calculate_latent_activations(sess, autoencoder, images, mean_img):
        all_images_norm = np.array([img - mean_img for img in images])
        return sess.run(autoencoder['z'], feed_dict={autoencoder['x']: all_images_norm})

