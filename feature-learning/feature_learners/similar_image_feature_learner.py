
import os
import numpy as np
from vionel_feature_learning.autoencoders.convolutional.autoencoder import ConvolutionalFeatureFinder
from vionel_feature_learning.autoencoders.similarity_space.autoencoder import SimilaritySpaceAutoencoder
from vionel_feature_learning.utils.image_utils import read_image_bgr, resize_image, bgr_to_lab, normalize_image


class SimilarImageFeatureLearner:

    def __init__(self, images_path):
        self.images_path = images_path
        self.image_width_height = 64

    def train(self):

        print "Preparing images.."
        image_file_paths, prepared_images = self._build_data_set(
            self.images_path,
            self.image_width_height
        )

        def activation_map_mean(activation_maps):

            n_samples, n_features, width, height = activation_maps.shape

            activations_as_vectors = np.reshape(
                activation_maps,
                (n_samples, n_features, height * width)
            )

            return np.mean(
                activations_as_vectors,
                axis=2
            )

        cae = ConvolutionalFeatureFinder(
            n_filters=[40, 40, 50],
            filter_sizes=[3, 5, 7],
            learning_rate=0.005,
            num_epochs=5
        )

        print "Learning CNN features.."
        cnn_activations = cae.learn(prepared_images, visualize_result=False)
        cnn_activation_means = activation_map_mean(cnn_activations)

        ssae = SimilaritySpaceAutoencoder(
            n_features=60,
            learning_rate=0.01
        )

        print "Optimizing similarity space.."
        ss_activations = ssae.learn(cnn_activation_means, visualize_result=False)

        print "Similarity space complete"

        return image_file_paths, ss_activations

    @staticmethod
    def _build_data_set(dir_path, width_height):

        def prepare_image(bgr_image):
            resized_bgr_image = resize_image(bgr_image, width_height, width_height)
            image_lab = bgr_to_lab(resized_bgr_image)
            image_lab_brighter_side_flipped = flip_brighter_side_left(image_lab)
            return normalize_image(image_lab_brighter_side_flipped)

        def flip_brighter_side_left(lab_image):

            rows, cols, channels = lab_image.shape

            left_side = lab_image[:, :cols / 2, 0][0]
            right_side = lab_image[:, cols / 2:, 0][0]

            is_right_side_brighter = np.mean(right_side) > np.mean(left_side)

            return np.fliplr(lab_image) if is_right_side_brighter else lab_image

        file_names = os.listdir(dir_path)
        file_names.sort()
        image_file_paths = []
        images = []
        c = 0
        for file_name in file_names:

            c += 1
            if c % 1000 == 0:
                print c, "images prepared"

            file_path = dir_path + "/" + file_name

            image_bgr = read_image_bgr(file_path)

            prepared_image = prepare_image(image_bgr)

            image_file_paths.append(file_path)
            images.append(prepared_image)

        return image_file_paths, np.array(images)