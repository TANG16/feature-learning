import numpy as np
from vionel_feature_learning.autoencoders.convolutional.autoencoder import ConvolutionalFeatureFinder
from vionel_feature_learning.autoencoders.similarity_space.autoencoder import SimilaritySpaceAutoencoder


images_path = "/home/erik/data/ads_datasets/welovead_64x64_lab_flipped.npy"
n_images = 50000
output_feature_data_path = "/home/erik/data/ads_datasets/welovead_64x64_lab_flipped_latent_features.npy"


def activation_map_mean(cnn_activations):

    n_samples, n_features, width, height = cnn_activations.shape

    activations_as_vectors = np.reshape(
        cnn_activations,
        (n_samples, n_features, height * width)
    )

    return np.mean(
        activations_as_vectors,
        axis=2
    )

print "Loading images.."
images_lab = np.load(images_path)[:n_images]

cae = ConvolutionalFeatureFinder(
    n_filters=[40, 40, 50],
    filter_sizes=[3, 5, 7],
    learning_rate=0.005,
    num_epochs=5
)

print "Learning cnn features.."
cnn_activations = cae.learn(images_lab, visualize_result=True)

print "Calculating mean activations.."
cnn_activation_means = activation_map_mean(cnn_activations)

ssae = SimilaritySpaceAutoencoder(
    n_features=60,
    learning_rate=0.01
)

print "Learning similarity space.."
ss_activations = ssae.learn(cnn_activation_means, visualize_result=True)

np.save(output_feature_data_path, ss_activations)
print "Saved latent features to " + output_feature_data_path