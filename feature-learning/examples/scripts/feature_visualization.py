import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from feature_learning import utils
from utils import semantic_hash_search


feature_data_path = "/data/ads_datasets/welovead_64x64_lab_flipped_latent_features.npy"
images_path = "/data/ads_datasets/welovead_64x64_rgb.npy"


def plot(axis, index, row_size, top_image):
    row = index / row_size
    column = index % row_size
    axis[row][column].imshow(top_image)

print "Loading feature data.."
feature_data = np.load(feature_data_path)

print "Loading images.."
images = np.load(images_path)

if len(images) != len(feature_data):
    raise ValueError(
        "Number of images and feature samples does not match: {} != {}".format(len(images), len(feature_data))
    )


def init():
    for i in range(0, n):
        plot(axs, i, 10, images[0])
    fig.show()
    plt.draw()
    plt.show(block=False)
    time.sleep(1)

print "Find top and bottom 5 posters"
n = 40
n_top_partition = int(0.01 * len(images))
row_size = 10
fig, axs = plt.subplots(4, row_size, figsize=(28, 28))
feature_index = 0
n_images, n_features = feature_data.shape

init()

while True:
    try:

        feature_index_data = feature_data[:, feature_index]

        def top(vector):

            unsorted_top_partition = np.argpartition(vector, -n_top_partition)[-n_top_partition:]

            unsorted_subsample = unsorted_top_partition[np.random.choice(n_top_partition, n)]

            def maximum(i1, i2):
                return \
                    -1 if vector[i1] > vector[i2] else\
                    1 if vector[i1] < vector[i2] else \
                    0

            return sorted(unsorted_subsample, cmp=maximum)

        top_feature_examples = top(feature_index_data)

        print "Plotting.."

        for idx in range(0, n):
            plot(axs, idx, row_size, images[top_feature_examples[idx]])

        fig.show()
        plt.draw()
        plt.show(block=False)

        feature_index = int(raw_input(
            "Feature [{} to {}]: ".format(0, n_features - 1)
        ))
        # feature_index += 1

    except :
        print "Unexpected error:", sys.exc_info()[0]
        feature_index = 0






