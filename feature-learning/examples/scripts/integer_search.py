import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from feature_learning import utils
from utils import semantic_hash_search


feature_data_path = "/data/ads_datasets/welovead_64x64_lab_flipped_latent_features.npy"
images_path = "/data/ads_datasets/welovead_64x64_rgb.npy"


def plot(axis, axis_horz_index, image, top_image, bottom_image):
    #  axis[1][axis_horz_index].imshow(image)
    axis[0][axis_horz_index].imshow(top_image)
    axis[1][axis_horz_index].imshow(bottom_image)

print "Loading feature data.."
feature_data = np.load(feature_data_path)

print "Calculating semantic integers.."
semantic_integers = semantic_hash_search.prepare_integers(feature_data)

print "Loading images.."
images = np.load(images_path)

if len(images) != len(feature_data):
    raise ValueError(
        "Number of images and feature samples does not match: {} != {}".format(len(images), len(feature_data))
    )


def init():
    for i in range(0, n):
        plot(axs, i, images[0], images[0], images[0])
    fig.show()
    plt.draw()
    plt.show(block=False)
    time.sleep(5)

print "Find top and bottom 5 posters"
n = 10
fig, axs = plt.subplots(2, n, figsize=(28, 28))
image_index = 0
n_images = len(images)

init()

while True:
    try:

        print "Image:", image_index

        closest = semantic_hash_search.find(semantic_integers[image_index], semantic_integers, n, True)
        furthest = semantic_hash_search.find(semantic_integers[image_index], semantic_integers, n, False)

        print "Plotting.."

        for idx in range(0, n):
            plot(axs, idx, images[image_index], images[closest[idx]], images[furthest[idx]])

        plt.title('Left upper image: {}'.format(image_index))

        fig.show()
        plt.draw()
        plt.show(block=False)

        # image_index = int(raw_input(
        #     "Image [{} to {}]: ".format(0, n_images - 1)
        # ))
        image_index += 1

    except :
        print "Unexpected error:", sys.exc_info()[0]
        image_index = 0






