
import numpy as np
import matplotlib.pyplot as plt
import cv2
import heapq
import sys


def visualize(images, original_latent_activations, original_reconstructions, output, mean_img):

    def float32_lab_to_uint8_rgb(image):
        upscaled_image = image * 256
        uint8_image = np.uint8(upscaled_image)
        return cv2.cvtColor(uint8_image, cv2.COLOR_LAB2RGB)

    def top_bottom_n(feature_activations, n):

        _, h, w = feature_activations.shape

        activation_vectors = np.reshape(feature_activations, (-1, h * w))

        activation_means = np.mean(activation_vectors, axis=1)

        activation_largest = heapq.nlargest(
            n,
            range(len(activation_means)),
            activation_means.take
        )

        activation_smallest = heapq.nsmallest(
            n,
            range(len(activation_means)),
            activation_means.take
        )
        activation_smallest.reverse()

        return activation_largest, activation_smallest

    def plot(axis, axis_horz_index, original_image, activation, feature_reconstruction, reconstruction):
        axis[0][axis_horz_index].imshow(float32_lab_to_uint8_rgb(original_image))
        axis[1][axis_horz_index].imshow(original_image)
        axis[2][axis_horz_index].imshow(activation, cmap='Greys_r', vmin=-1, vmax=1)
        axis[3][axis_horz_index].imshow(float32_lab_to_uint8_rgb(feature_reconstruction))
        axis[4][axis_horz_index].imshow(reconstruction + mean_img)
        axis[5][axis_horz_index].imshow(float32_lab_to_uint8_rgb(reconstruction + mean_img))

    print "\nTop and bottom 5 posters"
    n_features = original_latent_activations.shape[3]
    n = 5
    fig, axs = plt.subplots(6, 10, figsize=(10, 10))
    feature_index = 0
    while True:
        try:

            print "\nFeature:", feature_index

            latent_layer_mask = np.zeros((1, n_features))
            latent_layer_mask[0, feature_index] = 1
            reconstructions, latent_activations = output(latent_layer_mask)
            activations = latent_activations[:, :, :, feature_index]

            activation_largest, activation_smallest = top_bottom_n(activations, n)

            for idx, image_index in enumerate(activation_largest + activation_smallest):
                plot(axs, idx, images[image_index], activations[image_index, :, :], reconstructions[image_index],
                     original_reconstructions[image_index])

            fig.show()
            plt.draw()
            plt.show(block=False)

            user_input = raw_input("\nFeature <{} to {}> to visualize, <c> to continue: ".format(0, n_features - 1))

            if user_input.lower() == "c":
                break
            else:
                feature_index = int(user_input)

        except:
            feature_index = 0
            print "Unexpected error:", sys.exc_info()[0]

    plt.close()
