import sys
import time

import matplotlib.pyplot as plt
from vionel_feature_learning.utils import semantic_hash_search
from vionel_feature_learning.utils.image_utils import read_image_bgr, resize_image, bgr_to_rgb


class IntegerSearcher(object):

    def __init__(self, image_paths, image_feature_vectors):

        if len(image_paths) != len(image_feature_vectors):
            raise ValueError(
                "Number of images and feature samples does not match: {} != {}".format(
                    len(image_paths),
                    len(image_feature_vectors)
                )
            )

        image_width_height = 200
        self.images = self._prepare_images(image_paths, image_width_height)
        self.image_feature_vectors = image_feature_vectors

    def visualize_similar_images(self):

        images = self.images

        def plot(axis, axis_horz_index, top_image, bottom_image):
            axis[0][axis_horz_index].imshow(top_image)
            axis[1][axis_horz_index].imshow(bottom_image)

        def init():
            for i in range(0, n):
                plot(axs, i, images[0], images[0])
            fig.show()
            plt.draw()
            plt.show(block=False)
            time.sleep(1)

        print "Calculating semantic integers.."
        semantic_integers = semantic_hash_search.prepare_integers(self.image_feature_vectors)

        n = 10
        fig, axs = plt.subplots(2, n, figsize=(28, 28))
        image_index = 0

        init()

        while True:
            try:

                print "Image:", image_index

                closest = semantic_hash_search.find(semantic_integers[image_index], semantic_integers, n, True)
                furthest = semantic_hash_search.find(semantic_integers[image_index], semantic_integers, n, False)

                print "Plotting.."

                for idx in range(0, n):
                    plot(axs, idx, images[closest[idx]], images[furthest[idx]])

                plt.title('Left upper image: {}'.format(image_index))

                fig.show()
                plt.draw()
                plt.show(block=False)

                cancel = raw_input("'c' to cancel, else next image: ").lower().strip() == "c"

                if cancel:
                    break

                image_index += 1

            except:
                print "Unexpected error:", sys.exc_info()[0]
                image_index = 0

    @staticmethod
    def _prepare_images(image_paths, width_height):

        images = []
        for image_path in image_paths:
            bgr_image = read_image_bgr(image_path)
            resized_bgr_image = resize_image(bgr_image, width_height, width_height)
            rgb_image = bgr_to_rgb(resized_bgr_image)
            images.append(rgb_image)

        return images




