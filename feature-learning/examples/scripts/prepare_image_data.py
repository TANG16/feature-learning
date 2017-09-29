
from vionel_feature_learning.utils.image_utils import *

images_path = "/home/erik/data/ads_datasets/welovead"
width_height = 64
output_data_set_path = "/home/erik/data/ads_datasets/welovead_64x64_lab_flipped.npy"
output_display_set_path = "/home/erik/data/ads_datasets/welovead_64x64_rgb.npy"

print "Preparing image data set.."
image_data_set = build_data_set(
    images_path,
    width_height,
    max_num_images=75000
)

print "Preparing image display set.."
image_display_set = build_rgb_image_set(
    images_path,
    width_height,
)

print "Displaying first image.."
display_image(
    resize_image(
        float32_lab_to_uint8_bgr(image_data_set[0]),
        w=200,
        h=200
    ),
    wait_ms=40
)

print "Saving image data set.."
np.save(output_data_set_path, image_data_set)

print "Saving image display set.."
np.save(output_display_set_path, image_display_set)
