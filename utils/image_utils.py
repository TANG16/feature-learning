import cv2
import numpy as np
import os
import sys

def float32_lab_to_uint8_bgr(image):
    upscaled_image = image * 256
    uint8_image = np.uint8(upscaled_image)
    return cv2.cvtColor(uint8_image, cv2.COLOR_LAB2BGR)


def float32_lab_to_uint8_rgb(image):
    upscaled_image = image * 256
    uint8_image = np.uint8(upscaled_image)
    return cv2.cvtColor(uint8_image, cv2.COLOR_LAB2RGB)


def normalize_image(image):
    return np.float32(image) / 256

def read_image_bgr(path):
    return cv2.imread(path, 3)


def bgr_to_lab(bgr_image):
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)


def bgr_to_rgb(bgr_image):
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

def display_image(image, wait_ms):
    cv2.imshow("image", image)
    cv2.waitKey(wait_ms)

def resize_image(image, w, h):
    return cv2.resize(image, (w, h))


def prepare_image(bgr_image, width_height):
    image_lab = bgr_to_lab(bgr_image)
    norm_image = normalize_image(image_lab)
    return resize_image(norm_image, width_height, width_height)

def flip_brighter_side_left(lab_image):

    rows, cols, channels = lab_image.shape

    left_side = lab_image[:, :cols / 2, 0][0]
    right_side = lab_image[:, cols / 2:, 0][0]

    is_right_side_brighter = np.mean(right_side) > np.mean(left_side)

    return np.fliplr(lab_image) if is_right_side_brighter else lab_image

def build_data_set(dir_path, width_height, max_num_images=sys.maxint):
    file_names = os.listdir(dir_path)
    file_names.sort()
    images = []
    c = 0
    for file_name in file_names:

        if c == max_num_images:
            break

        if c % 1000 == 0:
            print c, "images"
        c += 1

        file_path = dir_path + "/" + file_name

        prepared_image = prepare_image(read_image_bgr(file_path), width_height)

        images.append(flip_brighter_side_left(prepared_image))

    return images

def build_rgb_image_set(dir_path, width_height):
    file_names = os.listdir(dir_path)
    file_names.sort()
    images = []
    for file_name in file_names:
        file_path = dir_path + "/" + file_name
        images.append(resize_image(bgr_to_rgb(read_image_bgr(file_path)), width_height, width_height))

    return images

