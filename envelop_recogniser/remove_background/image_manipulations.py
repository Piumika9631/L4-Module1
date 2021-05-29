import cv2
import numpy as np

from envelop_recogniser.remove_background.roi_extraction import get_largest_element


def apply_background_mask(image, largest_element):
    image_copy = image.copy()
    # a mask is the same size as our image, but has only two pixel values,
    # 0 and 255 -- pixels with a value of 0 (background) are ignored in the original image
    # while mask pixels with a value of 255 (foreground) are allowed to be kept

    mask = np.zeros(image.shape[:2], np.uint8)
    largest_contour = largest_element['contour']
    cv2.drawContours(mask, [largest_contour], -1, 255, -1)
    masked_image = cv2.bitwise_and(image_copy, image_copy, mask=mask)
    return masked_image


def background_crop(image):
    largest_element = get_largest_element(image)
    envelop_contour = largest_element['contour']
    x, y, w, h = cv2.boundingRect(envelop_contour)
    crop_img = image[y:y + h, x:x + w]
    return crop_img


def resize_image(selected_input_image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    dimension = None
    (h, w) = selected_input_image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return selected_input_image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dimension = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dimension = (width, int(h * r))

    # resize the image
    resized = cv2.resize(selected_input_image, dimension, interpolation=inter)

    # return the resized image
    return resized
