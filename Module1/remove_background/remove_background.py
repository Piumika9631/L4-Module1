import cv2 as cv
import numpy as np


def background_mask(image, largest_element):
    image_copy = image.copy()
    # a mask is the same size as our image, but has only two pixel values, 0 and 255 -- pixels with a value of 0 (background) are ignored in the original image while mask pixels with a value of 255 (foreground) are allowed to be kept
    mask = np.zeros(image.shape[:2], np.uint8)
    c = largest_element['contour']
    cv.drawContours(mask, [c], -1, 255, -1)
    masked_image = cv.bitwise_and(image_copy, image_copy, mask=mask)
    return masked_image


def background_crop(image, largest_element):
    image_copy = image.copy()
    envelop_contour = largest_element['contour']
    x, y, w, h = cv.boundingRect(envelop_contour)
    crop_img = image_copy[y:y + h, x:x + w]
    cv.imshow("cropped", crop_img)
    cv.waitKey(0)
