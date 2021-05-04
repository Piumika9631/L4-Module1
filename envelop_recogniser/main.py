import glob

import cv2 as cv

from envelop_recogniser import constants
from envelop_recogniser.configurations import read_image
from remove_background.roi_extraction import get_largest_element
from remove_background.image_manipulations import apply_background_mask
from remove_background.image_manipulations import resize_image
from identify_elements.skew_correction import correct_skewness

mode = constants.mode_dir


def main_function(selected_input_image):
    # # take mask colors
    # b, g, r = (selected_input_image[300, 300])
    # print (r)
    # print (g)
    # print (b)

    # view original image
    # cv.imshow('original', selected_input_image)

    # view resized image
    resized_image = resize_image(selected_input_image, height=700)
    cv.imshow('resized', resized_image)

    # extract the envelop
    envelop_marked_obj = get_largest_element(resized_image)
    # cv.imshow('envelop', envelop_marked_obj['envelop'])

    # remove background
    ROI = apply_background_mask(resized_image, envelop_marked_obj)
    # cv.imshow('masked', ROI)

    # skew correction
    rotated_image = correct_skewness(ROI)
    cv.imshow('rotated', rotated_image)




if mode == constants.mode_file:
    selectedImagePath = constants.image14Path
    selectedImage = read_image(selectedImagePath)
    main_function(selectedImage)
    cv.waitKey(0)
elif mode == constants.mode_dir:
    for img in glob.glob("../new_data/*.jpg"):
        selectedImage = read_image(img)
        main_function(selectedImage)
        cv.waitKey(1000)


