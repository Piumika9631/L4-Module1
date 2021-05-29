import glob
import os

import cv2

from envelop_recogniser import constants
from envelop_recogniser.configurations import read_image
from envelop_recogniser.identify_elements.element_analyzation import get_elements
from envelop_recogniser.identify_elements.swt_calculation import swt_main
from envelop_recogniser.utils import save_output
from remove_background.roi_extraction import get_largest_element
from remove_background.image_manipulations import apply_background_mask, background_crop
from remove_background.image_manipulations import resize_image
from identify_elements.skew_correction import correct_skewness

mode = constants.mode_file


def main_function(selected_input_image):
    # # take mask colors
    # b, g, r = (selected_input_image[300, 300])
    # print (r)
    # print (g)
    # print (b)

    # view original image
    # cv2.imshow('original', selected_input_image)

    # view resized image
    resized_image = resize_image(selected_input_image, constants.resize_height)
    # cv2.imshow('resized', resized_image)

    # extract the envelop
    envelop_marked_obj = get_largest_element(resized_image)
    # cv2.imshow('marked area envelop', envelop_marked_obj['envelop'])

    # remove background using mask
    ROI = apply_background_mask(resized_image, envelop_marked_obj)
    # cv2.imshow('masked', ROI)

    # correct skewness of image
    rotated_image = correct_skewness(ROI)
    # cv2.imshow('rotated', rotated_image)

    # crop unnecessary background
    cropped = background_crop(rotated_image)
    # cv2.imshow('cropped', cropped)

    # analise elements
    # get_elements method not worked due to threshold defining not reliable enough
    get_elements(cropped)
    white_board, cropped_image, vis = swt_main(cropped)

    cv2.imshow('white_board', white_board)
    cv2.imshow('cropped_image', cropped_image)
    cv2.imshow('vis', vis)

    # save_output(tail, white_board, "1")
    # save_output(tail, cropped_image, "2")
    # save_output(tail, vis, "complete")


if mode == constants.mode_file:
    selectedImagePath = constants.image17Path
    selectedImage = read_image(selectedImagePath)
    head, tail = os.path.split(selectedImagePath)
    main_function(selectedImage)
    cv2.waitKey(0)
elif mode == constants.mode_dir:
    for img in glob.glob("../new_data/*.jpg"):
        selectedImage = read_image(img)
        head, tail = os.path.split(img)
        main_function(selectedImage)
        cv2.waitKey(100)
        cv2.destroyAllWindows()
