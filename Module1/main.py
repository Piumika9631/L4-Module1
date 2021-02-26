import cv2 as cv
from Module1.preprocessing.resize import image_resize
from Module1.remove_background.extract_largest import element_largest
from Module1.remove_background.remove_background import background_mask
from Module1.identify_elements.skew_correction import identify_elements
from Module1.identify_elements.element_analization import main_method

# noise_reduction
# resizing
image = cv.imread('../images/ad_d2.jpg', cv.IMREAD_UNCHANGED)
resized_image = image_resize(image, height=700)
# cv.imshow('resized', resized_image)

# extract the largest element
largest_element = element_largest(resized_image)
# cv.imshow('largest', largest_element['envelop'])

# remove background
ROI = background_mask(resized_image, largest_element)
# cv.imshow('masked', ROI)

# skew correction
rotated_image = identify_elements(ROI)
# cv.imshow('rotated', rotated_image)

# analise elements
main_method(rotated_image)

cv.waitKey(0)
