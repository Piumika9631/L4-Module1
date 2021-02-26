import cv2 as cv
import numpy as np
from Module1.remove_background.extract_largest import element_largest

def main_method(image):
    get_elements(image)

def get_elements(image):
    envelop_obj = element_largest(image)
    envelop_contour = envelop_obj['contour']
    mask = np.full(image.shape[:2], 255, np.uint8)
    cv.drawContours(mask, [envelop_contour], -1, 255, -1)
    masked_image = cv.bitwise_and(image, image, mask=mask)

    # gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    # thresh_image = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    cv.imshow('Mask', mask)
    cv.imshow('Masked image', masked_image)


    x, y, w, h = cv.boundingRect(envelop_contour)

    # mask = np.zeros_like(envelop_image)  # Create mask where white is what we want, black otherwise
    # cv.drawContours(mask, envelop_contour, 0, 255, -1)  # Draw filled contour in mask
    # out = np.zeros_like(envelop_image)  # Extract out the object and place into output image
    # out[mask == 255] = envelop_image[mask == 255]
    # (y, x) = np.where(mask == 255)
    # (topy, topx) = (np.min(y), np.min(x))
    # (bottomy, bottomx) = (np.max(y), np.max(x))
    # out = out[topy:bottomy + 1, topx:bottomx + 1]

    # Show the output image
    # cv.imshow('Output', out)

    # crop_img = thresh[y:y + h, x:x + w]
    # cv.imshow("cropped", crop_img)

    # gray = cv.cvtColor(crop_img, cv.COLOR_BGR2GRAY)
    # edges = cv.Canny(crop_img, 110, 210)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    # dilate = cv.dilate(edges, kernel, iterations=2)

    # cv.imshow('Dilated', dilate)
    cv.waitKey(0)


