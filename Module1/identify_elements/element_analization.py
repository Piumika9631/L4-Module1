import cv2 as cv
import numpy as np


def get_elements(image):
    white_board = np.full(image.shape[:2], 255, np.uint8)
    all_contours = get_rectangles(image);

    for i in all_contours:
        contour = i
        area = cv.contourArea(contour)
        x, y, w, h = cv.boundingRect(contour)
        area = w * h
        # assuming 5 is the smallest ratio of a character height and width
        if area < 5000 and (w / h < 5 and h / w < 5):
            cv.rectangle(white_board, (x, y), (x + w, y + h), (0, 0, 255), -1)
    cv.imshow('White board', white_board)

    edges = cv.Canny(white_board, 110, 210)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))
    dilate = cv.dilate(edges, kernel, iterations=4)
    cv.imshow('Dilated', dilate)

    contours_list = get_objects(dilate)
    labelled_list = get_label(contours_list)
    slicing(image, labelled_list)


def get_rectangles(image):
    # Remove noise
    image_copy = image.copy()
    gray = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
    thresh_image = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 13, 14)
    edges = cv.Canny(thresh_image, 3, 3)
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def get_objects(image):
    edges = cv.Canny(image, 110, 210)
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours_list = []

    for i in contours:
        M = cv.contourArea(i)
        x, y, w, h = cv.boundingRect(i)
        my_object = {'contour': i, 'area': M, 'dimensions': {'x': x, 'y': y, 'w': w, 'h': h}, 'label': 'none'}
        contours_list.append(my_object)

    return contours_list


def get_label(contours_list):
    for obj in contours_list:
        my_area = obj['area']
        if my_area > 25000:
            obj['label'] = 'TempAdd'
        else:
            obj['label'] = 'Other'

    return contours_list


def slicing(image, contours_list):
    i = 0
    for obj in contours_list:
        s = 'address_'
        if obj['label'] == 'TempAdd':
            nws = s+str(i)
            address_contour = obj['contour']
            x, y, w, h = cv.boundingRect(address_contour)
            crop_img = image[y:y + h, x:x + w]
            cv.imshow(nws, crop_img)
            i = i +1
