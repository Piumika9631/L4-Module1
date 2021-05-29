import cv2
import numpy as np

show_whiteboard = True
show_dilated = True
show_edges = False


def get_elements(image):
    white_board = np.full(image.shape[:2], 255, np.uint8)
    all_contours = get_rectangles(image)
    count = 0

    for i in all_contours:
        contour = i
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        # assuming 5 is the smallest ratio of a character height and width
        if area < 5000 and (w / h < 5 and h / w < 5):
            cv2.rectangle(white_board, (x, y), (x + w, y + h), (0, 0, 255), -1)
            count += 1
    if show_whiteboard:
        cv2.imshow('White board', white_board)

    edges = cv2.Canny(white_board, 110, 210)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    dilate = cv2.dilate(edges, kernel, iterations=4)
    if show_dilated:
        cv2.imshow('Dilated', dilate)

    contours_list = get_objects(dilate, image)
    labelled_list = get_label(contours_list)
    slicing(image, labelled_list)


def get_rectangles(image):
    # Remove noise
    image_copy = image.copy()
    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    thresh_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 14)
    edges = cv2.Canny(thresh_image, 3, 3)
    if show_edges:
        cv2.imshow('edges', edges)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def get_objects(image, original):
    image_copy = original.copy()
    edges = cv2.Canny(image, 110, 210)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 1)
    # cv2.imshow('copy', image_copy)
    contours_list = []

    # if len(contours) == 0:
    #     rotated = imutils.rotate_bound(original, 90)
    #     get_elements(rotated)

    for i in contours:
        x, y, w, h = cv2.boundingRect(i)
        M = w * h
        # print('area ' + str(M))
        my_object = {'contour': i, 'area': M, 'dimensions': {'x': x, 'y': y, 'w': w, 'h': h}, 'label': 'none'}
        contours_list.append(my_object)

    return contours_list


def get_label(contours_list):
    for obj in contours_list:
        my_area = obj['area']

        # height & width
        if my_area > 5000:
            obj['label'] = 'TempAddress'
        else:
            obj['label'] = 'Other'

    return contours_list


def slicing(image, contours_list):
    i = 0
    j = 0
    for obj in contours_list:
        s = 'address_'
        nws = s + str(i)

        if obj['label'] == 'TempAddress':
            # print(obj['area'])
            address_contour = obj['contour']
            x, y, w, h = cv2.boundingRect(address_contour)
            crop_img = image[y:y + h, x:x + w]
            # cv2.imshow(nws, crop_img)
            i = i + 1

        j = j + 1
