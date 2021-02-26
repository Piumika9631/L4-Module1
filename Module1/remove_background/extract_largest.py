import cv2 as cv


def element_largest(image):
    image_copy = image.copy()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    edges = cv.Canny(gray, 110, 210)
    contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(image_copy, contours, -1, (0, 255, 0), 3)
    contours_list = []

    for i in contours:
        M = cv.contourArea(i)
        my_object = {'contour': i, 'area': M}
        contours_list.append(my_object)

    max = contours_list[0]['area']
    max_obj = contours_list[0]
    for obj in contours_list:
        if obj['area'] > max:
            max = obj['area']
            max_obj = obj

    cv.drawContours(image_copy, max_obj['contour'], -1, (0, 0, 255), 3)
    max_obj['envelop'] = image_copy

    return max_obj
