import cv2 as cv


def get_largest_element(original_image):
    image_copy = original_image.copy()
    gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
    cv.drawContours(image_copy, contours, -1, (0, 255, 0), 1)
    contours_list = []

    for i in contours:
        M = cv.contourArea(i)
        my_object = {'contour': i, 'area': M}
        # print('AREA: ' + str(M))
        contours_list.append(my_object)

    max = 0
    max_obj = contours_list[0]
    for obj in contours_list:
        if obj['area'] > max:
            max = obj['area']
            max_obj = obj

    # print('MAX AREA: ' + str(max))
    cv.drawContours(image_copy, max_obj['contour'], -1, (0, 0, 255), 2)
    max_obj['envelop'] = image_copy

    return max_obj
