import cv2 as cv


def get_largest_element(original_image):
    image_copy = original_image.copy()
    image_draw_copy = original_image.copy()

    gray = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
    # Blurring to reduce high frequency noise to make our contour detection process more accurate
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    thresh = cv.threshold(blurred, 60, 255, cv.THRESH_BINARY)[1]
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Not necessary to draw, only for visualizing purpose
    cv.drawContours(image_draw_copy, contours, -1, (0, 255, 0), 1)
    contours_list = []

    for i in contours:
        area = cv.contourArea(i)
        my_object = {'contour': i, 'area': area}
        # print('AREA: ' + str(area))
        contours_list.append(my_object)

    max_area = 0
    max_obj = contours_list[0]
    for obj in contours_list:
        if obj['area'] > max_area:
            max_area = obj['area']
            max_obj = obj

    # print('MAX AREA: ' + str(max_area))
    # cv.drawContours(image_draw_copy, max_obj['contour'], -1, (0, 0, 255), 2)
    # visualize the largest area using red color contour
    # 'envelop' label is use to only visualizing purposes
    max_obj['envelop'] = image_draw_copy

    return max_obj
