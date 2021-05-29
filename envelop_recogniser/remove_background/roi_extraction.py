import cv2


def get_largest_element(original_image):
    image_copy = original_image.copy()
    image_draw_copy = original_image.copy()

    gray = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
    # Blurring to reduce high frequency noise to make our contour detection process more accurate
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Not necessary to draw, only for visualizing purpose
    cv2.drawContours(image_draw_copy, contours, -1, (0, 0, 255), 3)
    contours_list = []

    for i in contours:
        area = cv2.contourArea(i)
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
    # cv2.drawContours(image_draw_copy, max_obj['contour'], -1, (0, 0, 255), 2)
    # visualize the largest area using red color contour
    # 'envelop' label is use to only visualizing purposes
    max_obj['envelop'] = image_draw_copy

    return max_obj
