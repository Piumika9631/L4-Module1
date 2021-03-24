import cv2 as cv
import imutils
from Module1.remove_background.extract_largest import element_largest
from Module1.preprocessing.resize import image_resize


def correct_skewness(image):
    envelop_obj = element_largest(image)
    angle = cv.minAreaRect(envelop_obj['contour'])[-1]

    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle

    rotated = imutils.rotate_bound(image, angle)

    # draw the correction angle on the image so we can validate it
    # cv.putText(rotated, "Angle: {:.2f} degrees".format(angle),
    #            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    #print("[INFO] angle: {:.3f}".format(angle))

    rotated_resized_image = image_resize(rotated, height=700)

    return rotated_resized_image
