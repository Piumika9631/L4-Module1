import cv2
import numpy as np
from envelop_recogniser.remove_background.roi_extraction import get_largest_element


def correct_skewness(image):
    envelop_obj = get_largest_element(image)
    angle = cv2.minAreaRect(envelop_obj['contour'])[-1]
    angle_print = angle

    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make it positive
    else:
        angle = -angle

    # In-build method removed due to blur effect
    # rotated = imutils.rotate_bound(image, angle)
    rotated = rotate_bound(image, angle)

    # draw the correction angle on the image so we can validate it
    # cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle_print),
    #            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # show the output image
    # print("[INFO] angle: {:.3f}".format(angle_print))

    return rotated


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w / 2, h / 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    # Added flag to reduce the blur effect
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC)
