import cv2 as cv


def noise_remover(image):
    # Need to add relevant Preprocessing technique from Median or Gaussian filter or mixed
    # img - original image

    # Median filter
    median = cv.medianBlur(image, 3)

    # Gaussian filter
    img = cv.GaussianBlur(image, (5, 5), 0)

    # convert to gray image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # sample threshold values for noise reduction
    image_copy = image.copy()
    gray = cv.cvtColor(image_copy, cv.COLOR_BGR2GRAY)
    thresh_image = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 6)
    return thresh_image

    # noise reduction can do using dilation after erosion (opening / closing)
