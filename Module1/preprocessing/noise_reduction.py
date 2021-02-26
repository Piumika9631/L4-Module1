import cv2 as cv

def noise_remover(image):
    #Need to add relevant Preprocessing technique from Median or Gaussian filter or mixed
    #img - original image

    # Median filter
    median = cv.medianBlur(img, 3)

    # Gaussian filter
    img = cv.GaussianBlur(im, (5, 5), 0)

    #convert to gray image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)