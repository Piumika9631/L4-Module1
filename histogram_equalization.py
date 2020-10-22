#import opencv library
import cv2 as cv
from matplotlib import pyplot as plt

#read the grayscale image from computer
img_gray = cv.imread('images/brain.jpg', 0)
cv.imshow('grayscale image', img_gray)

#perform equalization
corrected_img = cv.equalizeHist(img_gray)
cv.imshow('Equalized image', corrected_img)

cv.waitKey(0)