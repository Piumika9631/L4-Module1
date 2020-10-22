#import opencv library
import cv2 as cv
#import numpy library
import numpy as np

#read the image from computer
img = cv.imread('images/pixelated.jpg')
img_noicy = cv.imread('images/rough_edge.jpg')
cv.imshow('original pixelated', img)
cv.imshow('original rough edge', img_noicy)

#create kernel for convolution
custom_kernel = np.ones((5,5), np.float32)/25

#custom smoothing
result = cv.filter2D(img, -1, custom_kernel)
cv.imshow('Smoothed image', result)

#perform average filtering
result_a = cv.blur(img, (5,5))
cv.imshow('Average smoothing', result_a)

#perform gaussian blur (contrast also retained)
result_g = cv.GaussianBlur(img, (5,5), 0)
cv.imshow('Gaussian smoothing', result_g)

#perform median blur filtering
result_m = cv.medianBlur(img_noicy, 5)
cv.imshow('Median blur', result_m)

#perform bilateral filtering
result_b = cv.bilateralFilter(img, 10, 75, 75)
cv.imshow('bilateral filtering', result_b)

cv.waitKey(0)