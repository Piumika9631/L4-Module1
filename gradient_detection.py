#import opencv library
import cv2 as cv

#read the image from computer
img = cv.imread('images/dog.jpg')
cv.imshow('original image', img)

#Laplacian gradient
gradient = cv.Laplacian(img, cv.CV_64F)
cv.imshow('Edge detected image', gradient)

#Sobel x gradient
gradient = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
cv.imshow('Sobel x gradient', gradient)

#Sobel y gradient
gradient = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
cv.imshow('Sobel y gradient', gradient)

cv.waitKey(0)