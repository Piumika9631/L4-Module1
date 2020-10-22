#import opencv library
import cv2 as cv

#read the image from computer
img = cv.imread('images/dog.jpg')
cv.imshow('original image', img)

#canny edge detection code
edges = cv.Canny(img, 110, 210)
cv.imshow('edge detected image', edges)

cv.waitKey(0)