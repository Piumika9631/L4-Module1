#import opencv library
import cv2 as cv

#read the image from computer
img = cv.imread('images/dog.jpg')
#show the image
cv.imshow('Dog original photo', img)


#show the gray color space image
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Dog gray photo', gray)
#show the HSV color space image
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
cv.imshow('Dog hsv photo', hsv)
#show the LAB color space image
lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
cv.imshow('Dog lab photo', lab)

cv.waitKey(0)