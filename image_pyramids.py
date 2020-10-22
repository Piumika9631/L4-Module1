#import opencv library
import cv2 as cv

#read the image from computer
img = cv.imread('images/dog.jpg')
cv.imshow('original image', img)

#up sample pyramid
up1 = cv.pyrUp(img)
cv.imshow('up1 image', up1)
up2 = cv.pyrUp(up1)
cv.imshow('up2 image', up2)

#down sample pyramid
down1 = cv.pyrDown(img)
cv.imshow('down1 image', down1)
down2 = cv.pyrDown(down1)
cv.imshow('down2 image', down2)

cv.waitKey(0)