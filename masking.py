#import opencv library
import cv2 as cv

#read the images from computer
mask = cv.imread('images/mask.jpg')
img = cv.imread('images/dog.jpg')

#perform bitwise operations for masking
masked_image = cv.bitwise_and(img, mask)
cv.imshow('Masked', masked_image)

cv.waitKey(0)