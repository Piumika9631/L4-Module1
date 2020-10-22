#import opencv library
import cv2 as cv

#read the images from computer
img1 = cv.imread('images/dog.jpg')
img2 = cv.imread('images/shapes.jpg')

#do simple array slicing
crooped_image1 = img1[60:400, 60:400]
crooped_image2 = img2[60:400, 60:400]

cv.imshow('original image1', crooped_image1)
cv.imshow('original image2', crooped_image2)

#adding the images
added_image = cv.add(crooped_image1, crooped_image2)
cv.imshow('Added image', added_image)

#substracting the images
substracted_image = cv.subtract(crooped_image1, crooped_image2)
cv.imshow('Substracted image', substracted_image)

cv.waitKey(0)