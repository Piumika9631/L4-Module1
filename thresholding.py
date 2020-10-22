#import opencv library
import cv2 as cv

#read the image from computer
img = cv.imread('images/brain.jpg')
cv.imshow('original image', img)

#read the grayscale image from computer
img_gray = cv.imread('images/brain.jpg', 0)
cv.imshow('grayscale image', img_gray)

#perform simple threshold (above 130 - white, below 130 - black)
thresh, thresh_image = cv.threshold(img, 130, 255, cv.THRESH_BINARY)
cv.imshow('thresh binary image', thresh_image)

#Otsu's binarization (Otsuu thresholding) - 0 value get the optimal threshold value
thresh, thresh_image = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow('Otsu thresholding', thresh_image)

#adaptive thresholding (Adaptive Mean C) - image divide into blocks based on lighting condition
#gray scale image, maximum value, adaptive method, threshold method, block size, constant
thresh_image = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
cv.imshow('Adaptive mean C threshold image', thresh_image)

#adaptive thresholding (Adaptive Gaussian C)
thresh_image = cv.adaptiveThreshold(img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
cv.imshow('Adaptive Gaussian C threshold image', thresh_image)

cv.waitKey(0)

