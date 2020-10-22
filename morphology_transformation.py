#import opencv library
import cv2 as cv
#import numpy library
import numpy as np

#read the image from computer
img = cv.imread('images/dog.jpg')
cv.imshow('original dog image', img)

#create an ones array as kernel, size 5x5
trans_kernal = np.ones((5, 5), dtype='uint8')

#defining the erosion transformation
trans_img = cv.erode(img, trans_kernal, iterations=1)

#show the erosion transformation
cv.imshow('eroded image', trans_img)

#defining the dilation transformation
trans_dil_img = cv.dilate(img, trans_kernal, iterations=1)

#show the dilation transformation
cv.imshow('dilate image', trans_dil_img)

#define opening transformation
trans_open_img = cv.morphologyEx(img, cv.MORPH_OPEN, trans_kernal)

#show the open transformation - remove white noice
cv.imshow('open transformation', trans_open_img)

#define closing transformation
trans_close_img = cv.morphologyEx(img, cv.MORPH_CLOSE, trans_kernal)

#show the close transformation - remove black noice
cv.imshow('close transformation', trans_close_img)

#define dradient transformation
trans_t_img = cv.morphologyEx(img, cv.MORPH_GRADIENT, trans_kernal)

#show the gradient transformation
cv.imshow('gradient transformation', trans_t_img)

#define top hat transformation
trans_th_img = cv.morphologyEx(img, cv.MORPH_TOPHAT, trans_kernal)

#show the top hat transformation
cv.imshow('top hat transformation', trans_th_img)

#define black hat transformation
trans_bh_img = cv.morphologyEx(img, cv.MORPH_BLACKHAT, trans_kernal)

#show the black hat transformation
cv.imshow('black hat transformation', trans_bh_img)

cv.waitKey(0)

'''Morphology - study of form and structure
   Transformation - any changes in form and structure
   Erosion - erodes away the boundaries of the object in focus,
    all the pixels near boundary will be discarded depending upon the size of kernel,
    useful for remove small white noises from images
  Dilation - opposite of erosion,
    increases the white region in the  image,
    typically in cases like black noise removal, erosion is followed by dilation,
    erosion removes white noises but it also shrinks object so we dilate it,
  Opening - erosion followed by dilation,
    used for white noise removal
  Closing - dilation followed by erosion,
    used for small black noise removal
  Gradient - difference between dilation and erosion,
    result will give an outline of the image
  Top Hat - difference between input image and Opening of the image
    result will get rid of the overlapping parts of image
  Black Hat - difference between input image and Closing of the image,
    result will get rid of all parts other than the overlapping parts of image'''