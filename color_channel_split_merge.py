#import opencv library
import cv2 as cv
#import numpy library
import numpy as np

#read the image from computer
img = cv.imread('images/shapes.jpg')
#show the image
cv.imshow('Shapes original photo', img)


#split the color channels
blue, green, red = cv.split(img)

#show the blue pixel intensity
cv.imshow('Blue pixel intensity', blue)
#show the green pixel intensity
cv.imshow('Green pixel intensity', green)
#show the red pixel intensity
cv.imshow('Red pixel intensity', red)

#merging the channels
img_merged = cv.merge((blue, green, red))

#show the merged image
cv.imshow('Merged shapes photo', img_merged)

#creating a zero array same size as that of our image
zeros_array = np.zeros(img_merged.shape[:2], dtype='uint8')

#get and show the red only channel image
red_only_image = cv.merge((zeros_array, zeros_array, red))
cv.imshow('Red only shapes image', red_only_image)
#get and show the green only channel image
green_only_image = cv.merge((zeros_array, green, zeros_array))
cv.imshow('Green only shapes image', green_only_image)
#get and show the Blue only channel image
blue_only_image = cv.merge((blue, zeros_array, zeros_array))
cv.imshow('Blue only shapes image', blue_only_image)

cv.waitKey(0)