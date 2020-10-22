#import opencv library
import cv2 as cv

#read the image from computer (as grayscale 0, as color 1, as original -1 or nothing)
img = cv.imread('images/shapes.jpg',0)

#print the shape of image
print(img.shape)

#write the image to computer
cv.imwrite('images/shapesgray.jpg', img)

#show the image
cv.imshow('Shapes photo', img)
cv.waitKey(0)