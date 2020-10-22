#import opencv library
import cv2 as cv
from matplotlib import pyplot as plt

#read the grayscale image from computer
img_gray = cv.imread('images/dog.jpg', 0)
cv.imshow('grayscale image', img_gray)

plt.hist(img_gray.ravel(), 256, [0, 256])
plt.show()

cv.waitKey(0)