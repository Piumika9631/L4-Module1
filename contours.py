#import opencv library
import cv2 as cv

#read the image from computer
img = cv.imread('images/coins.jpg')
img_original = img.copy()
cv.imshow('original image', img)

#apply preprocessing conver to gray, blur, threshold
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (15,15), 0)
img = cv.Canny(img, 30, 170)
thresh, thresh_img = cv.threshold(img, 127, 255, 0)

#pass pre-processed image, Contour retrieval mode and Contour approximation
#will return back Detected contours and hierarchy (containing information about the image topology)
contours, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

#pass image, contours, contourIdx (-1 means all), color, thickness
#Contours will be drawn on the image passed
cv.drawContours(img_original, contours, -1, (0,255,0), 3)

cv.imshow('Preprocessed image', thresh_img)
cv.imshow('Contours drawn on original', img_original)

print('No of objects detected is {}'.format(len(contours)))

cv.waitKey(0)