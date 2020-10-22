#import opencv library
import cv2 as cv
#import numpy library
import numpy as np

#read the image from computer
img = cv.imread('images/dog.jpg')
cv.imshow('original dog image', img)

#resize - scale up image
resized_up = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
cv.imshow('Scaled up', resized_up)

#resize - scale down image
resized_down = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
cv.imshow('Scaled down', resized_down)

#translation
#define the transformation matrix (row, colomn, (-)to left)
trans_matrix = np.float32([[1,0,150], [0,1,100]])
#shifting the image
shifted_image = cv.warpAffine(img, trans_matrix, (img.shape[1], img.shape[0]))
cv.imshow('Shifted image', shifted_image)

#rotation
#get cetral point (width/2 height/2)
central_point = img.shape[1] // 2 , img.shape[0] // 2
#define transformation matrix (central point, angel, scale (scaling value 1 is use to same size))
trans_matrix_rot = cv.getRotationMatrix2D(central_point, 75, 1)
#rotated image
rotated_image = cv.warpAffine(img, trans_matrix_rot, (img.shape[1], img.shape[0]))
cv.imshow('Rotated image', rotated_image)

#Flip (1-horizontal, 0-vertical, -1-both)
f_h_image = cv.flip(img, 1)
cv.imshow('Flipped Horizontal', f_h_image)

f_v_image = cv.flip(img, 0)
cv.imshow('Flipped Vertical', f_v_image)

f_image = cv.flip(img, -1)
cv.imshow('Flipped Both', f_image)

#Crop
cropped_img = img[100:400 , 200:300]
cv.imshow('Cropped image', cropped_img)

cv.waitKey(0)