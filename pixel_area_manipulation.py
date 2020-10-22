#import opencv library
import cv2 as cv

#read the image from computer (as grayscale 0, as color 1, as original -1 or nothing)
img = cv.imread('images/shapes.jpg')
#print the shape of image
print(img.shape)
#show the image
cv.imshow('Shapes photo', img)


#Pixel
#get pixel at raw,colomn position
pixel = img[120, 120]
print('Color intensity at (120,120) pixel is Blue: {}, Green: {}, Red: {}'.format(pixel[0], pixel[1], pixel[2]))

#get only single color value for a pixel (as Blue 0, as Green 1, as Red 2)
red_pixel = img[120, 120, 2]
print('Color intensity of red pixel at (120,120) pixel is {}'.format(red_pixel))

#set the color of a pixel to specific RGB as (BGR) color
img[120, 120] = (0, 255, 0)
pixel = img[120, 120]
print('Color intensity at (120,120) pixel is Blue: {}, Green: {}, Red: {}'.format(pixel[0], pixel[1], pixel[2]))

#write the edited image to computer
cv.imwrite('images/shapes_pixel_edited.jpg', img)

#Area
#get the region from the image
slice = img[0:120, 0:120]
cv.imshow('sliced image', slice)

#set the color of an area to specific RGB as (BGR) color
img[0:120, 0:120] = (0, 255, 0)
cv.imshow('manipulated slice of image', img)

#printing properties of image
print('Shape of image: ', img.shape)
print('Total number of pixels in the image: ', img.size)
print('Data type of image: ', img.dtype)

cv.waitKey(0)
