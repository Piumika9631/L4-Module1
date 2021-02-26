#import opencv library
import cv2 as cv
import numpy as np

#read the image from computer and resize
im = cv.imread('images/ad_d3.jpg')
img = cv.medianBlur(im, 3)
#img = cv.GaussianBlur(im, (5,5), 0)
resized_img = cv.resize(img,(500,500))
img_original = resized_img.copy()

#finding all contours
gray = cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)
edges = cv.Canny(gray,100,200)
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img_original, contours, -1, (0,255,0), 3)

#create object with contour and it's area
contours_list = []
for i in contours:
    M = cv.contourArea(i)
    my_object={'contour':i,'area':M}
    contours_list.append(my_object)

#finding maximum area contour
max = contours_list[0]['area']
max_obj=contours_list[0]
for obj in contours_list:
    if obj['area']>max:
        max=obj['area']
        max_obj=obj

cv.drawContours(img_original, max_obj['contour'], -1, (0,0,255), 3)
c=max_obj['contour']

#(
# # determine the most extreme points along the contour
# extLeft = tuple(c[c[:, :, 0].argmin()][0])
# extRight = tuple(c[c[:, :, 0].argmax()][0])
# extTop = tuple(c[c[:, :, 1].argmin()][0])
# extBot = tuple(c[c[:, :, 1].argmax()][0])
# # draw the outline of the object, then draw each of the
# # extreme points, where the left-most is red, right-most
# # is green, top-most is blue, and bottom-most is teal
# cv.drawContours(img_original, [c], -1, (0, 255, 255), 2)
# cv.circle(img_original, extLeft, 8, (0, 0, 255), -1)
# cv.circle(img_original, extRight, 8, (0, 255, 0), -1)
# cv.circle(img_original, extTop, 8, (255, 0, 0), -1)
# cv.circle(img_original, extBot, 8, (255, 255, 0), -1)
#)

# Masking the part other than the number plate
mask = np.zeros(gray.shape,np.uint8)
new_image = cv.drawContours(mask,[c],0,255,-1,)
new_image = cv.bitwise_and(resized_img,resized_img,mask=mask)
cv.imshow('new after mask', new_image)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5,5))
dilate = cv.dilate(new_image, kernel, iterations=2)
cv.imshow('Dilated', dilate)

x,y,w,h = cv.boundingRect(c)
ROI = img_original[y:y+h,x:x+w]
cv.imshow("Largest Contour",ROI)
image = cv.resize(ROI, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
gray = cv.bitwise_not(gray)
thresh = cv.threshold(gray, 0, 255,	cv.THRESH_BINARY | cv.THRESH_BINARY_INV | cv.THRESH_OTSU)[1]
cv.imshow("Thresh", thresh)
coords = np.column_stack(np.where(thresh > 0))
angle = cv.minAreaRect(coords)[-1]
if angle < -45:
	angle = -(90 + angle)
else:
	angle = -angle
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv.getRotationMatrix2D(center, angle, 1.0)
rotated = cv.warpAffine(image, M, (w, h),
	flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
cv.putText(rotated, "Angle: {:.2f} degrees".format(angle),
	(10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
print("[INFO] angle: {:.3f}".format(angle))
cv.imshow("Input", image)
cv.imshow("Thresh", thresh)
cv.imshow("Rotated", rotated)

# kernel = cv.getStructuringElement(cv.MORPH_RECT, (50,10))
# dilate = cv.dilate(ROI, kernel, iterations=2)
# cv.imshow('dilated image', dilate)

cv.imshow('resized image', resized_img)
cv.imshow('output image', edges)
cv.imshow('contour image', img_original)
cv.waitKey(0)
