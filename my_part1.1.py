#import opencv library
import cv2 as cv
import operator

#read the image from computer
img = cv.imread('images/ad_d2.jpg')
resized_img = cv.resize(img,(500,500))
img_original = resized_img.copy()
gray = cv.cvtColor(resized_img,cv.COLOR_BGR2GRAY)

edges = cv.Canny(gray,100,200)

# kernel = cv.getStructuringElement(cv.MORPH_RECT, (10,10))
# dilate = cv.dilate(edges, kernel, iterations=2)
# cv.imshow('dilate image', dilate)

# ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(img_original, contours, -1, (0,255,0), 3)

contours_list = []
for i in contours:
    M = cv.contourArea(i)
    my_object={'contour':i,'area':M}
    contours_list.append(my_object)

max = contours_list[0]['area']
max_obj=contours_list[0]
for obj in contours_list:
    if obj['area']>max:
        max=obj['area']
        max_obj=obj

print("max area is:", max)
print("new element is:", max_obj)
print("new obj is:", max_obj['contour'])
cv.drawContours(img_original, max_obj['contour'], -1, (255,255,0), 3)

cv.imshow('resized image', resized_img)
cv.imshow('output image', edges)
cv.imshow('contour image', img_original)

cv.waitKey(0)
