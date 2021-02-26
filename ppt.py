#import opencv2/text/erfilter.hpp
import cv2
import numpy as np

pathname = 'E:/L4 Sem 1/Project/OpenCV/OpenCVExCode/images'
mser = cv2.MSER_create()


im = cv2.imread('images/dog.jpg')
image = cv2.resize(im, (500,500))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
vis = image.copy()
regions, _ = mser.detectRegions(gray)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(vis, hulls, 1, (0, 255, 0))




cv2.imshow('img', vis)

cv2.waitKey(0)

mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.uint8)

for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

text_only = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("text only", text_only)
cv2.waitKey(0)
