#import opencv library
import cv2 as cv
#import numpy library
import numpy as np

#create a black image (canvas) of black color
#using numpy zeros method for a zeros array
zeros_array = np.zeros((500, 500, 3), dtype='uint8')

#draw line
img = cv.line(zeros_array, (0,0), (500,500), (255,255,255), 3)

#draw circle
img = cv.circle(zeros_array, (250,250), 50, (0,255,0), 3)

#draw rectangle
img = cv.rectangle(zeros_array, (50,50), (470,470), (0,0,255), 3)

#draw ellipse
img = cv.ellipse(zeros_array, (200,200), (100,50), 0, 0, 360, (255,0,0), 3)

#draw polygon
vertices = np.array([[170,400],[150,100],[250,200],[250,300]], np.int32)
vertices = vertices.reshape((-1, 1, 2))
print(vertices)
img = cv.polylines(img, [vertices], True, (0, 255, 255), 3)

#write text
img = cv.putText(zeros_array, 'Learn OpenCV', (50,50), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255))

#show the Geometric shapes in image
cv.imshow('Geometric Shapes', zeros_array)

cv.waitKey(0)