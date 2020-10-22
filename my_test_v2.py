import cv2



image = cv2.imread('images/address2.jpeg')


original = image.copy()



gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,110,210)
# thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


height, width = image.shape[:2]

cv2.imshow('Thresh',edges)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,10))
dilate = cv2.dilate(edges, kernel, iterations=2)

cv2.imshow('Dilated',dilate)

# Extract each line contour
lines = []
contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
print(contours[0])
for line in contours:

    x,y,w,h = cv2.boundingRect(line)
    cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 2)

print(len(contours))
# cv2.imshow("Image",image)

cv2.waitKey(0)
