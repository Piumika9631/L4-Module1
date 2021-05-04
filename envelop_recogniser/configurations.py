import cv2 as cv


#  read image
def read_image(selected_image_path):
    selectedInputImage = cv.imread(selected_image_path, cv.IMREAD_UNCHANGED)

    if selectedInputImage is None:
        print('Image not found. Load an image to proceed.')
        exit()
    else:
        return selectedInputImage


