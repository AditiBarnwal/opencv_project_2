# Image Segmentation using Morphological operations


#  Morphological operations are some simple operations based on the image shape.
#  It is normally performed on binary images.
#  2 basic morphological operators : Erosion and Dilation

#  To process on I’ll use OTSU’s threshold algorithm where this removes
#  over segmented result due to noise or any other irregularities in the image and implement with OpenCV.


# img transfer using threshold
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Meyer.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

cv2.imshow("thresh", thresh)


# This output shows that image is transformed by thresholding operation where foreground still contains some noises.
# Now,need to remove any small white noises in the image i.e. foreground. For this use morphological closing.
# To remove any small holes in the foreground object, use morphological closing.
# To obtain background we dilate the image. Dilation increases object boundary to background.


# noise removal using morphological closing operation
kernel = np.ones((3, 3), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# bg area using dilation
bg = cv2.dilate(closing, kernel, iterations=1)

# finding foreground area
dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 0)
ret, fg = cv2.threshold(dist_transform, 0.02*dist_transform.max(), 255, 0)

cv2.imshow('img', fg)

cv2.waitKey(0)
cv2.destroyAllWindows()
