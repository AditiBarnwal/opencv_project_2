# Image Segmentation using K - means Clustering

import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("Meyer.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#plt.imshow(img.reshape((28, 28)))
#plt.show()
# Reshaping the image into a 2D array of pixels and 3 color values (RGB)
pixels_val = img.reshape((-1, 3))

# cvt to float type
pixels_val = np.float32(pixels_val)


# TermCriteria() The maximum number of iterations or elements to compute.
# The desired accuracy or change in parameters at which the iterative algorithm stops.

# max_iter : int, default: 300 Maximum number of iterations of the k-means algorithm for a single run.
# But in my opinion if I have 100 Objects the code must run 100 times,
# if I have 10.000 Objects the code must run 10.000 times to classify every object.
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
k = 100
retval, labels, centers = cv2.kmeans(pixels_val, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# cvt data into 8-bit-value
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]
segmented_img = segmented_data.reshape((img.shape))

cv2.imshow("segmented_img", segmented_img)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
