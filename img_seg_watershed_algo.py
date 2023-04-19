# Image Segmentation with Watershed Algorithm

# Image segmentation is the process of dividing an image into several disjoint small local
# areas or cluster sets according to certain rules and principles.

# watershed algorithm(CV technique) used for image region segmentation.

# Segmentation process will take the similarity with adjacent pixels of the image as an
# important reference to connect pixels with similar spatial positions and gray values.
# Constitute a closed contour(outline), and this closure is an imp feature of the watershed algorithm.

# In short: it is an algorithm that correctly determines the “outline of an object“.
# Steps of process:
# 1.Marker Placement         2.Flooding        3.Catchment basin formation       4.Boundary indentification

# The resulting segmentation can be used for object recognition, image analysis, and feature extraction tasks.

# Step1
import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Image, display


# Step2 : Perform Preprocessing
# define a function “imshow” to display the processed image. The code loads an image and
# converts it to grayscale using OpenCV’s “cvtColor” method.
# grayscale image is stored in a variable “gray”.

def imshow(img, ax=None):
    if ax is None:
        ret, encoded = cv2.imencode("Meyer.jpg", img)
        display(Image(encoded))
    else:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.axis("off")


img = cv2.imread("Meyer.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Step 3: Threshold Processing
ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Step 4: Noise Removal
# To clean the object’s outline (boundary line), noise is removed using morphological gradient processing.
# noise removal
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=2)
# {{
# Step 5: Grasping the black background and foreground of the image

# The first operation is dilation using “cv2.dilate” which expands the bright regions of the image, creating
# the “sure_bg” variable representing the sure background area. Shown through “imshow” function.
# Create subplots with 1 row and 2 columns
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
# sure background area
sure_bg = cv2.dilate(bin_img, kernel, iterations=3)
imshow(sure_bg, axes[0, 0])
axes[0, 0].set_title('Sure Background')


# Distance transform
# The next operation is “cv2.distanceTransform” which calculates the distance of each white pixel in the
# binary image to the closest black pixel. The result is stored in the “dist” variable and displayed using “imshow”.
dist = cv2.distanceTransform(bin_img, cv2.DIST_L2, 5)
imshow(dist, axes[0, 1])
axes[0, 1].set_title('Distance Transform')


# foreground area
# Then, foreground area is obtained by applying a threshold on the “dist”
# variable using “cv2.threshold”. The threshold is set to 0.5 times the maximum value of “dist”.
ret, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
sure_fg = sure_fg.astype(np.uint8)
imshow(sure_fg, axes[1, 0])
axes[1, 0].set_title('Sure Foreground')


# unknown area
# Finally, unknown area is calculated as the difference between the sure background and
# sure foreground areas using “cv2.subtract”.
# The result is stored in the “unknown” variable and displayed using “imshow”.
unknown = cv2.subtract(sure_bg, sure_fg)
imshow(unknown, axes[1, 1])
axes[1, 1].set_title('Unknown')

plt.show()
# }}

# {{
# Step 6: Place markers on local minima
# There is a gray area between the white area in this part of the background and clearly visible
# white part of the foreground. This is still uncharted territory, an undefined part. So subtracting this area.
# Marker labelling
# step1. the “connectedcomponents”=cc method from OpenCV is used to find the
# cc in the sure foreground image “sure_fg”. The result is stored in “markers”.
# sure foreground
ret, markers = cv2.connectedComponents(sure_fg)

# step2: To distinguish the background and foreground, the values in “markers” are incremented by 1.
# Add one to all labels so that background is not 0, but 1
markers += 1

# step3: The unknown region, represented by pixels with a value of 255 in “unknown”, is labeled with 0 in “markers”.
# mark the region of unknown with zero
markers[unknown == 255] = 0

# step4: Finally, the “markers” image is displayed using Matplotlib’s “imshow”
# method with a color map of “tab20b”.Result is shown in a figure of size 6×6.
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(markers, cmap="tab20b")
ax.axis('off')
plt.show()

# }}

# Step 7: Apply Watershed Algorithm to Markers
# watershed Algorithm
markers = cv2.watershed(img, markers)

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(markers, cmap="tab20b")
ax.axis('off')
plt.show()

labels = np.unique(markers)

coins = []
for label in labels[2:]:
    # Create a binary image in which only the area of the label is in the foreground
    # and the rest of the image is in the background
    target = np.where(markers == label, 255, 0).astype(np.uint8)

    # Perform contour extraction on the created binary image
    contours, hierarchy = cv2.findContours(
        target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    coins.append(contours[0])

# Draw the outline
img = cv2.drawContours(img, coins, -1, color=(0, 23, 223), thickness=2)
cv2.imshow("img", img)
cv2.imshow("binary img", bin_img)
cv2.imshow("img", img)
cv2.waitKey(0)
