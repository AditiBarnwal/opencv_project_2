# Morphological Operations in Image Processing (Opening)

# Morphological operations : used to extract image components that are useful in the representation
# and description of region shape.
# Morphological operations dependent on the picture shape.
# performed on binary images. needs 2 data sources :
#                                      - input image
#                                      - structuring component

# Morphological operators take an input image and a structuring component as input
# and these elements are then combines using the set operators.

# The objects in the input image are processed depending on attributes of the shape of the image,
# which are encoded in the structuring component.

# Opening is similar to erosion as it tends to remove the bright foreground pixels
# from the edges of regions of foreground pixels.

# Opening operation : used for removing internal noise in an image.
import cv2
import numpy

screenRead = cv2.VideoCapture(0)

# loops run if capturing is initialized
while(1):
    # reads frame from camera
    _, img = screenRead.read()
    # converts to HSV color space, OCV reads colors as BGR
    # frame is converted to hsv
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # defining the range of mask
    blue1 = numpy.array([110, 50, 50])
    blue2 = numpy.array([130, 255, 255])

    # initializing the mask to be
    # convoluted over input range
    mask = cv2.inRange(hsv, blue1, blue2)

    # passing the bitwise_and over each pixel convoluted
    res = cv2.bitwise_and(img, img, mask=mask)

    # defining kernel i.e. structuring element
    kernel = numpy.ones((5, 5), numpy.uint8)

    # opening function over the img and structuring element
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # mask and opening function is shown
    cv2.imshow("opening", opening)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# de-allocate any associated memory usage
cv2.destroyAllWindows()

# release the webcam
screenRead.release()

# Opening operator applying the erosion operation after dilation.
# It helps in removing the internal noise in the image.
