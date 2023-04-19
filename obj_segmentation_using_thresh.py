# Segmentation using Thresholding

# Thresholding is a very popular segmentation technique : used for separating an object from its background.

# The process of thresholding involves, comparing each pixel value of the image
# (pixel intensity) to a specified threshold.
# This divides all the pixels of the input image into 2 groups:
#           - Pixels having intensity value lower than threshold.
#           - Pixels having intensity value greater than threshold.

# These 2 groups are now given different values, depending on various segmentation types.
# OpenCV supports 5 different thresholding schemes on Grayscale(8-bit) images using the function :
#

# {{  Double threshold(InputArray src, OutputArray dst, double thresh, double maxval, int type)  }}
#   {{               Parameters:
#
#                    InputArray src: Input Image (Mat, 8-bit or 32-bit)
#                    OutputArray dst: Output Image ( same size as input)
#                    double thresh: Set threshold value
#                    double maxval: maxVal, used in type 1 and 2
#                    int type* :Specifies the type of threshold to be use. (0-4)
#
#    }}
#
#    Thresholding types:
#    1. Binary Threshold(int type=0)
#    2. Inverted Binary Threshold(int type=1)
#    3. Truncate Thresholding(int type=2)
#    4. Threshold to Zero(int type=3)
#    5. Threshold to Zero, Inverted(int type=4)

# input RGB image is first converted to a grayscale image before thresholding is done.
