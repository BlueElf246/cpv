# hese are the following steps for image segmentation using watershed algorithm:
#
# Step 1: Finding the sure background using morphological operation like opening and dilation.
#
# Step 2: Finding the sure foreground using distance transform.
#
# Step 3: Unknown area is the area neither lies in foreground and background and used it as a marker for watershed algorithm.
import cv2
import numpy as np
# reference: https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
# reference: https://www.aegissofttech.com/articles/watershed-algorithm-and-limitations.html

def water_shed(path):
    img= cv2.imread(path)
    # convert to gray
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Applying dilation for sure_bg detection
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Defining kernel for opening operation
    kernel= np.ones((3,3), np.uint8)
    # use open to remove any small white noises in the image
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    cv2.imshow('Open', opening)
    # After opening, will perform dilation, Dilation increases object boundary to background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Sure background image
    cv2.imshow('dilated', sure_bg)
    # foreground extraction
    # there are two options for fg extract: one is distance transform, second is erosion
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    # Finding the Unknown Area (Neither sure Foreground Nor for Background)
    unknown = np.subtract(sure_bg, sure_fg)
    cv2.imshow('unknown', unknown)
    # apply watershed algorithm
    ret, markers = cv2.connectedComponents(sure_fg)
    print(markers)
    # Add one so that sure background is not 1
    markers = markers + 1
    # Making the unknown area as 0
    markers[unknown == 255] = 0
    # cv2.imshow('markers2', markers)
    cv2.waitKey(0)
    markers = cv2.watershed(img, markers)
    # boundary region is marked with -1
    img[markers == -1] = (255, 0, 0)
    cv2.imshow('watershed', img)
    cv2.waitKey(0)
