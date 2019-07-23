# USAGE
# python object_size.py --image images/example_01.png --width 0.955
# python object_size.py --image images/example_02.png --width 0.955
# python object_size.py --image images/example_03.png --width 3.5

# import the necessary packages
# import tensorflow
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import matplotlib.pyplot as plt
import preprocessing
import edgeDetection
import boundingBoxes

# def midpoint(ptA, ptB):
# 	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in meters)")
args = vars(ap.parse_args())

# convert image to grayscale, and blur it to remove some noise
image = cv2.imread(args["image"])
gray = preprocessing.blurImage(image)

# perform edge detection, then rough image closing to complete edges
edged = edgeDetection.gray2binaryEdgedImage(gray)
# extract contours
contours = edgeDetection.returnContours(edged)
pixelsPerMetric = None

# loop over the contours individually
for c in contours:
	# if the contour is not sufficiently large, ignore it
	if cv2.contourArea(c) < 2000:
		continue

	orig = image.copy()
	# draw the outline of the contour's bounding box
	box = boundingBoxes.findBoundingBox(c)
	(tl, tr, br, bl) = box
	orig, pixelsPerMetric = boundingBoxes.drawBoundingBoxes(orig, box, args["width"], pixelsPerMetric)
	# show the output image with bounding box drawn
	cv2.imshow("Image", orig)

	# create a simple mask image similar to the loaded image 
	maskOrig = image.copy()
	mask = np.zeros(maskOrig.shape[:2], np.uint8) 
   
	# specify the background and foreground model 
	# using numpy the array is constructed of 1 row 
	# and 65 columns, and all array elements are 0 
	# Data type for the array is np.float64 (default) 
	backgroundModel = np.zeros((1, 65), np.float64) 
	foregroundModel = np.zeros((1, 65), np.float64) 
   
	# define the Region of Interest (ROI) 
	# as the coordinates of the rectangle 
	# where the values are entered as 
	# (startingPoint_x, startingPoint_y, width, height) 
	# these coordinates are according to the input image 
	# it may vary for different images 
	rectangle = (tl[0], tl[1], tr[0] - tl[0], bl[1] - tl[1]) 
	# apply the grabcut algorithm with appropriate 
	# values as parameters, number of iterations = 3  
	# cv2.GC_INIT_WITH_RECT is used because 
	# of the rectangle mode is used  
	cv2.grabCut(maskOrig, mask, rectangle, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT) 
	mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8') 
	maskOrig = maskOrig * mask2[:, :, np.newaxis]
	cv2.imshow('mask', maskOrig) 
	# plt.colorbar() 
	# plt.show() 
	cv2.waitKey(0)