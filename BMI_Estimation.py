# USAGE
# python object_size.py --image images/example_01.png --width 0.955
# python object_size.py --image images/example_02.png --width 0.955
# python object_size.py --image images/example_03.png --width 3.5

import argparse
import cv2
from matplotlib import pyplot as plt
import numpy as np
import preprocessing
import edgeDetection
import boundingBoxes
import maskExtraction
import binaryMask

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
	orig, pixelsPerMetric = boundingBoxes.drawBoundingBoxes(orig, box, args["width"], pixelsPerMetric)
	# show the output image with bounding box drawn
	# cv2.imshow("Image", orig)

	# create a blank mask image similar to the loaded image 
	maskOrig = image.copy()
	# extract object mask
	maskOrig = maskExtraction.extractObjectForegroundMask(maskOrig, box)
	# cv2.imshow('mask', maskOrig) 

	binImage = binaryMask.mask2binary(maskOrig)
	cv2.imshow("binmask", binImage)

	thickness = [sum(row)/(255*pixelsPerMetric) for row in binImage]
	thickness = [thickness[index] for index in np.nonzero(thickness)[0]]
	print(thickness, len(thickness)/pixelsPerMetric)

	cv2.waitKey(0) 