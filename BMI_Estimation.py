# USAGE
# python object_size.py --width 0.955

import argparse
import cv2
import numpy as np
import preprocessing
import edgeDetection
import boundingBoxes
import maskExtraction
import binaryMask
import os
import csv

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in meters)")
args = vars(ap.parse_args())

csvFile = open('person.csv', 'w')

for filename in os.listdir('images'):
	if os.path.isdir('images/' + filename):
		continue

	# convert image to grayscale, and blur it to remove some noise
	image = cv2.imread('images/' + filename)
	gray = preprocessing.blurImage(image)

	# perform edge detection, then rough image closing to complete edges
	edged = edgeDetection.gray2binaryEdgedImage(gray)
	# cv2.namedWindow("edged", cv2.WINDOW_NORMAL)
	# cv2.imshow("edged", edged)
	# extract contours
	contours = edgeDetection.returnContours(edged)
	pixelsPerMetric = None

	# if the contour is not sufficiently large, ignore it
	contours = [c for c in contours if cv2.contourArea(c) > 2000]
	# loop over the contours individually
	for c in contours[0:2]:
	# for c in contours:
		orig = image.copy()
		# draw the outline of the contour's bounding box
		box = boundingBoxes.findBoundingBox(c)
		orig, pixelsPerMetric = boundingBoxes.drawBoundingBoxes(orig, box, args["width"], pixelsPerMetric)
		# show the output image with bounding box drawn
		# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
		# cv2.imshow("Image", orig)

		# create a blank mask image similar to the loaded image 
		maskOrig = image.copy()
		# extract object mask
		maskOrig = maskExtraction.extractObjectForegroundMask(maskOrig, box)
		# cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
		# cv2.imshow('mask', maskOrig) 

		binImage = binaryMask.mask2binary(maskOrig)
		cv2.namedWindow("binmask", cv2.WINDOW_NORMAL)
		cv2.imshow("binmask", binImage)

		thickness = [sum(row)/(255*pixelsPerMetric) for row in binImage]
		thickness = [thickness[index] for index in np.nonzero(thickness)[0]]
		thickness.insert(0, len(thickness)/pixelsPerMetric)
		# print(thickness)
		cv2.waitKey(0)
		writer = csv.writer(csvFile)
		writer.writerows(map(lambda x: [x], thickness))
		csvFile.write('END\n')

csvFile.close()
