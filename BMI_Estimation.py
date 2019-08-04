# USAGE
# python object_size.py --width 0.955

import argparse
import csv
import cv2
import numpy as np
from initialise import CLASS_NAMES, COLORS, config
import preprocessing
import edgeDetection
import boundingBoxes
import maskExtraction
import binaryMask
import os
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import visualize

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in meters)")
args = vars(ap.parse_args())

csvFile = open('person.csv', 'w')

print("[INFO] loading Mask R-CNN model...")
model = modellib.MaskRCNN(mode="inference", config=config, model_dir="logs")
model.load_weights("mask_rcnn_coco.h5", by_name=True)

for filename in os.listdir('images'):
	if os.path.isdir('images/' + filename):
		continue

	image = cv2.imread('images/' + filename)
	# convert to rgb image for model
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# perform forward pass of the network
	print("[INFO] making predictions with Mask R-CNN...")
	r = model.detect([image], verbose=1)[0]

	# loop over of the detected object's bounding boxes and masks
	for i in range(0, r["rois"].shape[0]):
		# extract the class ID and mask for the current detection, then
		# grab the color to visualize the mask (in BGR format)
		classID = r["class_ids"][i]
		# ignore all non-people objects
		if CLASS_NAMES[classID] != 'person':
			continue
		mask = r["masks"][:, :, i]
		color = COLORS[classID][::-1]
		# visualize the pixel-wise mask of the object
		image = visualize.apply_mask(image, mask, color, alpha=0.5)
		(startY, startX, endY, endX) = r["rois"][i]
		# convert the image to BGR for OpenCV use
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		clone = image.copy()
		# extract the ROI of the image
		roi = clone[startY:endY, startX:endX]
		visMask = (mask * 255).astype("uint8")
		# instance = cv2.bitwise_and(roi, roi, mask=visMask)
		cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
		cv2.imshow("ROI", roi)
		cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
		cv2.imshow("Mask", visMask)
		# cv2.namedWindow("Segmented", cv2.WINDOW_NORMAL)
		# cv2.imshow("Segmented", instance)
		cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
		cv2.imshow("Output", image)
		cv2.waitKey()
		break

	# # convert image to grayscale, and blur it to remove some noise
	# gray = preprocessing.blurImage(image)

	# # perform edge detection, then rough image closing to complete edges
	# edged = edgeDetection.gray2binaryEdgedImage(gray)
	# # cv2.namedWindow("edged", cv2.WINDOW_NORMAL)
	# # cv2.imshow("edged", edged)
	# # extract contours
	# contours = edgeDetection.returnContours(edged)
	# pixelsPerMetric = None

	# # if the contour is not sufficiently large, ignore it
	# contours = [c for c in contours if cv2.contourArea(c) > 2000]
	# # loop over the contours individually
	# for c in contours[0:2]:
	# # for c in contours:
	# 	orig = image.copy()
	# 	# draw the outline of the contour's bounding box
	# 	box = boundingBoxes.findBoundingBox(c)
	# 	orig, pixelsPerMetric = boundingBoxes.drawBoundingBoxes(orig, box, args["width"], pixelsPerMetric)
	# 	# show the output image with bounding box drawn
	# 	# cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
	# 	# cv2.imshow("Image", orig)

	# 	# create a blank mask image similar to the loaded image 
	# 	maskOrig = image.copy()
	# 	# extract object mask
	# 	maskOrig = maskExtraction.extractObjectForegroundMask(maskOrig, box)
	# 	# cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
	# 	# cv2.imshow('mask', maskOrig) 

	# 	binImage = binaryMask.mask2binary(maskOrig)
	# 	cv2.namedWindow("binmask", cv2.WINDOW_NORMAL)
	# 	cv2.imshow("binmask", binImage)

	# 	thickness = [sum(row)/(255*pixelsPerMetric) for row in binImage]
	# 	thickness = [thickness[index] for index in np.nonzero(thickness)[0]]
	# 	thickness.insert(0, len(thickness)/pixelsPerMetric)
	# 	# print(thickness)
	# 	cv2.waitKey(0)
	# 	writer = csv.writer(csvFile)
	# 	writer.writerows(map(lambda x: [x], thickness))
	# 	csvFile.write('END\n')

csvFile.close()
