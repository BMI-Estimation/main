import cv2
from imutils import grab_contours
from imutils import contours
import numpy as np

def gray2binaryEdgedImage(gray):
	edged = cv2.Canny(gray, 100, 110)
	# perform rough closing on the image to remove small contours
	for x in range(0, 1):
		edged = cv2.dilate(edged, None, iterations=20)
		edged = cv2.erode(edged, None, iterations=20)
	return edged

def returnContours(edgedImage):
	# find contours in the edge map
	contoursInImage = cv2.findContours(edgedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	contoursInImage = grab_contours(contoursInImage)
	# sort the contours from left-to-right
	(contoursInImage, _) = contours.sort_contours(contoursInImage)
	return contoursInImage

def detectRectangle(image):
	# find all the 'black' shapes in the image
	lower = np.array([0, 0, 0])
	upper = np.array([130, 130, 130])
	shapeMask = cv2.inRange(image, lower, upper)

	contoursInImage = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
	contoursInImage = grab_contours(contoursInImage)
	contoursInImage = [c for c in contoursInImage if cv2.contourArea(c) > 20000]
	(contoursInImage, _) = contours.sort_contours(contoursInImage)

	# find contours that approximate a rectangle
	contoursInImage = [c for c in contoursInImage if len(cv2.approxPolyDP(c, 0.1*cv2.arcLength(c, True), True)) == 4]

	return contoursInImage
