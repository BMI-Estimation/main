from edgeDetection import gray2binaryEdgedImage, returnContours
import cv2
from boundingBoxes import findBoundingBox, drawBoundingBoxes
from binaryMask import mask2binary
import numpy as np

def blurImage(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	for x in range(0, 1):
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

	return gray

def extractObjectForegroundMask(img, box):
	(tl, tr, br, bl) = box
	mask = np.zeros(img.shape[:2], np.uint8) 
	
	# specify the background and foreground model 
	backgroundModel = np.zeros((1, 65), np.float64) 
	foregroundModel = np.zeros((1, 65), np.float64) 
	 
	# define the Region of Interest (ROI) (startingPoint_x, startingPoint_y, width, height) 
	rectangle = (tl[0], tl[1], tr[0] - tl[0], bl[1] - tl[1]) 
	
	# apply the grabcut algorithm
	cv2.grabCut(img, mask, rectangle, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT) 
	mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
	img = img * mask2[:, :, np.newaxis]
	return img

def findReferenceObject(clone, width, show, mask):
	# convert image to grayscale, and blur it to remove some noise
	gray = blurImage(clone)
	# perform edge detection, then rough image closing to complete edges
	edged = gray2binaryEdgedImage(gray)
	# extract contours
	contours = returnContours(edged)
	pixelsPerMetric = None

	# if the contour is not sufficiently large, ignore it
	contours = [c for c in contours if cv2.contourArea(c) > 2000]
	# loop over the contours individually
	orig = clone.copy()
	# draw the outline of the contour's bounding box
	box = findBoundingBox(contours[0])
	orig, pixelsPerMetric = drawBoundingBoxes(orig, box, width, pixelsPerMetric)

	# create a blank mask image similar to the loaded image 
	maskOrig = clone.copy()
	# extract object mask
	maskOrig = extractObjectForegroundMask(maskOrig, box)
	binImage = mask2binary(maskOrig)
		
	if show or mask:
		if show:
			# cv2.namedWindow("edged", cv2.WINDOW_NORMAL)
			# cv2.imshow("edged", edged)
			# show the output image with bounding box drawn
			cv2.namedWindow("Box", cv2.WINDOW_NORMAL)
			cv2.imshow("Box", orig)
			cv2.namedWindow("Refmask", cv2.WINDOW_NORMAL)
			cv2.imshow('Refmask', maskOrig)
		if mask:
			cv2.namedWindow("Ref binmask", cv2.WINDOW_NORMAL)
			cv2.imshow("Ref binmask", binImage)
		cv2.waitKey(0)

	return pixelsPerMetric
	