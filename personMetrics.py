from binaryMask import mask2binary
import cv2
import numpy as np

def personArea(binImage, pixelsPerMetric):
	metersperpix = 1/pixelsPerMetric
	areaperpix = metersperpix*metersperpix
	numPixelsPerRow = [sum(row)/(255) for row in binImage]
	numPixelsPerRow = [numPixelsPerRow[index] for index in np.nonzero(numPixelsPerRow)[0]]
	areaPerRow = [areaperpix*numPix for numPix in numPixelsPerRow]
	totArea = sum(areaPerRow)
	return totArea

def maskThickness(BinMask, PixelsPerMetric):
	# find mask thicknesses 
	thickness = [sum(row)/(255*PixelsPerMetric) for row in BinMask]
	thickness = [thickness[index] for index in np.nonzero(thickness)[0]]
	height = len(thickness)/PixelsPerMetric
	# split thickness vector into 8 sections which best describe the human body
	thickness = np.array_split(thickness, 8)
	thickness = [max(section) for section in thickness]
	thickness = [thickness[index] for index in range(1,5)]
	thickness.insert(0, height)
	# print(thickness)
	area = personArea(BinMask, PixelsPerMetric)
	# print(area)
	thickness.insert(0, area)
	# output vector now in the form [area, height, slice1,slice2, sclice3, slice4]
	return thickness

def extractMaskFromROI(img):
	mask = np.zeros(img.shape[:2], np.uint8) 	
	# specify the background and foreground model 
	backgroundModel = np.zeros((1, 65), np.float64) 
	foregroundModel = np.zeros((1, 65), np.float64) 
	 
	# define the Region of Interest (ROI) (startingPoint_x, startingPoint_y, width, height) 
	rectangle = (0, 0, img.shape[1] - 1, img.shape[0] - 1)
	
	# apply the grabcut algorithm
	cv2.grabCut(img, mask, rectangle, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_RECT) 
	mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
	img = img * mask2[:, :, np.newaxis]
	img = mask2binary(img)
	return img