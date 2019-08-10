import numpy as np
import cv2
from findPerson import findPersonInPhoto as persons
from referenceObject import findReferenceObject as findRef

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

def extractMasks(listOfImages, args):
	listOfBinMasks = persons(listOfImages, args["visualise"], args["mask"])
	listOfPixelsPerMetric = findRef(listOfImages, args["width"], args["visualise"], args["mask"], [args['fimg'], args['simg']])
	return listOfPixelsPerMetric, listOfBinMasks