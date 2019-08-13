import cv2
from initialise import model, CLASS_NAMES, color
from mrcnn import visualize
import numpy as np
from binaryMask import mask2binary

def findPersonInPhoto(image, show, showMask):
	# convert to rgb image for model
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  # perform forward pass of the network
  print("[INFO] making predictions with Mask R-CNN...")
  r = model.detect([image], verbose=1)[0]
  # loop over of the detected object's bounding boxes and masks
  for i in range(0, r["rois"].shape[0]):
	 	# extract the class ID and mask
    classID = r["class_ids"][i]
    # ignore all non-people objects
    if CLASS_NAMES[classID] != 'person':
      continue

    clone = image.copy()
    mask = r["masks"][:, :, i]
    # visualize the pixel-wise mask of the object
    image = visualize.apply_mask(image, mask, color, alpha=0.5)
    # convert the image to BGR for OpenCV use
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    (startY, startX, endY, endX) = r["rois"][i]
      
    # extract the ROI of the image, use foreground extraction for mask
    roi = clone[startY:endY, startX:endX]
    roiMask = extractMaskFromROI(roi)
    # extract the mask produced by CNN
    visMask = (mask * 255).astype("uint8")
    visMask = visMask[startY:endY, startX:endX]
    # extract overlapping regions of both masks to minimalise mask errors.
    finalMask = cv2.bitwise_and(roiMask, visMask)

    if show or showMask:
      if show:
        cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
        cv2.imshow("ROI", roi)
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        cv2.imshow("Output", image)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
        cv2.imshow("Mask", visMask)
        cv2.namedWindow('roi mask', cv2.WINDOW_NORMAL)
        cv2.imshow('roi mask', roiMask)
      if showMask:
        cv2.namedWindow('final mask', cv2.WINDOW_NORMAL)
        cv2.imshow('final mask', finalMask)
      cv2.waitKey(0)
    break
  return finalMask


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