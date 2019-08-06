import preprocessing
import edgeDetection
import cv2
import boundingBoxes
import maskExtraction
import binaryMask

def findReferenceObject(clone, width, show, mask, filename):
  # convert image to grayscale, and blur it to remove some noise
  gray = preprocessing.blurImage(clone)
  # perform edge detection, then rough image closing to complete edges
  edged = edgeDetection.gray2binaryEdgedImage(gray)
  # extract contours
  contours = edgeDetection.returnContours(edged)
  pixelsPerMetric = None

  # if the contour is not sufficiently large, ignore it
  contours = [c for c in contours if cv2.contourArea(c) > 2000]
  # loop over the contours individually
  orig = clone.copy()
  # draw the outline of the contour's bounding box
  box = boundingBoxes.findBoundingBox(contours[0])
  orig, pixelsPerMetric = boundingBoxes.drawBoundingBoxes(orig, box, width, pixelsPerMetric)

  # create a blank mask image similar to the loaded image 
  maskOrig = clone.copy()
  # extract object mask
  maskOrig = maskExtraction.extractObjectForegroundMask(maskOrig, box)
  binImage = binaryMask.mask2binary(maskOrig)
  
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
      cv2.namedWindow(filename + " binmask", cv2.WINDOW_NORMAL)
      cv2.imshow(filename + " binmask", binImage)

  return pixelsPerMetric

  