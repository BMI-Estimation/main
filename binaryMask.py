import numpy as np
import cv2
from edgeDetection import closeImage, returnContours

def mask2binary(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  retval, binImage = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
  binImage = np.where(binImage > 0, 255, 0).astype('uint8')
  return binImage

def findBinaryImageEdges(img):
  edged = cv2.Canny(img, 100, 110)
  edged = closeImage(edged, 100)
  contours = returnContours(edged)
  contours = sorted(contours, key=lambda x: cv2.contourArea(x))
  return edged, contours[-1]



