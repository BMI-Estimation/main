import cv2
from imutils import grab_contours
from imutils import contours

def gray2binaryEdgedImage(gray):
  edged = cv2.Canny(gray, 100, 110)
  for x in range(0, 1):
	  edged = cv2.dilate(edged, None, iterations=20)
	  edged = cv2.erode(edged, None, iterations=20)
  return edged

def returnContours(edgedImage):
  # find contours in the edge map
  cnts = cv2.findContours(edgedImage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = grab_contours(cnts)
  # sort the contours from left-to-right
  (cnts, _) = contours.sort_contours(cnts)
  return cnts