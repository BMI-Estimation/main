import cv2
from imutils import is_cv2
from imutils import perspective
import numpy as np
from scipy.spatial import distance as dist

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

def findBoundingBox(contour):
  # compute the bounding box of the contour
  box = cv2.minAreaRect(contour)
  box = cv2.cv.BoxPoints(box) if is_cv2() else cv2.boxPoints(box)
  box = np.array(box, dtype="int")
  box = perspective.order_points(box)
  return box

def drawBoundingBoxes(img, box, refWidth, pixelsPerMetric):
  cv2.drawContours(img, [box.astype("int")], -1, (0, 255, 0), 2)
  # draw corners
  for (x, y) in box:
	  cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
  
  # compute the midpoints between corners
  (tl, tr, br, bl) = box
  (tltrX, tltrY) = midpoint(tl, tr)
  (blbrX, blbrY) = midpoint(bl, br)
  (tlblX, tlblY) = midpoint(tl, bl)
  (trbrX, trbrY) = midpoint(tr, br)

  # draw the midpoints on the image
  cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
  cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
  cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
  cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

  	# draw lines between the midpoints
  cv2.line(img, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
  cv2.line(img, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

  # compute the Euclidean distance between the midpoints
  dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
  dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

  # only set pixelspermetric for the first contour that is evaluated (left-most => reference object)
  if pixelsPerMetric is None:
	  pixelsPerMetric = dB/refWidth
	  print('pixelspermetric is ', pixelsPerMetric)

	# compute the size of the object
  dimA = dA / pixelsPerMetric
  dimB = dB / pixelsPerMetric

  # draw the object sizes on the image
  cv2.putText(img, "{:.1f}m".format(dimA),
		(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)
  cv2.putText(img, "{:.1f}m".format(dimB),
		(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
		0.65, (255, 255, 255), 2)

  return img, pixelsPerMetric