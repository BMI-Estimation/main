import numpy as np
import cv2

def mask2binary(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	retval, binImage = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY)
	binImage = np.where(binImage > 0, 255, 0).astype('uint8')
	return binImage
