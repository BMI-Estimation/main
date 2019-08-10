import cv2

def blurImage(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  for x in range(0, 1):
	  gray = cv2.GaussianBlur(gray, (7, 7), 0)

  return gray