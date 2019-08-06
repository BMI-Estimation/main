# USAGE
# python object_size.py --width 0.955

import argparse
import csv
import cv2
import numpy as np
import initialise
from findPerson import findPersonInPhoto as person
from findPerson import personArea
from referenceObject import findReferenceObject as findRef
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in meters)")
ap.add_argument("-v", "--visualise", nargs='?', const=True, type=bool, required=False, default=False, help="show all images etc.")
ap.add_argument("-m", "--mask", nargs='?', const=True, type=bool, required=False, default=False, help="show masks on images.")
args = vars(ap.parse_args())

csvFile = open('person.csv', 'w')

for filename in os.listdir('images'):
	if os.path.isdir('images/' + filename):
		continue

	image = cv2.imread('images/' + filename)
	clone = image.copy()

	pixelsPerMetric = findRef(clone, args["width"], args["visualise"], args["mask"], filename)
	roi, binImage = person(image, args["visualise"], args["mask"])
	
	thickness = [sum(row)/(255*pixelsPerMetric) for row in binImage]
	thickness = [thickness[index] for index in np.nonzero(thickness)[0]]
	thickness.insert(0, len(thickness)/pixelsPerMetric)
	# print(thickness)
	area = personArea(binImage, pixelsPerMetric)
	# print(area)
	thickness.insert(0, area)
	# output vector now in the form [area, height, slice1, slice2, sclice3, ...]

	writer = csv.writer(csvFile)
	writer.writerows(map(lambda x: [x], thickness))
	csvFile.write('END\n')
	cv2.destroyAllWindows()

csvFile.close()
