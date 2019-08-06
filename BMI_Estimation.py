import argparse
import csv
import cv2
import initialise
from findPerson import findPersonInPhoto as persons
from findPerson import personArea, maskThickness
from referenceObject import findReferenceObject as findRef
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in meters)")
ap.add_argument("-v", "--visualise", nargs='?', const=True, type=bool, required=False, default=False, help="show all images etc.")
ap.add_argument("-m", "--mask", nargs='?', const=True, type=bool, required=False, default=False, help="show masks on images.")
args = vars(ap.parse_args())

csvFrontFile = open('front.csv', 'w')
csvSideFile = open('side.csv', 'w')

frontWriter = csv.writer(csvFrontFile, delimiter=',')
sideWriter = csv.writer(csvSideFile, delimiter=',')

listOfFrontImages = []
listOfFrontImageNames = []
listOfSideImages = []
listOfSideImageNames = []

widths = []
depths= []

# Read in images from folder
print('[INFO] Reading Images')
for filename in os.listdir('images'):
	if os.path.isdir('images/' + filename):
		continue

	image = cv2.imread('images/' + filename)
	if 'Front' in filename or 'F' in filename:
		listOfFrontImages.append(image)
		listOfFrontImageNames.append(filename)
	elif 'Side' in filename or 'S' in filename:
		listOfSideImages.append(image)
		listOfSideImageNames.append(filename)

print('[INFO] Extracting Front Masks')
listOfFrontBinMasks = persons(listOfFrontImages, args["visualise"], args["mask"])

print('[INFO] Extracting Side Masks')
listOfSideBinMasks = persons(listOfSideImages, args["visualise"], args["mask"])

print('[INFO] Finding Front Ref. Metric')
listOfFrontPixelsPerMetric = findRef(listOfFrontImages, args["width"], args["visualise"], args["mask"],
																		listOfFrontImageNames)

print('[INFO] Finding Side Ref. Metric')
listOfSidePixelsPerMetric = findRef(listOfSideImages, args["width"], args["visualise"], args["mask"],
																		listOfSideImageNames)
	
print('[INFO] Finding Widths')
widths = maskThickness(listOfFrontBinMasks, listOfFrontPixelsPerMetric)
print('[INFO] Finding Depths')
depths = maskThickness(listOfSideBinMasks, listOfSidePixelsPerMetric)

for width, depth in zip(widths, depths):
	frontWriter.writerow(width)
	sideWriter.writerow(depth)
cv2.destroyAllWindows()

csvFrontFile.close()
csvSideFile.close()
