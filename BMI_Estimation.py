from findPerson import findPersonInPhoto as persons
from personMetrics import maskThickness
from referenceObject import findReferenceObject as findRef
import cv2

def gen():
	import argparse
	import csv
	import os
	
	ap = argparse.ArgumentParser()
	ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in meters)")
	ap.add_argument("-v", "--visualise", nargs='?', const=True, type=bool, required=False, default=False, help="show all images etc.")
	ap.add_argument("-m", "--mask", nargs='?', const=True, type=bool, required=False, default=False, help="show masks on images.")
	args = vars(ap.parse_args())

	csvFrontFile = open('front.csv', 'w', newline='')
	csvSideFile = open('side.csv', 'w', newline='')
	frontWriter = csv.writer(csvFrontFile, delimiter=',')
	sideWriter = csv.writer(csvSideFile, delimiter=',')
	listOfFrontImages = []
	listOfFrontImageNames = []
	listOfSideImages = []
	listOfSideImageNames = []
	widths = []
	depths= []

	print('[INFO] Reading Images')
	for filename in os.listdir('images'):
		if os.path.isdir('images/' + filename):
			continue
		name = 'images/' + filename
		if 'F' in filename or 'f' in filename:
			listOfFrontImageNames.append(name)
		elif 'S' in filename or 's' in filename:
			listOfSideImageNames.append(name)

	print('[INFO] Extracting Widths From Front Image')
	width = extractDimensions(listOfFrontImageNames, args, frontWriter)
	csvFrontFile.close()
	print('[INFO] Widths Extracted')

	print('[INFO] Extracting Depths From Side Image')
	depth = extractDimensions(listOfSideImageNames, args, sideWriter)
	csvSideFile.close()
	print('[INFO] Depths Extracted')

def extractDimensions(listOfNames, args, writer):
	for name in listOfNames:
		image = cv2.imread(name)
		BinMask = persons(image, args["visualise"], args["mask"])
		PixelsPerMetric = findRef(image, args["width"], args["visualise"], args["mask"])
		dimension = maskThickness(BinMask, PixelsPerMetric)
		writer.writerow(dimension)
		print('[INFO] Dimension Extracted from ', name)
	return

def detect(args):
	print("Detect Mode, Arguments: ", args)
	print('[INFO] Finding Front and Side Masks')
	listOfPixelsPerMetric, listOfBinMasks = extractMasks(args)
	print('[INFO] Extracting Front and Side Dimensions')
	dimensions = []
	
	for mask, ppm in zip(listOfBinMasks,listOfPixelsPerMetric):
		dimensions.append(maskThickness(mask, ppm))

	cv2.destroyAllWindows()
	return dimensions

def extractMasks(args):
	listOfBinMasks = []
	listOfPixelsPerMetric = []
	for img in args["images"]:
		image = cv2.imread(img)
		listOfBinMasks.append(persons(image, args["visualise"], args["mask"]))
		listOfPixelsPerMetric.append(findRef(image, args["width"], args["visualise"], args["mask"]))
	return listOfPixelsPerMetric, listOfBinMasks

if __name__ == "__main__" : gen()