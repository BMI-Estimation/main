import argparse
import cv2
import initialise
from findPerson import findPersonInPhoto as persons
from findPerson import personArea, maskThickness
from referenceObject import findReferenceObject as findRef
import os
import csv

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--width", type=float, required=True, help="width of the left-most object in the image (in meters)")
ap.add_argument("-v", "--visualise", nargs='?', const=True, type=bool, required=False, default=False, help="show all images etc.")
ap.add_argument("-m", "--mask", nargs='?', const=True, type=bool, required=False, default=False, help="show masks on images.")
ap.add_argument("-g", "--gen", nargs='?', const=True, type=bool, required=False, default=False, help="Generate csv.")
ap.add_argument("-f", "--fimg", required=False, help="Front Input Image.")
ap.add_argument("-s", "--simg", required=False, help="Side Input Image.")
args = vars(ap.parse_args())

def gen():
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
	listOfFrontPixelsPerMetric = findRef(listOfFrontImages, args["width"], args["visualise"], args["mask"], listOfFrontImageNames)
	print('[INFO] Finding Side Ref. Metric')
	listOfSidePixelsPerMetric = findRef(listOfSideImages, args["width"], args["visualise"], args["mask"], listOfSideImageNames)
	print('[INFO] Finding Widths')
	widths = maskThickness(listOfFrontBinMasks, listOfFrontPixelsPerMetric)
	print('[INFO] Finding Depths')
	depths = maskThickness(listOfSideBinMasks, listOfSidePixelsPerMetric)

	for width, depth in zip(widths, depths):
		frontWriter.writerow(width)
		sideWriter.writerow(depth)
	csvFrontFile.close()
	csvSideFile.close()

def detect():
	print("Detect Mode, Arguments: ", args)
	# save images in root folder
	fImage = cv2.imread(args['fimg'])
	sImage = cv2.imread(args['simg'])
	print('[INFO] Finding Front and Side Masks')
	listOfPixelsPerMetric, listOfBinMasks = extractMasks([fImage, sImage], args)
	print('[INFO] Extracting Front and Side Dimensions')
	dimensions = maskThickness(listOfBinMasks, listOfPixelsPerMetric)
	frontImageDimensions = dimensions[0]
	sideImageDimensions = dimensions[1]
	csvFile = open('dimensions.csv', 'w', newline='')
	frontWriter = csv.writer(csvFile, delimiter=',')
	frontWriter.writerow(frontImageDimensions)
	frontWriter.writerow(sideImageDimensions)
	csvFile.close()
	cv2.destroyAllWindows()

def extractMasks(listOfImages, args):
	listOfBinMasks = persons(listOfImages, args["visualise"], args["mask"])
	listOfPixelsPerMetric = findRef(listOfImages, args["width"], args["visualise"], args["mask"], [args['fimg'], args['simg']])
	return listOfPixelsPerMetric, listOfBinMasks

if args["gen"]: gen()
else: detect()