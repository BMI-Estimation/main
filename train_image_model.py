import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--batch", type=int, required=True, help="Batch Size.")
ap.add_argument("-e", "--epochs", type=int, required=True, help="Number of Epochs per training cycle.")
ap.add_argument("-n", "--number", type=int, required=True, help="Number of training iterations.")
ap.add_argument("-f", "--front", nargs='?', const=True, type=bool, required=False, default=False, help="Train Front Model.")
ap.add_argument("-s", "--side", nargs='?', const=True, type=bool, required=False, default=False, help="Train Side Model.")
ap.add_argument("-v", "--visualize", nargs='?', const=True, type=bool, required=False, default=False, help="View Training Graphs.")
ap.add_argument("-he", "--height", nargs='?', const=True, type=bool, required=False, default=False, help="Train vs Height.")
ap.add_argument("-m", "--mass", nargs='?', const=True, type=bool, required=False, default=False, help="Train vs Weight.")
args = vars(ap.parse_args())

if args["height"] and args["mass"]:
	print("Please Either Specify whether training vs. Height (-h), Mass (-m) or BMI (default), and not multiple.")
	exit()

fileNames = {}

if args["side"] and args["front"]:
	print("Please Specify Either Front or Side Model training, not both.")
	exit()
elif args["side"]:
	fileNames["file"] = "side.csv"
	fileNames["Classic_Model_File"] = 'Classical_Side.h5'
	fileNames["Cross_Model_File"] = 'Cross_Side.h5'
	fileNames["Best_Classical"] = 'Best_Class_Side.h5'
	fileNames["Best_Cross"] = 'Best_Cross_Side.h5'
elif args["front"]:
	fileNames["file"] = "front.csv"
	fileNames["Classic_Model_File"] = 'Classical_Front.h5'
	fileNames["Cross_Model_File"] = 'Cross_Front.h5'
	fileNames["Best_Classical"] = 'Best_Class_Front.h5'
	fileNames["Best_Cross"] = 'Best_Cross_Front.h5'
else:
	print("Please Specify Front or Side Model training.")
	exit()

from trainingFunctions import trainWithBMI, trainMass, trainHeight
import numpy as np
import csv

# load dataset inputs
dataframe_traning_inputs = open(fileNames["file"], 'r')
reader = csv.reader(dataframe_traning_inputs, delimiter=",")
Input_parameters = [[float(entry) for entry in row] for row in reader]
badDataIndex = [index for index, row in enumerate(Input_parameters) if row[1] > 2.5]
Input_parameters = [row for index, row in enumerate(Input_parameters) if index not in badDataIndex]
Input_parameters = np.asarray(Input_parameters)
dataframe_traning_inputs.close()

# load dataset BMI
BMI_file = "BMI.csv"
dataframe_training_outputs = open(BMI_file, 'r')
reader = csv.reader(dataframe_training_outputs, delimiter=",")
output = [[float(entry) for entry in row] for row in reader]
output = [row for index, row in enumerate(output) if index not in badDataIndex]
output = np.asarray(output)
dataframe_training_outputs.close()

if args["height"]:
	height = output[:,1]
	trainHeight(Input_parameters[:,1], height, args, fileNames)
elif args["mass"]:
	mass = output[:,0]
	trainMass(Input_parameters, mass, args, fileNames)
else:
	BMI = output[:,2]
	trainWithBMI(Input_parameters, BMI, args, fileNames)