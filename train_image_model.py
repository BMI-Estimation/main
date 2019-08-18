import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-b", "--batch", type=int, required=True, help="Batch Size.")
ap.add_argument("-e", "--epochs", type=int, required=True, help="Number of Epochs per training cycle.")
ap.add_argument("-n", "--number", type=int, required=True, help="Number of training iterations.")
ap.add_argument("-f", "--front", nargs='?', const=True, type=bool, required=False, default=False, help="Train Front Model.")
ap.add_argument("-s", "--side", nargs='?', const=True, type=bool, required=False, default=False, help="Train Side Model.")
ap.add_argument("-v", "--visualize", nargs='?', const=True, type=bool, required=False, default=False, help="View Training Graphs.")
ap.add_argument("-hw", "--heightWeight", nargs='?', const=True, type=bool, required=False, default=False, help="Train vs Height and Weight instead of BMI.")
args = vars(ap.parse_args())

fileNames = {}

if args["side"] and args["front"]:
    print("Please Specify Either Front or Side Model training, not both.")
    exit()
elif args["side"]:
    fileNames["file"] = "side.csv"
    fileNames["Final_Model_File"] = "Final_Model_Side.h5"
    fileNames["Low_Score_File"] = "Lowest_Score_Side.txt"
    fileNames["Classic_Model_File"] = 'Classical_Side.h5'
    fileNames["Cross_Model_File"] = 'Cross_Side.h5'
    fileNames["Best_Classical"] = 'Best_Class_Side.h5'
    fileNames["Best_Cross"] = 'Best_Cross_Side.h5'
elif args["front"]:
    fileNames["file"] = "front.csv"
    fileNames["Final_Model_File"] = "Final_Model_Front.h5"
    fileNames["Low_Score_File"] = "Lowest_Score_Front.txt"
    fileNames["Classic_Model_File"] = 'Classical_Front.h5'
    fileNames["Cross_Model_File"] = 'Cross_Front.h5'
    fileNames["Best_Classical"] = 'Best_Class_Front.h5'
    fileNames["Best_Cross"] = 'Best_Cross_Front.h5'
else:
    print("Please Specify Front or Side Model training.")
    exit()

from trainingFunctions import trainWithBMI
import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
import csv
import os
import datetime

# define base model
def baseline_model():
	# create model
    regressor = Sequential()
    regressor.add(Dense(7, input_dim=6, activation="relu"))
    regressor.add(Dense(5, activation="relu"))
    regressor.add(Dense(1, activation="linear"))
    regressor.compile(optimizer='adam', loss='mean_absolute_error')
    return regressor

# Initialise Models and Folder Structure
Classic_Model = baseline_model()
Cross_Val_Regressor = KerasRegressor(build_fn=baseline_model, epochs=args['epochs'], batch_size=args['batch'], verbose=1)

NetworkArc = [str(row.units) for row in Classic_Model.model.layers]
NetworkArc = '-'.join(NetworkArc)
today = str(datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S"))
fileNames["directory"] = 'models/' + NetworkArc + '/' + today + '/'
os.makedirs(fileNames["directory"])

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

if args["heightWeight"]:
    print('why')
    mass = output[:,0]
    height = output[:,1]
else:
    BMI = output[:,2]
    trainWithBMI(Input_parameters, BMI, args, Classic_Model, Cross_Val_Regressor, fileNames)