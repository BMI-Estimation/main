import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--visualize", nargs='?', const=True, type=bool, required=False, default=False, help="View Training Graphs.")
ap.add_argument("-b", "--batch", type=int, required=True, help="Batch Size.")
ap.add_argument("-e", "--epochs", type=int, required=True, help="Number of Epochs per training cycle.")
ap.add_argument("-n", "--iter", type=int, required=True, help="Iterations.")
ap.add_argument("-f", "--fold", type=int, required=True, help="Folds.")
ap.add_argument("-s", "--strat", nargs='?', const=True, type=bool, required=False, default=False, help="Stratified.")
args = vars(ap.parse_args())

import numpy as np
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
import csv
import os
import datetime
from trainingFunctions import overallscore, baseline_model, showBMIGraphs, Final_Graphs, trainCompensator
from keras import optimizers

fileNames = {}

# Required files
Front_input_file = "front.csv"
Side_input_file = "side.csv"
Final_BMI_Model = 'Final_Model_BMI.h5'
Low_Score_File = 'Lowest_Score_BMI.txt'
fileNames["Classic_Model_File"] = 'Classical_BMI.h5'
fileNames["Cross_Model_File"] = 'Cross_BMI.h5'
fileNames["Best_Classic_File"] = 'Best_Classic.h5'
fileNames["Best_Cross_File"] = 'Best_Cross.h5'
Front_Model_file = 'Front.h5'
Side_Model_file = 'Side.h5'
Path_Extension = '/BMI_comp/'

# load dataset inputs
dataframe_traning_Side = open(Side_input_file, 'r')
reader = csv.reader(dataframe_traning_Side, delimiter=",")
Input_parameters_Side = [[float(entry) for entry in row] for row in reader]
badDataIndex = [index for index, row in enumerate(Input_parameters_Side) if row[1] > 2.5]

dataframe_traning_Front = open(Front_input_file, 'r')
reader = csv.reader(dataframe_traning_Front, delimiter=",")
Input_parameters_Front = [[float(entry) for entry in row] for row in reader]
[badDataIndex.append(index) for index, row in enumerate(Input_parameters_Front) if row[1] > 2.5 and index not in badDataIndex]

Input_parameters_Side = [row for index, row in enumerate(Input_parameters_Side) if index not in badDataIndex]
Input_parameters_Side = np.asarray(Input_parameters_Side)
Input_parameters_Front = [row for index, row in enumerate(Input_parameters_Front) if index not in badDataIndex]
Input_parameters_Front = np.asarray(Input_parameters_Front)

# Load front and side models
Front_Model = load_model(Front_Model_file)
Side_Model = load_model(Side_Model_file)

# Get respective predictions
Y_Front = Front_Model.predict(Input_parameters_Front)
Y_Side = Side_Model.predict(Input_parameters_Side)

# filter data of points whose predictions contain errors
diff = Y_Front - Y_Side
diff = [i for i, dif in enumerate(diff) if dif > 5]
Y_Front = [yf for i, yf in enumerate(Y_Front) if i not in diff]
Y_Side = [ys for i, ys in enumerate(Y_Side) if i not in diff]

Y_Front = np.asarray(Y_Front)
Y_Side = np.asarray(Y_Side)

# load dataset BMI
BMI_file = "BMI.csv"
dataframe_traning_outputs = open(BMI_file, 'r')
reader = csv.reader(dataframe_traning_outputs, delimiter=",")
BMI = [[float(entry) for entry in row] for row in reader]
BMI = [row for index, row in enumerate(BMI) if index not in badDataIndex]
BMI = [b for i, b in enumerate(BMI) if i not in diff]
BMI = np.asarray(BMI)
BMI = BMI[:,2]

# split into input (X) and output (Y) variables
X = np.column_stack((Y_Side,Y_Front))
Y = BMI
Y= Y.reshape(-1,1)
# model dimensions
neuronsPerLayerExceptOutputLayer = [3,2]

# Classic and cross model creation
build = baseline_model(2, neuronsPerLayerExceptOutputLayer, 0.001)
Regressor = build()
Cross_Val_Regressor = KerasRegressor(build_fn=build, epochs=args['epochs'], batch_size=args['batch'], verbose=1)

# Directory functionality
NetworkArc = [str(row.units) for row in Regressor.model.layers]
NetworkArc = '-'.join(NetworkArc)
today = str(datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S"))
directory = 'models/' + today + Path_Extension + NetworkArc + '/'
os.makedirs(directory)

# create infofiles
infoFile = open(directory + 'info.txt', 'w', newline='')

# Separate file data into seen and unseen data prior to model comparison
seed = 10
np.random.seed(seed)
X_full = X.copy()
Y_full = Y.copy()
X,X_Unseen,Y,Y_Unseen = train_test_split(X,Y,test_size=0.2)
# Prediction Averages performance
Y_Average =(Y_Side+Y_Front)/2
MSE = mean_squared_error(Y_full, Y_Average)
MAE = mean_absolute_error(Y_full, Y_Average)
Max = max_error(Y_full, Y_Average)
Line = "MSE:"+str(MSE)+" MAE:"+str(MAE)+" Max:"+str(Max)
infoFile.write(Line)
infoFile.write(str('\n'))

bestscores, fold_string = trainCompensator(args, Regressor, Cross_Val_Regressor, X, Y, directory, fileNames, X_Unseen, Y_Unseen)

Best_Classic_Model = load_model(directory + fileNames["Best_Classic_File"])
Best_Cross_Model = load_model(directory + fileNames["Best_Cross_File"])
Y_Proposed_Classic = Best_Classic_Model.predict(X_Unseen)
Y_Proposed_Cross = Best_Cross_Model.predict(X_Unseen)
Y_Proposed_Cross_full = Best_Cross_Model.predict(X_full)
Y_Proposed_Classic_full = Best_Classic_Model.predict(X_full)
infoFile.write(str(bestscores))
infoFile.write(str('\n'))
infoFile.write(str({'Batch': args["batch"], 'Epochs': args['epochs'], fold_string: args['fold']}))
infoFile.close()
Final_Graphs(Y_Unseen,Y_Proposed_Classic,Y_Proposed_Cross,False,Y_Average,Y_full, directory)
Final_Graphs(Y_full,Y_Proposed_Classic_full,Y_Proposed_Cross_full,True,Y_Average,Y_full, directory)





