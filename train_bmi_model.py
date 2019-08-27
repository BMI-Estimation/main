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
from trainingFunctions import overallscore, baseline_model, showBMIGraphs, Final_Graphs
from keras import optimizers

# Required files
Front_input_file = "front.csv"
Side_input_file = "side.csv"
Final_BMI_Model = 'Final_Model_BMI.h5'
Low_Score_File = 'Lowest_Score_BMI.txt'
Classic_Model_File = 'Classical_BMI.h5'
Cross_Model_File = 'Cross_BMI.h5'
Best_Classic_File = 'Best_Classic.h5'
Best_Cross_File = 'Best_Cross.h5'
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

#Load front and side models
Front_Model = load_model(Front_Model_file)
Side_Model = load_model(Side_Model_file)

#Get respective predictions
Y_Side = Side_Model.predict(Input_parameters_Side)
print(len(Y_Side))
Y_Front = Front_Model.predict(Input_parameters_Front)
print(len(Y_Front))

diff = Y_Front - Y_Side
diff = [i for i, dif in enumerate(diff) if dif > 5]
Y_Front = [yf for i, yf in enumerate(Y_Front) if i not in diff]
Y_Side = [ys for i, ys in enumerate(Y_Side) if i not in diff]

Y_Front = np.asarray(Y_Front)
Y_Side = np.asarray(Y_Side)

#load dataset BMI
BMI_file = "BMI.csv"
dataframe_traning_outputs = open(BMI_file, 'r')
reader = csv.reader(dataframe_traning_outputs, delimiter=",")
BMI = [[float(entry) for entry in row] for row in reader]
BMI = [row for index, row in enumerate(BMI) if index not in badDataIndex]
BMI = [b for i, b in enumerate(BMI) if i not in diff]
BMI = np.asarray(BMI)
BMI = BMI[:,2]

# # split into input (X) and output (Y) variables
X = np.column_stack((Y_Side,Y_Front))
Y = BMI
Y= Y.reshape(-1,1)
print(X)
print(Y)
#model dimensions
neuronsPerLayerExceptOutputLayer = [1]
save = input("continue")

#Classic and cross model creation
build = baseline_model(2, neuronsPerLayerExceptOutputLayer)
Regressor= build()
Cross_Val_Regressor = KerasRegressor(build_fn=build, epochs=args['epochs'], batch_size=args['batch'], verbose=1)

#Directory functionality
NetworkArc = [str(row.units) for row in Regressor.model.layers]
NetworkArc = '-'.join(NetworkArc)
today = str(datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S"))
directory = 'models/' + today + Path_Extension + NetworkArc + '/'
os.makedirs(directory)

#create infofiles
infoFile = open(directory + 'info.txt', 'w', newline='')

#Separate file data into seen and unseen data prior to model comparison
seed = 10
np.random.seed(seed)
X_full = X.copy()
Y_full = Y.copy()
X,X_Unseen,Y,Y_Unseen = train_test_split(X,Y,test_size=0.2)
#Prediction Averages performance
Y_Average =(Y_Side+Y_Front)/2
MSE = mean_squared_error(Y_full, Y_Average)
MAE = mean_absolute_error(Y_full, Y_Average)
Max = max_error(Y_full, Y_Average)
Line = "MSE:"+str(MSE)+" MAE:"+str(MAE)+" Max:"+str(Max)
infoFile.write(Line)
infoFile.write(str('\n'))
bestscores = {}
best_class_score = 0
best_cross_score = 0
ClassHistory = None
CrossHistory = None

#finding optimal model
for x in range(args['iter']):
	np.random.seed(x)
	# classic test split
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
	history = Regressor.fit(X_train, Y_train, batch_size=args['batch'], epochs=args['epochs'], verbose=1, validation_data=(X_test, Y_test))
	Y_Classic = Regressor.predict(X_test)
	Regressor.model.save(Classic_Model_File)
	if args['strat']:
		kfold = args['fold']
		fold_string ="Sfold"
	else:
		kfold = KFold(n_splits=args['fold'], random_state=x)
		fold_string = "Fold"
	#Cross validation
	# Optimal estimator extraction
	Cross_Val = cross_validate(Cross_Val_Regressor, X, Y, cv=kfold, return_estimator=True)
	estimator_array = Cross_Val['estimator']
	results = Cross_Val['test_score']
	Lowest_Score = np.amin(results)
	Lowest_Score_Index = np.where(results==Lowest_Score)
	Cross_val_estimator = estimator_array[-1]
	# Cross_val_estimator = estimator_array[np.ndarray.item(Lowest_Score_Index[0])]
	Cross_val_estimator.model.save(Cross_Model_File)
	# Assessing optimal model
	Cross_model = load_model(Cross_Model_File)
	Y_Cross_Val = Cross_model.predict(X_test)
	# unseen data tests
	Y_Unseen_Classical = Regressor.predict(X_Unseen)
	Y_Unseen_Cross = Cross_model.predict(X_Unseen)
	# Performance indicators
	Classical_MSE = mean_squared_error(Y_Unseen, Y_Unseen_Classical)
	Cross_MSE = mean_squared_error(Y_Unseen, Y_Unseen_Cross)
	Classical_MAE = mean_absolute_error(Y_Unseen, Y_Unseen_Classical)
	Cross_MAE = mean_absolute_error(Y_Unseen, Y_Unseen_Cross)
	Classical_Max = max_error(Y_Unseen, Y_Unseen_Classical)
	Cross_Max = max_error(Y_Unseen, Y_Unseen_Cross)
	print(str(Classical_MSE) + "\t" + str(Cross_MSE))
	print(str(Classical_MAE) + "\t" + str(Cross_MAE))
	print(str(Classical_Max) + "\t" + str(Cross_Max))
	#save file option
	Classical_Overall = overallscore(Classical_MAE, Classical_Max)
	Cross_Overall = overallscore(Cross_MAE, Cross_Max)
	if Classical_Overall > best_class_score:
		Regressor.model.save(directory + Best_Classic_File)
		bestscores['Classical'] = [Classical_MAE, Classical_Max]
		best_class_score = Classical_Overall
		ClassHistory = history

	if Cross_Overall > best_cross_score:
		Cross_model.model.save(directory + Best_Cross_File)
		bestscores['Cross'] = [Cross_MAE, Cross_Max]
		best_cross_score = Cross_Overall
		CrossHistory = results
	if args["visualize"]:showBMIGraphs(history,Y_Classic,Y_test,Y_Cross_Val,x, directory, results)

Best_Classic_Model = load_model(directory+Best_Classic_File)
Best_Cross_Model = load_model(directory+Best_Cross_File)
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





