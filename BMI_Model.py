import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
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
import argparse
from scipy.interpolate import interp1d
import os
import datetime
from trainingFunctions import baseline_model, overallscore

#Visualisation functions
def showGraphs(history, Y_Classic, Y_Test,records,Y_Cross):
	# loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train','test'], loc='upper left')
	plt.show()

	# Cross validation model analysis (loss)
	plt.plot(results)
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('Fold')
	plt.legend(['train'], loc='upper left')
	plt.show()

	#Classic scatter
	plt.scatter(Y_test, Y_Classic)
	plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Classic Model - Test Data')
	plt.show()

	#Cross scatter
	plt.scatter(Y_test, Y_Cross)
	plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Cross Model - Test Data')
	plt.show()
	return

def Final_Graphs(Y_Unseen,Y_Best_Classic,Y_Best_Cross):
	#Best Classic
	plt.scatter(Y_test, Y_Best_Classic)
	plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Final Model - Classic')
	plt.show()

	# Best Cross
	plt.scatter(Y_Unseen, Y_Best_Cross)
	plt.plot([Y_Unseen.min(), Y_Unseen.max()], [Y_Unseen.min(), Y_Unseen.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Final Model - Cross')
	plt.show()
	return

#argparse
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mass", nargs='?', const=True, type=bool, required=False, default=False, help="Train using mass and height.")
ap.add_argument("-b", "--BMI", nargs='?', const=True, type=bool, required=False, default=False, help="Train using BMI.")
ap.add_argument("-v", "--visualize", nargs='?', const=True, type=bool, required=False, default=False, help="View Training Graphs.")
ap.add_argument("-bs", "--batch", type=int, required=True, help="Batch Size.")
ap.add_argument("-e", "--epochs", type=int, required=True, help="Number of Epochs per training cycle.")
args = vars(ap.parse_args())

# Required files
Front_input_file = "Front_Model.csv"
Side_input_file = "Side_Model.csv"
Final_BMI_Model = 'Final_Model_BMI.h5'
Low_Score_File = 'Lowest_Score_BMI.txt'
Classic_Model_File = 'Classical_BMI.h5'
Cross_Model_File = 'Cross_BMI.h5'
Best_Classic_File = 'Best_Classic.h5'
Best_Cross_File = 'Best_Cross.h5'
if args["mass"]:
	Front_Model_file = 'Mass_Model_Front.h5'
	Side_Model_file = 'Mass_Model_Side.h5'
	Height_Model_file = 'Height_Model.h5'
	Path_Extension = '/Mass/'
elif args["BMI"]:
	Front_Model_file = 'Final_Model_Front.h5'
	Side_Model_file = 'Final_Model_Side_1.h5'
	Path_Extension = '/BMI/'

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

# height related functionality
if args["mass"]:
	Height_Side = Input_parameters_Side[:,1]
	Height_Front = Input_parameters_Front[:,1]
	Height = np.column_stack((Height_Front,Height_Side))
	Height_Model = load_model(Height_Model_file)
	Height_predictions = Height_Model.predict(Height)

# load dataset BMI
BMI_file = "BMI.csv"
dataframe_traning_outputs = open(BMI_file, 'r')
reader = csv.reader(dataframe_traning_outputs, delimiter=",")
BMI = [[float(entry) for entry in row] for row in reader]
BMI = [row for index, row in enumerate(BMI) if index not in badDataIndex]
BMI = np.asarray(BMI)
BMI = BMI[:,2]

# Load front and side models
Front_Model = load_model(Front_Model_file)
Side_Model = load_model(Side_Model_file)

# Get respective predictions
Y_Side = Side_Model.predict(Input_parameters_Side)
Y_Front = Front_Model.predict(Input_parameters_Front)
if args["mass"]:
	Y_Side = Y_Side/(Height*Height)
	Y_Front = Y_Front*(Height*Height)

# split into input (X) and output (Y) variables
X = np.column_stack((Y_Side,Y_Front))
Y = BMI
Y= Y.reshape(-1,1)

# model dimensions
neuronsPerLayerExceptOutputLayer = [4, 3, 2]

# Classic and cross model creation
build = baseline_model(2, neuronsPerLayerExceptOutputLayer)
Regressor= build()
Cross_Val_Regressor = KerasRegressor(build_fn=build, epochs=args['epochs'], batch_size=args['batch'], verbose=1)

# Directory functionality
NetworkArc = [str(row.units) for row in Regressor.model.layers]
NetworkArc = '-'.join(NetworkArc)
today = str(datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S"))
directory = 'models/' + today + '/BMIAverager/' + NetworkArc + '/'
os.makedirs(directory)

# create infofiles
infoFile = open(directory + 'info.txt', 'w', newline='')

# Separate file data into seen and unseen data prior to model comparison
seed = 10
np.random.seed(seed)
X,X_Unseen,Y,Y_Unseen = train_test_split(X,Y,test_size=0.2)

bestscores = {}
best_class_score = 0
best_cross_score = 0
ClassHistory = None
CrossHistory = None

# finding optimal model
for x in range(3):
	np.random.seed(x)
	# classic test split
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
	history = Regressor.fit(X_train, Y_train, batch_size=args['batch'], epochs=args['epochs'], verbose=1, validation_data=(X_test, Y_test))
	Y_Classic = Regressor.predict(X_test)
	Regressor.model.save(Classic_Model_File)
	kfold = KFold(n_splits=10, random_state=x)
	#Cross validation
	# Optimal estimator extraction
	Cross_Val = cross_validate(Cross_Val_Regressor, X, Y, cv=kfold, return_estimator=True)
	estimator_array = Cross_Val['estimator']
	results = Cross_Val['test_score']
	Cross_val_estimator = estimator_array[-1]
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
	save = input("Save model?")
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
			
	if args["visualize"]:showGraphs(history,Y_Classic,Y_Cross_Val)

Best_Classic_Model = load_model(directory+Best_Classic_File)
Best_Cross_Model = load_model(directory+Best_Cross_File)
Y_Proposed_Classic = Best_Classic_Model.predict(X_Unseen)
Y_Proposed_Cross = Best_Cross_Model.predict(X_Unseen)
infoFile.write(str(bestscores))
infoFile.write(str('\n'))
infoFile.write(str({'Batch': args["batch"], 'Epochs': args['epochs'], 'Folds': kfold.get_n_splits()}))
infoFile.close()
Final_Graphs(Y_Unseen,Y_Proposed_Classic,Y_Proposed_Cross)
