import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
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
#argparse
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mass", nargs='?', const=True, type=bool, required=False, default=False, help="Train using mass and height.")
ap.add_argument("-b", "--BMI", nargs='?', const=True, type=bool, required=False, default=False, help="Train using BMI.")
ap.add_argument("-v", "--visualize", nargs='?', const=True, type=bool, required=False, default=False, help="View Training Graphs.")
ap.add_argument("-bs", "--batch", type=int, required=True, help="Batch Size.")
ap.add_argument("-e", "--epochs", type=int, required=True, help="Number of Epochs per training cycle.")
ap.add_argument("-n", "--iter", type=int, required=True, help="Iterations.")
ap.add_argument("-f", "--fold", type=int, required=True, help="Folds.")
ap.add_argument("-s", "--strat", nargs='?', const=True, type=bool, required=False, default=False, help="Stratified.")
args = vars(ap.parse_args())

# Required files
Front_input_file = "front.csv"
Side_input_file = "side.csv"
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
	train_mass = True
	Path_Extension = '/Mass/'
elif args["BMI"]:
	Front_Model_file = 'Front.h5'
	Side_Model_file = 'Side.h5'
	train_mass = False
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

#height related functionality
if train_mass==True:
	Height_Side = Input_parameters_Side[:,1]
	Height_Front = Input_parameters_Front[:,1]
	Height = np.column_stack((Height_Front,Height_Side))
	Height_Model = load_model(Height_Model_file)
	Height_predictions = Height_Model.predict(Height)


#load dataset BMI
BMI_file = "BMI.csv"
dataframe_traning_outputs = open(BMI_file, 'r')
reader = csv.reader(dataframe_traning_outputs, delimiter=",")
BMI = [[float(entry) for entry in row] for row in reader]
BMI = [row for index, row in enumerate(BMI) if index not in badDataIndex]
BMI = np.asarray(BMI)
BMI = BMI[:,2]

#Load front and side models
Front_Model = load_model(Front_Model_file)
Side_Model = load_model(Side_Model_file)

#Get respective predictions
Y_Side = Side_Model.predict(Input_parameters_Side)
print(len(Y_Side))
Y_Front = Front_Model.predict(Input_parameters_Front)
print(len(Y_Front))
# if train_mass==True:
#     Y_Side = Y_Side/(Height*Height)
#     Y_Front = Y_Front*(Height*Height)
#
# # split into input (X) and output (Y) variables
X = np.column_stack((Y_Side,Y_Front))
Y = BMI
Y= Y.reshape(-1,1)
print(X)
print(Y)
#model dimensions
neuronsPerLayerExceptOutputLayer = [2]
save = input("continue")
# define base model
def baseline_model():
	# create model
	regressor = Sequential()
	regressor.add(Dense(neuronsPerLayerExceptOutputLayer[0], input_dim=2, activation="relu"))
	for units in neuronsPerLayerExceptOutputLayer[1:]:
		regressor.add(Dense(units, activation="relu"))
	regressor.add(Dense(1, activation="linear"))
	regressor.compile(optimizer='adam', loss='mean_absolute_error')
	return regressor

#Classic and cross model creation
Regressor= baseline_model()
Cross_Val_Regressor = KerasRegressor(build_fn=baseline_model, epochs=args['epochs'], batch_size=args['batch'], verbose=1)

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
# score initialisation
def overallscore(MAE, Max):
	if MAE < 4:
		m = interp1d([0,4],[4,0])
		return MAE/Max + m(MAE)
	else: return MAE/Max
bestscores = {}
best_class_score = 0
best_cross_score = 0
ClassHistory = None
CrossHistory = None

#Visualisation functions
def showGraphs( history, Y_Classic, Y_Test,Y_Cross,x):
	# loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train','test'], loc='upper left')
	fig = plt.gcf()
	fig.savefig(directory+'loss_run_'+str(x)+'.png')
	plt.show()
	plt.clf()

	# Cross validation model analysis (loss)
	plt.plot(results)
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('Fold')
	plt.legend(['train'], loc='upper left')
	fig = plt.gcf()
	fig.savefig(directory+'fold_run_'+str(x)+'.png')
	plt.show()
	plt.clf()

	#Classic scatter
	plt.scatter(Y_Test, Y_Classic)
	plt.plot([Y_Test.min(), Y_Test.max()], [Y_Test.min(), Y_Test.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Classic Model - Test Data')
	fig = plt.gcf()
	fig.savefig(directory+'classic_run_'+str(x)+'.png')
	plt.show()
	plt.clf()

	#Cross scatter
	plt.scatter(Y_Test, Y_Cross)
	plt.plot([Y_Test.min(), Y_Test.max()], [Y_Test.min(), Y_Test.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Cross Model - Test Data')
	fig = plt.gcf()
	fig.savefig(directory+'cross_run_'+str(x)+'.png')
	plt.show()
	plt.clf()
	return



def Final_Graphs(Y_Unseen,Y_Best_Classic,Y_Best_Cross,Full_dataset,Y_Average,Y_full):

	if (Full_dataset == True):
		Classic_Path = directory + 'best_classic_full.png'
		Cross_Path = directory + 'best_cross_full.png'
	else:
		Classic_Path = directory + 'best_classic.png'
		Cross_Path = directory + 'best_cross.png'

	#Best Classic
	plt.scatter(Y_Unseen,Y_Best_Classic)
	plt.plot([Y_Unseen.min(), Y_Unseen.max()], [Y_Unseen.min(), Y_Unseen.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Final Model - Classic')
	fig = plt.gcf()
	fig.savefig(Classic_Path)
	plt.show()
	plt.clf()
	# Best Cross
	plt.scatter(Y_Unseen, Y_Best_Cross)
	plt.plot([Y_Unseen.min(), Y_Unseen.max()], [Y_Unseen.min(), Y_Unseen.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Final Model - Cross')
	fig = plt.gcf()
	fig.savefig(Cross_Path)
	plt.show()
	plt.clf()
	#Average
	if (Full_dataset==True):
		plt.scatter(Y_full, Y_Average)
		plt.plot([Y_full.min(), Y_full.max()], [Y_full.min(), Y_full.max()], 'k--', lw=4)
		plt.xlabel('Measured')
		plt.ylabel('Predicted')
		plt.title('Average')
		fig = plt.gcf()
		fig.savefig(directory + 'Average.png')
		plt.show()
		plt.clf()
	return

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
	Cross_val_estimator = estimator_array[np.ndarray.item(Lowest_Score_Index[0])]
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
	if args["visualize"]:showGraphs(history,Y_Classic,Y_test,Y_Cross_Val,x)
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
Final_Graphs(Y_Unseen,Y_Proposed_Classic,Y_Proposed_Cross,False,Y_Average,Y_full)
Final_Graphs(Y_full,Y_Proposed_Classic_full,Y_Proposed_Cross_full,True,Y_Average,Y_full)





