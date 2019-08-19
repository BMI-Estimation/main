import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold, train_test_split, cross_validate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
from scipy.interpolate import interp1d
import pandas
import os
import datetime

# define base model
def baseline_model(inputDim, neuronsPerLayerExceptOutputLayer):
	# create model
	def build_fn():
		regressor = Sequential()
		regressor.add(Dense(neuronsPerLayerExceptOutputLayer[0], input_dim=inputDim, activation="relu"))

		for units in neuronsPerLayerExceptOutputLayer[1:]:
			regressor.add(Dense(units, activation="relu"))
	
		regressor.add(Dense(1, activation="linear"))
		regressor.compile(optimizer='adam', loss='mean_absolute_error')
		return regressor
	
	return build_fn

def showGraphs(crossValTestResults, Y_Unseen, Y_Unseen_Classical, Y_Unseen_Cross, Y_test, Y_Classic, Y_Cross_Val, history):
	# Cross validation model analysis (loss)
	plt.plot(abs(crossValTestResults))
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('Fold')
	plt.legend(['train'], loc='upper left')
	plt.show()

	plt.plot(Y_Unseen)
	plt.plot(Y_Unseen_Classical)
	plt.plot(Y_Unseen_Cross)
	plt.legend(['Actual','Classical','Cross Validation'], loc='upper left')
	plt.title('Unseen Data Performance')
	plt.show()

	plt.plot(Y_test)
	plt.plot(Y_Classic)
	plt.plot(Y_Cross_Val)
	plt.legend(['Actual','Classical','Cross Validation'], loc='upper left')
	plt.title('Test Data Performance')
	plt.show()

	# loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train','test'], loc='upper left')
	plt.show()

	plt.scatter(Y_test, Y_Classic)
	plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Classic method prediction')
	plt.show()

	plt.scatter(Y_test, Y_Cross_Val)
	plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Cross val method prediction')
	plt.show()
	return

# Model selection
def overallscore(MAE, Max):
	if MAE < 4:
		m = interp1d([0,4],[4,0])
		return MAE/Max + m(MAE)
	else: return MAE/Max

def train(X, Y, args, CM, CVR , fileNames, infoFile):
	Classic_Model = CM
	Cross_Val_Regressor = CVR
	progress = []
	best_scores = {}
	ClassHistory = None
	CrossHistory = None
	# Initialise Best Scores to be Overwritten
	best_class_score = 0
	best_cross_score = 0
	Y = Y.reshape(-1,1)
	# Separate file data into seen and unseen data prior to model comparison
	seed = 10
	np.random.seed(seed)
	X,X_Unseen,Y,Y_Unseen = train_test_split(X,Y,test_size=0.2)

	# finding optimal model
	for x in range(args["number"]):
		# classic test split
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
		history = Classic_Model.fit(X_train, Y_train, batch_size=args['batch'], epochs=args['epochs'], verbose=1, validation_data=(X_test, Y_test))
		Y_Classic = Classic_Model.predict(X_test)
		Classic_Model.model.save(fileNames["Classic_Model_File"])

		# unseen data tests
		Y_Unseen_Classical = Classic_Model.predict(X_Unseen)

		# Performance indicators
		Classical_MSE = mean_squared_error(Y_Unseen, Y_Unseen_Classical)
		Classical_MAE = mean_absolute_error(Y_Unseen, Y_Unseen_Classical)
		Classical_Max = max_error(Y_Unseen, Y_Unseen_Classical)
		print('Classical MSE', str(Classical_MSE))
		print('Classical MAE', str(Classical_MAE))
		print('Classical Max', str(Classical_Max))

		Classical_Overall = overallscore(Classical_MAE, Classical_Max)

		if Classical_Overall > best_class_score:
			Classic_Model.model.save(fileNames["directory"] + fileNames["Best_Classical"])
			best_scores['Classical'] = [Classical_MAE, Classical_Max]
			best_class_score = Classical_Overall
			ClassHistory = history

		infoFile.write(str(best_scores))
		infoFile.write(str('\n'))
		infoFile.write(str({'Batch': args["batch"], 'Epochs': args['epochs']}))

	# Results
	Proposed_Classic_Model = load_model(fileNames["directory"] + fileNames["Best_Classical"])
	Y_Proposed_Class = Proposed_Classic_Model.predict(X_Unseen)
	Y_Classic = Proposed_Classic_Model.predict(X_test)
	plt.scatter(Y_Proposed_Class, Y_Unseen)
	plt.plot([Y_Proposed_Class.min(), Y_Proposed_Class.max()], [Y_Proposed_Class.min(), Y_Proposed_Class.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Final Classic Model')
	plt.show()

	for x in range(args["number"]):
		np.random.seed(x)
		# evaluate model with dataset for Cross Validation
		kfold = KFold(n_splits=10, random_state=x)
		# Optimal estimator extraction
		Cross_Val = cross_validate(Cross_Val_Regressor, X, Y, cv=kfold, return_estimator=True, scoring='neg_mean_absolute_error')
		results = Cross_Val["test_score"]
		estimator_array = Cross_Val['estimator']
		Cross_val_estimator = estimator_array[-1]
		Cross_val_estimator.model.save(fileNames["Cross_Model_File"])
		# Assessing optimal model
		Cross_Val_Model = load_model(fileNames["Cross_Model_File"])
		Y_Cross_Val = Cross_Val_Model.predict(X_test)
		
		Y_Unseen_Cross = Cross_Val_Model.predict(X_Unseen)
		Cross_MSE = mean_squared_error(Y_Unseen, Y_Unseen_Cross)
		Cross_MAE = mean_absolute_error(Y_Unseen, Y_Unseen_Cross)
		Cross_Max = max_error(Y_Unseen, Y_Unseen_Cross)

		print('Cross MSE', str(Cross_MSE))
		print('Cross MAE', str(Cross_MAE))
		print('Cross Max', str(Cross_Max))

		Cross_Overall = overallscore(Cross_MAE, Cross_Max)
		if Cross_Overall > best_cross_score:
			Cross_Val_Model.model.save(fileNames["directory"] + fileNames["Best_Cross"])
			best_scores['Cross'] = [Cross_MAE, Cross_Max]
			best_cross_score = Cross_Overall
			CrossHistory = results

	# Results
	Proposed_Cross_Model = load_model(fileNames["directory"] + fileNames["Best_Cross"])
	Y_Proposed_Cross = Proposed_Cross_Model.predict(X_Unseen)
	Y_Cross_Val = Proposed_Cross_Model.predict(X_test)

	plt.scatter(Y_Proposed_Cross, Y_Unseen)
	plt.plot([Y_Proposed_Cross.min(), Y_Proposed_Cross.max()], [Y_Proposed_Cross.min(), Y_Proposed_Cross.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Final Cross Model')
	plt.show()

	infoFile.close()
	print('Progress', progress)

	if args["visualize"]: showGraphs(CrossHistory, Y_Unseen, Y_Proposed_Class, Y_Proposed_Cross, Y_test, Y_Classic, Y_Cross_Val, ClassHistory)

	return

def trainWithBMI(X, Y, args, fileNames):
	# Initialise Models and Folder Structure
	inputDim = 6
	neuronsPerLayerExceptOutputLayer = [7, 4]
	build = baseline_model(inputDim, neuronsPerLayerExceptOutputLayer)
	Classic_Model = build()
	Cross_Val_Regressor = KerasRegressor(build_fn=build, epochs=args['epochs'], batch_size=args['batch'], verbose=1)

	# Created Directory
	NetworkArc = [str(row.units) for row in Classic_Model.model.layers]
	NetworkArc = '-'.join(NetworkArc)
	today = str(datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S"))
	fileNames["directory"] = 'models/' + today + '/BMI/' + NetworkArc + '/'
	os.makedirs(fileNames["directory"])

	infoFile = open(fileNames["directory"] + 'BMI-info.txt', 'w', newline='')
	train(X, Y, args, Classic_Model, Cross_Val_Regressor, fileNames, infoFile)
	return

def trainHeight(X, Y, args, fileNames):
	print("[INFO] Training Against Height")
	# Initialise Models and Folder Structure
	inputDim = 2
	neuronsPerLayerExceptOutputLayer = [3]
	build = baseline_model(inputDim, neuronsPerLayerExceptOutputLayer)
	Classic_Model = build()
	Cross_Val_Regressor = KerasRegressor(build_fn=build, epochs=args['epochs'], batch_size=args['batch'], verbose=1)

	# Created Directory
	NetworkArc = [str(row.units) for row in Classic_Model.model.layers]
	NetworkArc = '-'.join(NetworkArc)
	today = str(datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S"))
	fileNames["directory"] = 'models/' + today + '/Height/' + NetworkArc + '/'
	os.makedirs(fileNames["directory"])

	# Initialise Models and Folder Structure
	infoFile = open(fileNames["directory"] + 'Height-info.txt', 'w', newline='')
	train(X, Y, args, Classic_Model, Cross_Val_Regressor, fileNames, infoFile)
	return

def trainMass(X, Y, args, fileNames):
	print("[INFO] Training Against Mass")
	# Initialise Models and Folder Structure
	inputDim = 6
	neuronsPerLayerExceptOutputLayer = [15, 14, 13, 12, 11, 10, 9]
	build = baseline_model(inputDim, neuronsPerLayerExceptOutputLayer)
	Classic_Model = build()
	Cross_Val_Regressor = KerasRegressor(build_fn=build, epochs=args['epochs'], batch_size=args['batch'], verbose=1)

	# Created Directory
	NetworkArc = [str(row.units) for row in Classic_Model.model.layers]
	NetworkArc = '-'.join(NetworkArc)
	today = str(datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S"))
	fileNames["directory"] = 'models/' + today + '/Mass/' + NetworkArc + '/'
	os.makedirs(fileNames["directory"])

	infoFile = open(fileNames["directory"] + 'Mass-info.txt', 'w', newline='')
	train(X, Y, args, Classic_Model, Cross_Val_Regressor, fileNames, infoFile)
	return