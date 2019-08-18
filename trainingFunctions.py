import numpy as np
from keras.models import load_model
from sklearn.model_selection import KFold, train_test_split, cross_validate
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error
from scipy.interpolate import interp1d
import pandas

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

def trainWithBMI(X, Y, args, Classic_Model, Cross_Val_Regressor, fileNames):
	infoFile = open(fileNames["directory"] + 'info.txt', 'w', newline='')
	progress = []
	best_scores = {}
	# Initialise Best Scores to be Overwritten
	best_class_score = 0
	best_cross_score = 0
	# Get current best model score
	score_file = pandas.read_csv(fileNames["Low_Score_File"], sep=" ", header=None, names=None)
	Lowest_score_data = score_file.values
	Lowest_score_array = Lowest_score_data[:1]
	lowest_score = Lowest_score_array.item(0)
	Y = Y.reshape(-1,1)
		# Separate file data into seen and unseen data prior to model comparison
	seed = 10
	np.random.seed(seed)
	X,X_Unseen,Y,Y_Unseen = train_test_split(X,Y,test_size=0.2)

	# finding optimal model
	for x in range(args["number"]):
		np.random.seed(x)
		# classic test split
		X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
		history = Classic_Model.fit(X_train, Y_train, batch_size=args['batch'], epochs=args['epochs'], verbose=1, validation_data=(X_test, Y_test))
		Y_Classic = Classic_Model.predict(X_test)
		Classic_Model.model.save(fileNames["Classic_Model_File"])

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

		# unseen data tests
		Y_Unseen_Classical = Classic_Model.predict(X_Unseen)
		Y_Unseen_Cross = Cross_Val_Model.predict(X_Unseen)

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

		Classical_Overall = overallscore(Classical_MAE, Classical_Max)
		Cross_Overall = overallscore(Cross_MAE, Cross_Max)
				
		if Classical_Overall > Cross_Overall:
				print("Classical")
				if Classical_Overall > lowest_score:
						Classic_Model.model.save(fileNames["Final_Model_File"])
						lowest_score=Classical_Overall
						print("new model "+ str(lowest_score))
						progress.append([lowest_score, 'Classical'])
		else:
				print("Cross")
				if Cross_Overall > lowest_score:
						Cross_Val_Model.model.save(fileNames["Final_Model_File"])
						lowest_score = Cross_Overall
						print("new model "+ str(lowest_score))
						progress.append([lowest_score, 'Cross'])

		if Classical_Overall > best_class_score:
				Classic_Model.model.save(fileNames["directory"] + fileNames["Best_Classical"])
				best_scores['Classical'] = [Classical_MAE, Classical_Max]
				best_class_score = Classical_Overall

		if Cross_Overall > best_cross_score:
				Cross_Val_Model.model.save(fileNames["directory"] + fileNames["Best_Cross"])
				best_scores['Cross'] = [Cross_MAE, Cross_Max]
				best_cross_score = Cross_Overall

		infoFile.write(str(best_scores))
		infoFile.write(str('\n'))
		infoFile.write(str({'Batch': args["batch"], 'Epochs': args['epochs']}))

		if args["visualize"]: showGraphs(results, Y_Unseen, Y_Unseen_Classical, Y_Unseen_Cross, Y_test, Y_Classic, Y_Cross_Val, history)

	print('Progress', progress)
	Proposed_Model = load_model(fileNames["Final_Model_File"])
	Y_Proposed = Proposed_Model.predict(X_Unseen)
	# plt.plot(Y_Unseen)
	# plt.plot(Y_Proposed)
	# plt.legend(['Actual', 'Model'], loc='upper left')
	# plt.show()
	score_file = open(fileNames["Low_Score_File"],"w")
	score_file.write(str(lowest_score))
	score_file.close()
	print(lowest_score)

	plt.scatter(Y_Proposed, Y_Unseen)
	plt.plot([Y_Proposed.min(), Y_Proposed.max()], [Y_Proposed.min(), Y_Proposed.max()], 'k--', lw=4)
	plt.xlabel('Measured')
	plt.ylabel('Predicted')
	plt.title('Final Model')
	plt.show()
	return

def trainHeight(X, Y, args, Classic_Model, Cross_Val_Regressor, fileNames):

	return

def trainMass(X, Y, args, Classic_Model, Cross_Val_Regressor, fileNames):
	
	return

def trainWithMassAndHeight(X, Y, args, Classic_Model, Cross_Val_Regressor, fileNames):
	print("[INFO} Training Against Mass")
	trainMass(X, Y[0], args, Classic_Model, Cross_Val_Regressor, fileNames)
	print("[INFO} Training Against Heights")
	trainHeight(X, Y[1], args, Classic_Model, Cross_Val_Regressor, fileNames)
	return