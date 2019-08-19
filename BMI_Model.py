import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
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
#argparse
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mass", nargs='?', const=True, type=bool, required=False, default=False, help="Train using mass and height.")
ap.add_argument("-b", "--BMI", nargs='?', const=True, type=bool, required=False, default=False, help="Train using BMI.")
args = vars(ap.parse_args())
# Required files
Front_input_file = "Front_Model.csv"
Side_input_file = "Side_Model.csv"
# Front_Model_file = 'Final_Model_Front.h5'
# Side_Model_file = 'Final_Model_Side_1.h5'
Final_BMI_Model = 'Final_Model_BMI.h5'
Low_Score_File = 'Lowest_Score_BMI.txt'
Classic_Model_File = 'Classical_BMI.h5'
Cross_Model_File = 'Cross_BMI.h5'
if args["mass"]:
    Front_Model_file = 'Mass_Model_Front.h5'
    Side_Model_file = 'Mass_Model_Side.h5'
    Height_Model_file = 'Height_Model.h5'
    train_mass = True
elif args["BMI"]:
    Front_Model_file = 'Final_Model_Front.h5'
    Side_Model_file = 'Final_Model_Side_1.h5'
    train_mass = False

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
print(BMI)
print('F', Input_parameters_Front)
print('S', Input_parameters_Side)
print('FL', len(Input_parameters_Front), 'SL', len(Input_parameters_Side), 'BL', len(BMI))
#Load front and side models
Front_Model = load_model(Front_Model_file)
Side_Model = load_model(Side_Model_file)
#Get respective predictions
Y_Side = Side_Model.predict(Input_parameters_Side)
Y_Front = Front_Model.predict(Input_parameters_Front)
if train_mass==True:
    Y_Side = Y_Side/(Height*Height)
    Y_Front = Y_Front*(Height*Height)
# split into input (X) and output (Y) variables
X = np.column_stack((Y_Side,Y_Front))
Y = BMI
#print(X)
Y= Y.reshape(-1,1)
#print(Y)
# define base model
def baseline_model():
	# create model
    regressor = Sequential()
    regressor.add(Dense(units=10, input_dim=2, activation="relu"))
    regressor.add(Dense(units=6, activation="relu"))
    regressor.add(Dense(units=4, activation="relu"))
    regressor.add(Dense(units=1, activation="linear"))
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    return regressor
#Get current best model score
score_file = pandas.read_csv(Low_Score_File,sep=" ",header=None,names=None)
Lowest_score_data = score_file.values
Lowest_score_array = Lowest_score_data[:1]
lowest_score = Lowest_score_array.item(0)
Regressor= baseline_model()
Cross_Val_Regressor = KerasRegressor(build_fn=baseline_model, epochs=500, batch_size=5, verbose=1)
progress = []
#Separate file data into seen and unseen data prior to model comparison
seed = 10
np.random.seed(seed)
X,X_Unseen,Y,Y_Unseen = train_test_split(X,Y,test_size=0.2)
# Model selection
def overallscore(MAE, Max):
	if MAE < 4:
		m = interp1d([0,4],[4,0])
		return MAE/Max + m(MAE)
	else: return MAE/Max
#finding optimal model
for x in range(3):
    np.random.seed(x)
    # classic test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    history = Regressor.fit(X_train, Y_train, batch_size=5, epochs=500, verbose=1, validation_data=(X_test, Y_test))
    # loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    # mean absolute error
    plt.plot(history.history['mean_absolute_error'])
    plt.plot(history.history['val_mean_absolute_error'])
    plt.title('Mean absolute error')
    plt.ylabel('MAE')
    plt.xlabel('epoch')
    plt.legend(['train','test'], loc='upper left')
    plt.show()
    Y_Classic = Regressor.predict(X_test)
    print(Y_Classic)
    plt.scatter(Y_test, Y_Classic)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title('Classic method prediction')
    plt.show()
    #evaluate model with dataset
    # estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=10, verbose=1)
    Regressor.model.save(Classic_Model_File)
    # estimator.model = load_model(Classic_Model_File)
    kfold = KFold(n_splits=10, random_state=x)
    # results = cross_val_score(estimator, X, Y, cv=kfold)
    # print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    # print(results)
    # #Cross validation model analysis (loss)
    # plt.plot(abs(results))
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('Fold')
    # plt.legend(['train'], loc='upper left')
    # plt.show()
    # Optimal estimator extraction
    Cross_Val = cross_validate(Cross_Val_Regressor, X, Y, cv=kfold, return_estimator=True)
    estimator_array = Cross_Val['estimator']
    Cross_val_estimator = estimator_array[9]
    Cross_val_estimator.model.save(Cross_Model_File)
    # Assessing optimal model
    Final = load_model(Cross_Model_File)
    Y_Cross_Val = Final.predict(X_test)
    #scatter cross val
    plt.scatter(Y_test, Y_Cross_Val)
    plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.title('Cross method prediction')
    plt.show()
    plt.plot(Y_test)
    plt.plot(Y_Classic)
    plt.plot(Y_Cross_Val)
    plt.legend(['Actual','Classical','Cross Validation'], loc='upper left')
    plt.show()
    # unseen data tests
    Y_Unseen_Classical = Regressor.predict(X_Unseen)
    Y_Unseen_Cross = Final.predict(X_Unseen)
    plt.plot(Y_Unseen)
    plt.plot(Y_Unseen_Classical)
    plt.plot(Y_Unseen_Cross)
    plt.legend(['Actual','Classical','Cross Validation'], loc='upper left')
    plt.show()
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
    if Classical_Overall < Cross_Overall:
        print("Classical")
        if Classical_Overall<lowest_score:
            Regressor.model.save(Final_BMI_Model)
            lowest_score=Classical_Overall
            print("new model "+ str(lowest_score))
            progress.append(lowest_score)
    else:
        print("Cross")
        if Cross_Overall<lowest_score:
            Final.model.save(Final_BMI_Model)
            lowest_score = Cross_Overall
            print("new model "+ str(lowest_score))
            progress.append(lowest_score)
print(progress)
Proposed_Model = load_model(Final_BMI_Model)
Y_Proposed = Proposed_Model.predict(X_Unseen)
plt.plot(Y_Unseen)
plt.plot(Y_Proposed)
plt.legend(['Actual', 'Model'], loc='upper left')
plt.show()
plt.scatter(Y_Unseen, Y_Proposed)
plt.plot([Y_Unseen.min(), Y_Unseen.max()], [Y_Unseen.min(), Y_Unseen.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Final method prediction')
plt.show()
score_file = open(Low_Score_File,"w")
score_file.write(str(lowest_score))
score_file.close()
print(lowest_score)
