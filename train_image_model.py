import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--number", type=int, required=True, help="Number of iterations.")
ap.add_argument("-f", "--front", nargs='?', const=True, type=bool, required=False, default=False, help="Train Front Model.")
ap.add_argument("-s", "--side", nargs='?', const=True, type=bool, required=False, default=False, help="Train Side Model.")
ap.add_argument("-v", "--visualize", nargs='?', const=True, type=bool, required=False, default=False, help="View Training Graphs.")
args = vars(ap.parse_args())

if args["side"] and args["front"]:
    print("Please Specify Either Front or Side Model training, not both.")
    exit()
elif args["side"]:
    file_ = "side.csv"
    Final_Model_File = "Final_Model_Side.h5"
    Low_Score_File = "Lowest_Score_Side.txt"
    Classic_Model_File = 'Classical_Side.h5'
    Cross_Model_File = 'Cross_Side.h5'
    Best_Classical = 'Best_Class_Side.h5'
    Best_Cross = 'Best_Cross_Side.h5'
elif args["front"]:
    file_ = "front.csv"
    Final_Model_File = "Final_Model_Front.h5"
    Low_Score_File = "Lowest_Score_Front.txt"
    Classic_Model_File = 'Classical_Front.h5'
    Cross_Model_File = 'Cross_Front.h5'
    Best_Classical = 'Best_Class_Front.h5'
    Best_Cross = 'Best_Cross_Front.h5'
else:
    print("Please Specify Front or Side Model training.")
    exit()

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

# Model selection
def overallscore(MSE, MAE, Max):
    score = (0.35 * MSE) + (0.35 * MAE) + (0.3 * Max)
    return score

# define base model
def baseline_model():
	# create model
    regressor = Sequential()
    regressor.add(Dense(units=7, input_dim=6, activation="relu"))
    regressor.add(Dense(units=7, activation="relu"))
    regressor.add(Dense(units=1, activation="linear"))
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])
    return regressor

# load dataset inputs
dataframe_traning_inputs = open(file_, 'r')
reader = csv.reader(dataframe_traning_inputs, delimiter=",")
Input_parameters = [[float(entry) for entry in row] for row in reader]
badDataIndex = [index for index, row in enumerate(Input_parameters) if row[1] > 2.5]
Input_parameters = [row for index, row in enumerate(Input_parameters) if index not in badDataIndex]
Input_parameters = np.asarray(Input_parameters)

# load dataset BMI
BMI_file = "BMI.csv"
dataframe_traning_outputs = open(BMI_file, 'r')
reader = csv.reader(dataframe_traning_outputs, delimiter=",")
BMI = [[float(entry) for entry in row] for row in reader]
BMI = [row for index, row in enumerate(BMI) if index not in badDataIndex]
BMI = np.asarray(BMI)
BMI = BMI[:,2]

# split into input (X) and output (Y) variables
X = Input_parameters
Y = BMI
Y = Y.reshape(-1,1)

# Get current best model score
score_file = pandas.read_csv(Low_Score_File, sep=" ", header=None, names=None)
Lowest_score_data = score_file.values
Lowest_score_array = Lowest_score_data[:1]
lowest_score = Lowest_score_array.item(0)

Classic_Model = baseline_model()
Cross_Val_Regressor = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=10, verbose=1)
progress = []

# Initialise Best Scores to be Overwritten
best_class_score = 10000
best_cross_score = 10000

# Separate file data into seen and unseen data prior to model comparison
seed = 10
np.random.seed(seed)
X,X_Unseen,Y,Y_Unseen = train_test_split(X,Y,test_size=0.2)

# finding optimal model
for x in range(args["number"]):
    np.random.seed(x)
    # classic test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    history = Classic_Model.fit(X_train, Y_train, batch_size=5, epochs=500, verbose=1, validation_data=(X_test, Y_test))
    Y_Classic = Classic_Model.predict(X_test)
    Classic_Model.model.save(Classic_Model_File)

    # evaluate model with dataset for Cross Validation
    kfold = KFold(n_splits=10, random_state=x)
    # Optimal estimator extraction
    Cross_Val = cross_validate(Cross_Val_Regressor, X, Y, cv=kfold, return_estimator=True)
    results = Cross_Val["test_score"]
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    estimator_array = Cross_Val['estimator']
    Cross_val_estimator = estimator_array[9]
    Cross_val_estimator.model.save(Cross_Model_File)
    # Assessing optimal model
    Cross_Val_Model = load_model(Cross_Model_File)
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

    Classical_Overall = overallscore(Classical_MSE, Classical_MAE, Classical_Max)
    Cross_Overall = overallscore(Cross_MSE, Cross_MAE, Cross_Max)
    
    if Classical_Overall < Cross_Overall:
        print("Classical")
        if Classical_Overall<lowest_score:
            Classic_Model.model.save(Final_Model_File)
            lowest_score=Classical_Overall
            print("new model "+ str(lowest_score))
            progress.append([lowest_score, 'Classical'])
    else:
        print("Cross")
        if Cross_Overall<lowest_score:
            Cross_Val_Model.model.save(Final_Model_File)
            lowest_score = Cross_Overall
            print("new model "+ str(lowest_score))
            progress.append([lowest_score, 'Cross'])

    if best_class_score < Classical_Overall:
        Classic_Model.model.save(Best_Classical)
        best_class_score = Classical_Overall

    if best_cross_score < Cross_Overall:
        Cross_Val_Model.model.save(Best_Cross)
        best_cross_score = Cross_Overall

    if args["visualize"]:
        # Cross validation model analysis (loss)
        plt.plot(abs(results))
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('Fold')
        plt.legend(['train'], loc='upper left')
        plt.show()

        plt.plot(Y_Unseen)
        plt.plot(Y_Unseen_Classical)
        plt.plot(Y_Unseen_Cross)
        plt.legend(['Actual','Classical','Cross Validation'], loc='upper left')
        plt.show()

        plt.plot(Y_test)
        plt.plot(Y_Classic)
        plt.plot(Y_Cross_Val)
        plt.legend(['Actual','Classical','Cross Validation'], loc='upper left')
        plt.show()

        # loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.show()

        # mean absolute error
        # print(history.history)
        plt.plot(history.history['mean_absolute_error'])
        plt.plot(history.history['val_mean_absolute_error'])
        plt.title('Mean absolute error')
        plt.ylabel('MAE')
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

print(progress)
Proposed_Model = load_model(Final_Model_File)
Y_Proposed = Proposed_Model.predict(X_Unseen)
# plt.plot(Y_Unseen)
# plt.plot(Y_Proposed)
# plt.legend(['Actual', 'Model'], loc='upper left')
# plt.show()
score_file = open(Low_Score_File,"w")
score_file.write(str(lowest_score))
score_file.close()
print(lowest_score)

plt.scatter(Y_Proposed, Y_Unseen)
plt.plot([Y_Proposed.min(), Y_Proposed.max()], [Y_Proposed.min(), Y_Proposed.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.title('Final Model')
plt.show()