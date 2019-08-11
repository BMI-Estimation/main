import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error

# load dataset
file = "Data.txt"
heading = ["Height","Weight","BMI"]
dataframe = pandas.read_csv(file,sep="\t",header=None,names=heading)
# split into input (X) and output (Y) variables
dataset=dataframe.values
X = dataset[:,0:2]
Y = dataset[:,2]
#normalise data
scale = MinMaxScaler()
X=scale.fit_transform(X)
Y= Y.reshape(-1,1)
Y= scale.fit_transform(Y)
#unseen data
unseen = "Unseen.txt"
dataframe_2 = pandas.read_csv(unseen, sep="\t", header=None, names=heading)
# split into input (X) and output (Y) variables
dataset_2 = dataframe_2.values
X_Unseen = dataset_2[:, 0:2]
Y_Unseen = dataset_2[:, 2]
# normalise data
X_Unseen = scale.fit_transform(X_Unseen)
Y_Unseen = Y_Unseen.reshape(-1, 1)
Y_Unseen = scale.fit_transform(Y_Unseen)
# define base model
def baseline_model():
	# create model
    regressor = Sequential()
    regressor.add(Dense(units=20, input_dim=2, activation="relu"))
    regressor.add(Dense(units=4, activation="relu"))
    regressor.add(Dense(units=1, activation="linear"))
    regressor.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse', 'accuracy'])
    return regressor
# fix random seed for reproducibility
Regressor= baseline_model()
lowest_score = 5
progress = []
#finding optimal model
for x in range(2):
    np.random.seed(x)
    # classic test split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    history = Regressor.fit(X_train, Y_train, batch_size=5, epochs=500, verbose=1, validation_data=(X_test, Y_test))
    # loss
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train','test'], loc='upper left')
    # plt.show()
    # accuracy
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train','test'], loc='upper left')
    # plt.show()
    Y_predict = Regressor.predict(X_test)
    # print(Y_predict)
    # plt.scatter(Y_test, Y_predict)
    # plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
    # plt.xlabel('Measured')
    # plt.ylabel('Predicted')
    # plt.show()
    # evaluate model with standardized dataset
    estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=10, verbose=1)
    Regressor.model.save('Classical.h5')
    estimator.model = load_model('Classical.h5')
    kfold = KFold(n_splits=10, random_state=x)
    results = cross_val_score(estimator, X, Y, cv=kfold)
    print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
    # print(results)
    # Cross validation model analysis (loss)
    # plt.plot(abs(results))
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('Fold')
    # plt.legend(['train'], loc='upper left')
    # plt.show()
    # Optimal estimator extraction
    Cross_Val = cross_validate(estimator, X, Y, cv=kfold, return_estimator=True)
    estimator_array = Cross_Val['estimator']
    optimal = estimator_array[9]
    optimal.model.save('Cross_Validation.h5')
    # Assessing optimal model
    Final = load_model('Cross_Validation.h5')
    Y_Cross_Val = Final.predict(X_test)
    # plt.plot(Y_test)
    # plt.plot(Y_predict)
    # plt.plot(Y_Cross_Val)
    # plt.legend(['Actual','Classical','Cross Validation'], loc='upper left')
    # plt.show()
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


    # Model selection
    def overallscore(MSE, MAE, Max):
        score = (0.35 * MSE) + (0.35 * MAE) + (0.3 * Max)
        return score


    Classical_Overall = overallscore(Classical_MSE, Classical_MAE, Classical_Max)
    Cross_Overall = overallscore(Cross_MSE, Cross_MAE, Cross_Max)
    if Classical_Overall < Cross_Overall:
        print("Classical")
        if Classical_Overall<lowest_score:
            Regressor.model.save('Final_Model.h5')
            lowest_score=Classical_Overall
            print("new model "+ str(lowest_score))
            progress.append(lowest_score)
    else:
        print("Cross")
        if Cross_Overall<lowest_score:
            Final.model.save('Final_Model.h5')
            lowest_score = Cross_Overall
            print("new model "+ str(lowest_score))
            progress.append(lowest_score)
print(progress)
