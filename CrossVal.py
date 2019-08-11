import numpy as np
from sklearn.model_selection import StratifiedKFold
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import load_model
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
seed = 7
np.random.seed(seed)
#classic test split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
Regressor= baseline_model()
history =Regressor.fit(X_train,Y_train,batch_size=5,epochs=500,verbose=1,validation_data=(X_test,Y_test))
#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
#accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()
Y_predict = Regressor.predict(X_test)
print(Y_predict)
plt.scatter(Y_test, Y_predict)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)
plt.xlabel('Measured')
plt.ylabel('Predicted')
plt.show()
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=10, verbose=1)
Regressor.model.save('Classical.h5')
estimator.model = load_model('Classical.h5')
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
print(results)
#Cross validation model analysis (loss)
plt.plot(abs(results))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('Fold')
plt.legend(['train'], loc='upper left')
plt.show()
#Optimal estimator extraction
from sklearn.model_selection import cross_validate
Cross_Val = cross_validate(estimator, X, Y, cv=kfold,return_estimator=True)
estimator_array = Cross_Val['estimator']
optimal = estimator_array[9]
optimal.model.save('Cross_Validation.h5')
#Assessing optimal model
Final = load_model('Cross_Validation.h5')
Y_Cross_Val = Final.predict(X_test)
plt.plot(Y_test)
plt.plot(Y_predict)
plt.plot(Y_Cross_Val)
plt.legend(['Actual','Classical','Cross Validation'], loc='upper left')
plt.show()
#unseen data tests
unseen = "Unseen.txt"
dataframe_2 = pandas.read_csv(unseen,sep="\t",header=None,names=heading)
# split into input (X) and output (Y) variables
dataset_2=dataframe_2.values
X_Unseen = dataset_2[:,0:2]
Y_Unseen = dataset_2[:,2]
#normalise data
X_Unseen=scale.fit_transform(X_Unseen)
Y_Unseen= Y_Unseen.reshape(-1,1)
Y_Unseen= scale.fit_transform(Y_Unseen)
Y_Unseen_Classical = Regressor.predict(X_Unseen)
Y_Unseen_Cross = Final.predict(X_Unseen)
plt.plot(Y_Unseen)
plt.plot(Y_Unseen_Classical)
plt.plot(Y_Unseen_Cross)
plt.legend(['Actual','Classical','Cross Validation'], loc='upper left')
plt.show()
