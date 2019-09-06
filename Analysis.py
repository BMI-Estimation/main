import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import csv
import os
import datetime

#Required files
Front_input_file = "front.csv"
Side_input_file = "side.csv"
Front_Model_file = 'Front.h5'
Side_Model_file = 'Side.h5'
BMI_Model_File = 'BMI.h5'
BMI_file = "BMI.csv"
#load datasets
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
Y_Front = Front_Model.predict(Input_parameters_Front)
Y_Side = Side_Model.predict(Input_parameters_Side)
diff = Y_Front - Y_Side
diff = [i for i, dif in enumerate(diff) if dif > 10]
Y_Front = [yf for i, yf in enumerate(Y_Front) if i not in diff]
Y_Side = [ys for i, ys in enumerate(Y_Side) if i not in diff]

Y_Front = np.asarray(Y_Front)
Y_Side = np.asarray(Y_Side)
#Load_Compensator
BMI_Model = load_model(BMI_Model_File)
inputs = np.column_stack((Y_Side,Y_Front))
Y_Comp = BMI_Model.predict(inputs)
#load dataset BMI
BMI_file = "BMI.csv"
dataframe_traning_outputs = open(BMI_file, 'r')
reader = csv.reader(dataframe_traning_outputs, delimiter=",")
BMI = [[float(entry) for entry in row] for row in reader]
BMI = [row for index, row in enumerate(BMI) if index not in badDataIndex]
BMI = [b for i, b in enumerate(BMI) if i not in diff]
BMI = np.asarray(BMI)
#BMI = BMI[:,2]
#Accuracy function
def GetAccuracy(Actual,Prediction,length):
	Pred_list = []
	for x in range(length):
		Pred_list.append(Prediction[x][0])
	Pred_list = np.asarray(Pred_list)
	diff = np.abs((Actual - Pred_list))
	diff = (diff / Actual) * 100
	diff = diff[(diff <= 10000)]
	avg = np.mean(diff)
	acc = 100 - avg
	print(acc)
	return acc
def MakeBar(Values,Categories,Model,directory,Real):
	# this is for plotting purpose
	index = np.arange(len(Categories))
	plt.bar(index, Values)
	plt.xlabel('Category', fontsize=7)
	plt.ylabel('Percentage Accuracy', fontsize=10)
	plt.xticks(index, Categories, fontsize=7, rotation=30)
	if Real==True:
		plt.title('Participant BMI Overview')
		plt.ylabel('Number of Participants', fontsize=10)
	else:
		plt.title('BMI Estimation Accuracy for ' + Model + ' Model')
		plt.yticks(np.arange(0, 100, 10))
		plt.ylabel('Percentage Accuracy', fontsize=10)
	fig = plt.gcf()
	fig.savefig(directory+Model+'.png')
	plt.show()
	plt.clf()
def ErrorAnalysis(Actual,Prediction,length,Dataset,Model):
	Pred_list = []
	for x in range(length):
		Pred_list.append(Prediction[x][0])
	Pred_list = np.asarray(Pred_list)
	diff = np.abs((Actual - Pred_list))
	diff = (diff / Actual) * 100
	diff = diff[(diff <= 10000)]
	#create and save histogram
	plt.hist(diff,bins=10)
	plt.title('BMI Estimation Error for ' + Model + ' Model on '+Dataset+' Participants')
	plt.ylabel('Number of participants', fontsize=10)
	plt.xlabel('Percentage Error', fontsize=7)
	fig = plt.gcf()
	fig.savefig(directory+Model+'_'+Dataset+'_'+'Hist.png')
	plt.show()
	plt.clf()
def TotalError(Actual,Prediction,Model):
	diff = np.abs((Actual - Prediction))
	diff = (diff / Actual) * 100
	print(diff[0:50])
	diff = diff[(diff <= 10000)]
	# print(len(diff))
	# print(diff[0:10])
	#create and save histogram
	plt.hist(diff,bins=15)
	plt.title('BMI Estimation Error for ' + Model + ' Model on All Participants')
	plt.ylabel('Number of participants', fontsize=10)
	plt.xlabel('Percentage Error', fontsize=7)
	fig = plt.gcf()
	fig.savefig(directory+Model+'_Total_'+'Hist.png')
	plt.show()
	plt.clf()
#Underweight check
Underweight = BMI.copy()
Underweight=[[i,row[2]]for i,row in enumerate(Underweight) if row[2]<18.5]
Underweight = np.asarray(Underweight)
Underweight_Values = Underweight[:,1]
Underweight_Index = Underweight[:,0]
Underweight_Prediction_S = [row for i,row in enumerate(Y_Side) if i in Underweight_Index]
Underweight_Prediction_F = [row for i,row in enumerate(Y_Front) if i in Underweight_Index]
Underweight_Prediction_C = [row for i,row in enumerate(Y_Comp) if i in Underweight_Index]

#Healthy weight
Healthyweight = BMI.copy()
Healthyweight = [[i,row[2]]for i,row in enumerate(Healthyweight) if (row[2]>18.5 and row[2]<=25)]
Healthyweight = np.asarray(Healthyweight)
Healthyweight_Values = Healthyweight[:,1]
Healthyweight_Index = Healthyweight[:,0]
Healthyweight_Prediction_S = [row for i,row in enumerate(Y_Side) if i in Healthyweight_Index]
Healthyweight_Prediction_F = [row for i,row in enumerate(Y_Front) if i in Healthyweight_Index]
Healthyweight_Prediction_C = [row for i,row in enumerate(Y_Comp) if i in Healthyweight_Index]

#Overweight
Overweight = BMI.copy()
Overweight = [[i,row[2]]for i,row in enumerate(Overweight) if (row[2]>25 and row[2]<=30)]
Overweight = np.asarray(Overweight)
Overweight_Values = Overweight[:,1]
Overweight_Index = Overweight[:,0]
Overweight_Prediction_S = [row for i,row in enumerate(Y_Side) if i in Overweight_Index]
Overweight_Prediction_F = [row for i,row in enumerate(Y_Front) if i in Overweight_Index]
Overweight_Prediction_C = [row for i,row in enumerate(Y_Comp) if i in Overweight_Index]

#Obese
Obese = BMI.copy()
Obese = [[i,row[2]]for i,row in enumerate(Obese) if (row[2]>30)]
Obese = np.asarray(Obese)
Obese_Values = Obese[:,1]
Obese_Index = Obese[:,0]
Obese_Prediction_S = [row for i,row in enumerate(Y_Side) if i in Obese_Index]
Obese_Prediction_F = [row for i,row in enumerate(Y_Front) if i in Obese_Index]
Obese_Prediction_C = [row for i,row in enumerate(Y_Comp) if i in Obese_Index]

#GetResults
Accuracy_Side = []
Accuracy_Side.append(GetAccuracy(Underweight_Values,Underweight_Prediction_S,len(Underweight)))
Accuracy_Side.append(GetAccuracy(Healthyweight_Values,Healthyweight_Prediction_S,len(Healthyweight)))
Accuracy_Side.append(GetAccuracy(Overweight_Values,Overweight_Prediction_S,len(Overweight)))
Accuracy_Side.append(GetAccuracy(Obese_Values,Obese_Prediction_S,len(Obese)))

Accuracy_Front = []
Accuracy_Front.append(GetAccuracy(Underweight_Values,Underweight_Prediction_F,len(Underweight)))
Accuracy_Front.append(GetAccuracy(Healthyweight_Values,Healthyweight_Prediction_F,len(Healthyweight)))
Accuracy_Front.append(GetAccuracy(Overweight_Values,Overweight_Prediction_F,len(Overweight)))
Accuracy_Front.append(GetAccuracy(Obese_Values,Obese_Prediction_F,len(Obese)))

Accuracy_Comp = []
Accuracy_Comp.append(GetAccuracy(Underweight_Values,Underweight_Prediction_C,len(Underweight)))
Accuracy_Comp.append(GetAccuracy(Healthyweight_Values,Healthyweight_Prediction_C,len(Healthyweight)))
Accuracy_Comp.append(GetAccuracy(Overweight_Values,Overweight_Prediction_C,len(Overweight)))
Accuracy_Comp.append(GetAccuracy(Obese_Values,Obese_Prediction_C,len(Obese)))

#BMI_Stats
BMI_Stats = []
BMI_Stats.append(len(Underweight_Index))
BMI_Stats.append(len(Healthyweight_Index))
BMI_Stats.append(len(Overweight_Index))
BMI_Stats.append(len(Obese_Index))
Category = ['Underweight','Healthy Weight','Overweight','Obese']
#create directory
today = str(datetime.datetime.now().strftime("%d-%b-%Y-%H-%M-%S"))
directory = 'analysis/' + today +'/'
os.makedirs(directory)

#Generate graphs
MakeBar(Accuracy_Front,Category,'Front',directory,False)
MakeBar(Accuracy_Side,Category,'Side',directory,False)
MakeBar(Accuracy_Comp,Category,'Compensator',directory,False)
MakeBar(BMI_Stats,Category,'BMI_stats',directory,True)

#Histogram
#Front Model
ErrorAnalysis(Underweight_Values,Underweight_Prediction_F,len(Underweight),'Underweight','Front')
ErrorAnalysis(Healthyweight_Values,Healthyweight_Prediction_F,len(Healthyweight),'Healthyweight','Front')
ErrorAnalysis(Overweight_Values,Overweight_Prediction_F,len(Overweight),'Overweight','Front')
ErrorAnalysis(Obese_Values,Obese_Prediction_F,len(Obese),'Obese','Front')
#Side Model
ErrorAnalysis(Underweight_Values,Underweight_Prediction_S,len(Underweight),'Underweight','Side')
ErrorAnalysis(Healthyweight_Values,Healthyweight_Prediction_S,len(Healthyweight),'Healthyweight','Side')
ErrorAnalysis(Overweight_Values,Overweight_Prediction_S,len(Overweight),'Overweight','Side')
ErrorAnalysis(Obese_Values,Obese_Prediction_S,len(Obese),'Obese','Side')
#Compensator
ErrorAnalysis(Underweight_Values,Underweight_Prediction_C,len(Underweight),'Underweight','Compensator')
ErrorAnalysis(Healthyweight_Values,Healthyweight_Prediction_C,len(Healthyweight),'Healthyweight','Compensator')
ErrorAnalysis(Overweight_Values,Overweight_Prediction_C,len(Overweight),'Overweight','Compensator')
ErrorAnalysis(Obese_Values,Obese_Prediction_C,len(Obese),'Obese','Compensator')

#total dataset
total = np.concatenate((Underweight_Values,Healthyweight_Values,Overweight_Values,Obese_Values))
print(total.shape)
print(total[0:10])
total_front = np.concatenate((Underweight_Prediction_F,Healthyweight_Prediction_F,Overweight_Prediction_F,Obese_Prediction_F))
total_front = total_front[:,0]
print(total_front.shape)
print(total_front[0:10])
total_side = np.concatenate((Underweight_Prediction_S,Healthyweight_Prediction_S,Overweight_Prediction_S,Obese_Prediction_S))
total_side = total_side[:,0]
total_comp = np.concatenate((Underweight_Prediction_C,Healthyweight_Prediction_C,Overweight_Prediction_C,Obese_Prediction_C))
total_comp = total_comp[:,0]
TotalError(total,total_front,'Front')
TotalError(total,total_side,'Side')
TotalError(total,total_comp,'Compensator')

