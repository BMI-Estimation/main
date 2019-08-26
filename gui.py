import tkinter as tk
from tkinter import filedialog
from os import getcwd
from PIL import Image, ImageTk
from BMI_Estimation import detect
from keras.models import load_model
import numpy as np

def Image_Segmentation_Data_Extraction(listOfImages):
	arguments = {}
	imageNames = [None, None]
	for image in listOfImages:
		if 'F' in image or 'f' in image:
			imageNames[0] = image
		elif 'S' in image or 's' in image:
			imageNames[1] = image

	imageNames = [entry for entry in imageNames if entry != None]
	arguments["images"] = imageNames

	arguments["width"] = RefObjectWidth.get()

	if ShowMasks.get(): arguments["mask"] = True
	else: arguments["mask"] = False

	if ShowPics.get(): arguments["visualise"] = True
	else: arguments["visualise"] = False

	listOfDimensions = detect(arguments)
	return listOfDimensions

def Predict_BMI(listOfDimensions):
	BMI = None
	if UseOnlyFrontImage.get():
		print("[INFO] Predicting Using Front Model Only", listOfDimensions)
		model = load_model("Front.h5")
		BMI = model.predict(np.asarray(listOfDimensions[0:1]))
	elif UseOnlySideImage.get():
		print("[INFO] Predicting Using Side Model Only", listOfDimensions)
		model = load_model("Side.h5")
		BMI = model.predict(np.asarray(listOfDimensions[0:1]))
	else:
		listOfDimensions = np.asarray(listOfDimensions)
		print("[INFO] Predicting Using All Models", listOfDimensions)
		Fmodel = load_model("Front.h5")
		fBMI = Fmodel.predict(listOfDimensions[0:1])
		Smodel = load_model("Side.h5")
		sBMI = Smodel.predict(listOfDimensions[1:])
		Bmodel = load_model("BMI.h5")
		BMI = Bmodel.predict(np.column_stack((fBMI, sBMI)))
	return BMI

class Application(tk.Frame):
	def __init__(self, master=None):
		super().__init__(master)
		self.master = master
		self.pack()
		self.create_widgets()

	def create_widgets(self):
		self.InstructionsFrame = tk.Frame(self)
		self.InstructionsFrame.grid(row=0)
		self.Instructions = tk.Label(self.InstructionsFrame, text="Please begin by choosing the Front and Side pictures from your computer.\nPictures with an \'F\' is in the file name are considered as the Front Image, and those with an \'S\' in the file name are considered as the Side Image.", wraplength=500)
		self.Instructions.grid(row=0)
		self.RefObjectMeasurement = tk.Frame(self)
		self.RefObjectMeasurement.grid(row=1)
		self.SelectImageFrame = tk.Frame(self)
		self.SelectImageFrame.grid(row=2, column=0)
		self.CheckBoxFrame = tk.Frame(self)
		self.CheckBoxFrame.grid(row=3, column=0)
		self.BMIFrame = tk.Frame(self)
		self.BMIFrame.grid(row=4, column=0)

		self.StartProgram = tk.Button(self.BMIFrame, text="Predict BMI", command=self.start)
		self.StartProgram.grid(row=0, column=0)
		self.Prediction = tk.Label(self.BMIFrame, text="Not Predicted")
		self.Prediction.grid(row=0, column=1)

		self.RefObjectMeasurementLabel = tk.Label(self.RefObjectMeasurement, text="Width of Reference Object in meters: ")
		self.RefObjectMeasurementLabel.grid(row=0, column=0)
		self.RefObjectMeasurementEntry = tk.Entry(self.RefObjectMeasurement, textvariable=RefObjectWidth)
		self.RefObjectMeasurementEntry.grid(row=0, column=1)

		self.FrontImageLabel = tk.Label(self.SelectImageFrame, text="Front Image:")
		self.FrontImageLabel.grid(row=0)
		self.SideImageLabel = tk.Label(self.SelectImageFrame, text="Side Image:")
		self.SideImageLabel.grid(row=1)

		self.FrontImageTextBox = tk.Entry(self.SelectImageFrame, textvariable=FrontFileName)
		self.FrontImageTextBox.grid(row=0, column=1)
		self.SideImageTextBox = tk.Entry(self.SelectImageFrame, textvariable=SideFileName)
		self.SideImageTextBox.grid(row=1, column=1)

		self.ChooseFrontImageButton = tk.Button(self.SelectImageFrame, command=self.chooseFrontFile)
		self.ChooseFrontImageButton["text"] = "Browse Files ..."
		self.ChooseFrontImageButton.grid(row=0, column=2)
		self.ChooseSideImageButton = tk.Button(self.SelectImageFrame, command=self.chooseSideFile)
		self.ChooseSideImageButton["text"] = "Browse Files ..."
		self.ChooseSideImageButton.grid(row=1, column=2)

		self.FrontPhotoFrame = tk.Frame(self.SelectImageFrame)
		self.FrontPhotoFrame.grid(row=0, column=3)
		self.FrontPhotoBox = tk.Canvas(self.FrontPhotoFrame, width=200)
		self.FrontPhotoBox.grid(row=0, column=1)
		self.FrontPhotoLabel = tk.Label(self.FrontPhotoFrame, text="No File Selected.")
		self.FrontPhotoLabel.grid(row=0,column=0)

		self.SidePhotoFrame = tk.Frame(self.SelectImageFrame)
		self.SidePhotoFrame.grid(row=1, column=3)
		self.SidePhotoBox = tk.Canvas(self.SidePhotoFrame, width=200)
		self.SidePhotoBox.grid(row=0, column=1)
		self.SidePhotoLabel = tk.Label(self.SidePhotoFrame, text="No File Selected.")
		self.SidePhotoLabel.grid(row=0, column=0)

		self.UseOnlyFrontImageTickBox = tk.Checkbutton(self.CheckBoxFrame, text="Only Use Front Image", variable=UseOnlyFrontImage, command=self.UseOnlyOneImage)
		self.UseOnlyFrontImageTickBox.grid(row=0, column=0)
		self.UseOnlySideImageTickBox = tk.Checkbutton(self.CheckBoxFrame, text="Only Use Side Image", variable=UseOnlySideImage, command=self.UseOnlyOneImage)
		self.UseOnlySideImageTickBox.grid(row=0, column=1)
		self.ShowMasksTickBox = tk.Checkbutton(self.CheckBoxFrame, text="Show Masks", variable=ShowMasks)
		self.ShowMasksTickBox.grid(row=0, column=2)
		self.ShowPicsTickBox = tk.Checkbutton(self.CheckBoxFrame, text="Show Image Segmentation", variable=ShowPics)
		self.ShowPicsTickBox.grid(row=0, column=3)

	def start(self):
		if UseOnlyFrontImage.get():
			listOfDimensions = Image_Segmentation_Data_Extraction([FrontFileName.get()])
		elif UseOnlySideImage.get():
			listOfDimensions = Image_Segmentation_Data_Extraction([SideFileName.get()])
		else: listOfDimensions = Image_Segmentation_Data_Extraction([FrontFileName.get(), SideFileName.get()])
		print('[INFO] Image Segmentation Completed')
		BMI = Predict_BMI(listOfDimensions)
		self.Prediction.config(text="\tYour BMI is " + str(BMI[0][0]))
		print('[INFO] BMI Prediction Completed')
	
	def UseOnlyOneImage(self):
		if UseOnlyFrontImage.get():
			[slave.config(state=tk.DISABLED) for slave in self.SelectImageFrame.grid_slaves()[:-1] if slave.grid_info()['row']== 1 and slave.winfo_class() != 'Frame']
			self.UseOnlySideImageTickBox.config(state=tk.DISABLED)
		else:
			[slave.config(state=tk.NORMAL) for slave in self.SelectImageFrame.grid_slaves()[:-1] if slave.grid_info()['row']== 1 and slave.winfo_class() != 'Frame']
			self.UseOnlySideImageTickBox.config(state=tk.NORMAL)
				
		if UseOnlySideImage.get():
			[slave.config(state=tk.DISABLED) for slave in self.SelectImageFrame.grid_slaves()[:-1] if slave.grid_info()['row']== 0 and slave.winfo_class() != 'Frame']
			self.UseOnlyFrontImageTickBox.config(state=tk.DISABLED)
		else:
			[slave.config(state=tk.NORMAL) for slave in self.SelectImageFrame.grid_slaves()[:-1] if slave.grid_info()['row']== 0 and slave.winfo_class() != 'Frame']
			self.UseOnlyFrontImageTickBox.config(state=tk.NORMAL)
		
	def returnFileName(self):
		fileName = filedialog.askopenfilename(initialdir = getcwd(), title = "Select file",filetypes = (("jpg files","*.jpg"), ("jpeg files","*.jpeg"),("all files","*.*")))
		if fileName != "": return fileName
		else: return "No File Selected."

	def chooseFile(self, textbox, box, label, is_front):
		fileName = self.returnFileName()
		if fileName == "No File Selected.":
			return
		textbox.insert(tk.INSERT, fileName)
		image = Image.open(fileName)
		image.thumbnail((200,200))
		fileName = fileName.split('/')[-1]
		if is_front:
			self.FrontPhoto = ImageTk.PhotoImage(image)
			label.config(text=fileName)
			box.create_image(self.FrontPhotoBox.winfo_width()/2, self.FrontPhotoBox.winfo_height()/2, image=self.FrontPhoto)
		else:
			self.SidePhoto = ImageTk.PhotoImage(image)
			label.config(text=fileName)
			box.create_image(self.SidePhotoBox.winfo_width()/2, self.SidePhotoBox.winfo_height()/2, image=self.SidePhoto)
		
	def chooseFrontFile(self):
		FrontFileName.set("")
		self.chooseFile(self.FrontImageTextBox, self.FrontPhotoBox, self.FrontPhotoLabel, True)
		return
		
	def chooseSideFile(self):
		SideFileName.set("")
		self.chooseFile(self.SideImageTextBox, self.SidePhotoBox, self.SidePhotoLabel, False)
		return

root = tk.Tk()
UseOnlyFrontImage = tk.BooleanVar()
UseOnlySideImage = tk.BooleanVar()
ShowMasks = tk.BooleanVar()
ShowPics = tk.BooleanVar()
FrontFileName = tk.StringVar()
SideFileName = tk.StringVar()
RefObjectWidth = tk.DoubleVar()
root.title("Digital BMI Estimator")
app = Application(master=root)
app.mainloop()