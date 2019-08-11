import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk
import subprocess

def BMI_Prediction(listOfImages):
    for image in listOfImages:
        if 'F' in image:
            Front = image
        elif 'S' in image:
            Side = image
    subprocess.run(["python", "BMI_Estimation.py", "-w", "0.29","-f", Front, "-s", Side, "-m"])
    return "BMI Prediction"

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
        self.SelectImageFrame = tk.Frame(self)
        self.SelectImageFrame.grid(row=1, column=0)
        self.CheckBoxFrame = tk.Frame(self)
        self.CheckBoxFrame.grid(row=2, column=0)

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

        self.StartProgram = tk.Button(self, text="Predict BMI", command=self.start)
        self.StartProgram.grid(row=3, column=0)

    def start(self):
        bmi = BMI_Prediction([FrontFileName.get(), SideFileName.get()])
        print(bmi)

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
        fileName = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select file",filetypes = (("jpg files","*.jpg"), ("jpeg files","*.jpeg"),("all files","*.*")))
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
FrontFileName = tk.StringVar()
SideFileName = tk.StringVar()
root.title("Digital BMI Estimator")
app = Application(master=root)
app.mainloop()