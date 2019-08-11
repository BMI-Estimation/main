import tkinter as tk
from tkinter import filedialog
import os

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.SelectImageFrame = tk.Frame(self)
        self.SelectImageFrame.pack()
        self.CheckBoxFrame = tk.Frame(self)
        self.CheckBoxFrame.pack()

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

        self.UseOnlyFrontImageTickBox = tk.Checkbutton(self.CheckBoxFrame, text="Only Use Front Image", variable=UseOnlyFrontImage, command=self.UseOnlyOneImage)
        self.UseOnlyFrontImageTickBox.pack()
        self.UseOnlySideImageTickBox = tk.Checkbutton(self.CheckBoxFrame, text="Only Use Side Image", variable=UseOnlySideImage, command=self.UseOnlyOneImage)
        self.UseOnlySideImageTickBox.pack()

        self.StartProgram = tk.Button(self, text="Predict BMI", command=self.start)
        self.StartProgram.pack(side=tk.BOTTOM)

    def start(self):
        print(FrontFileName.get(), SideFileName.get())

    def UseOnlyOneImage(self):
        if UseOnlyFrontImage.get():
            [slave.config(state=tk.DISABLED) for slave in self.SelectImageFrame.grid_slaves() if slave.grid_info()['row']== 1]
            self.UseOnlySideImageTickBox.config(state=tk.DISABLED)
        else:
            [slave.config(state=tk.NORMAL) for slave in self.SelectImageFrame.grid_slaves() if slave.grid_info()['row']== 1]
            self.UseOnlySideImageTickBox.config(state=tk.NORMAL)
        
        if UseOnlySideImage.get():
            [slave.config(state=tk.DISABLED) for slave in self.SelectImageFrame.grid_slaves() if slave.grid_info()['row']== 0]
            self.UseOnlyFrontImageTickBox.config(state=tk.DISABLED)
        else:
            [slave.config(state=tk.NORMAL) for slave in self.SelectImageFrame.grid_slaves() if slave.grid_info()['row']== 0]
            self.UseOnlyFrontImageTickBox.config(state=tk.NORMAL)
    
    def returnFileName(self):
        fileName = filedialog.askopenfilename(initialdir = os.getcwd(), title = "Select file",filetypes = (("jpg files","*.jpg"), ("jpeg files","*.jpeg"),("all files","*.*")))
        return fileName
    
    def chooseFrontFile(self):
        self.FrontImageTextBox.insert(tk.INSERT, self.returnFileName())
    
    def chooseSideFile(self):
        self.SideImageTextBox.insert(tk.INSERT, self.returnFileName())

root = tk.Tk()
UseOnlyFrontImage = tk.BooleanVar()
UseOnlySideImage = tk.BooleanVar()
FrontFileName = tk.StringVar()
SideFileName = tk.StringVar()
root.title("Digital BMI Estimator")
app = Application(master=root)
app.mainloop()