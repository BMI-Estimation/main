import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk

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

        self.FrontPhotoFrame = tk.Frame(self.SelectImageFrame)
        self.FrontPhotoFrame.grid(row=0, column=3)
        self.FrontPhotoLabel = tk.Label(self.FrontPhotoFrame, text="No File Selected.", image="")
        self.FrontPhotoLabel.pack()
        self.SidePhotoFrame = tk.Frame(self.SelectImageFrame)
        self.SidePhotoFrame.grid(row=1, column=3)
        self.SidePhotoLabel = tk.Label(self.SidePhotoFrame, text="No File Selected.", image="")
        self.SidePhotoLabel.pack(side=tk.BOTTOM)

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

    def chooseFile(self, textbox, label, is_front):
        fileName = self.returnFileName()
        if fileName == "No File Selected.":
            return
        textbox.insert(tk.INSERT, fileName)
        image = Image.open(fileName)
        image.thumbnail((100,100))
        if is_front:
            self.FrontPhoto = ImageTk.PhotoImage(image)
            label.config(text=fileName, image=self.FrontPhoto)
        else:
            self.SidePhoto = ImageTk.PhotoImage(image)
            label.config(text=fileName, image=self.SidePhoto)
        label.pack(side=tk.BOTTOM)
    
    def chooseFrontFile(self):
        FrontFileName.set("")
        self.chooseFile(self.FrontImageTextBox, self.FrontPhotoLabel, True)
        return
    
    def chooseSideFile(self):
        SideFileName.set("")
        self.chooseFile(self.SideImageTextBox, self.SidePhotoLabel, False)
        return
        

root = tk.Tk()
UseOnlyFrontImage = tk.BooleanVar()
UseOnlySideImage = tk.BooleanVar()
FrontFileName = tk.StringVar()
SideFileName = tk.StringVar()
root.title("Digital BMI Estimator")
app = Application(master=root)
app.mainloop()