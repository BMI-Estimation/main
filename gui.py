import tkinter as tk

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

        self.FrontImageTextBox = tk.Entry(self.SelectImageFrame)
        self.FrontImageTextBox.grid(row=0, column=1)
        self.SideImageTextBox = tk.Entry(self.SelectImageFrame)
        self.SideImageTextBox.grid(row=1, column=1)

        self.ChooseFrontImageButton = tk.Button(self.SelectImageFrame)
        self.ChooseFrontImageButton["text"] = "Browse Files ..."
        self.ChooseFrontImageButton.grid(row=0, column=2)
        self.ChooseSideImageButton = tk.Button(self.SelectImageFrame)
        self.ChooseSideImageButton["text"] = "Browse Files ..."
        self.ChooseSideImageButton.grid(row=1, column=2)

        self.UseOnlyFrontImageTickBox = tk.Checkbutton(self.CheckBoxFrame, text="Only Use Front Image", variable=UseOnlyFrontImage, command=self.UseOnlyOneImageCheckBox)
        self.UseOnlyFrontImageTickBox.pack()
        self.UseOnlySideImageTickBox = tk.Checkbutton(self.CheckBoxFrame, text="Only Use Side Image", variable=UseOnlySideImage, command=self.UseOnlyOneImageCheckBox)
        self.UseOnlySideImageTickBox.pack()

        self.StartProgram = tk.Button(self)
        self.StartProgram["text"] = "Predict BMI"
        self.StartProgram.pack(side=tk.BOTTOM)

    def UseOnlyOneImageCheckBox(self):
        if UseOnlyFrontImage.get():
            self.SideImageTextBox.config(state=tk.DISABLED)
            self.UseOnlySideImageTickBox.config(state=tk.DISABLED)
        else:
            self.SideImageTextBox.config(state=tk.NORMAL)
            self.UseOnlySideImageTickBox.config(state=tk.NORMAL)
        
        if UseOnlySideImage.get():
            self.FrontImageTextBox.config(state=tk.DISABLED)
            self.UseOnlyFrontImageTickBox.config(state=tk.DISABLED)
        else:
            self.FrontImageTextBox.config(state=tk.NORMAL)
            self.UseOnlyFrontImageTickBox.config(state=tk.NORMAL)

        # print(UseOnlyFrontImage.get(), UseOnlySideImage.get())

root = tk.Tk()
UseOnlyFrontImage = tk.BooleanVar()
UseOnlySideImage = tk.BooleanVar()
root.title("Digital BMI Estimator")
app = Application(master=root)
app.mainloop()