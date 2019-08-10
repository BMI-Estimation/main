import tkinter as tk

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        SelectImageFrame = tk.Frame(self)
        SelectImageFrame.pack()
        LabelFrame = tk.Frame(SelectImageFrame)
        LabelFrame.pack(side=tk.LEFT)
        TextFrame = tk.Frame(SelectImageFrame)
        TextFrame.pack(side=tk.LEFT)
        ButtonFrame = tk.Frame(SelectImageFrame)
        ButtonFrame.pack(side=tk.LEFT)

        FrontImageLabel = tk.Label(LabelFrame, text="Front Image:")
        FrontImageLabel.pack()
        SideImageLabel = tk.Label(LabelFrame, text="Side Image:")
        SideImageLabel.pack()

        FrontImageTextBox = tk.Entry(TextFrame, state=tk.DISABLED)
        FrontImageTextBox.pack()
        SideImageTextBox = tk.Entry(TextFrame, state=tk.DISABLED)
        SideImageTextBox.pack()

        ChooseFrontImageButton = tk.Button(ButtonFrame)
        ChooseFrontImageButton["text"] = "Browse Files ..."
        ChooseFrontImageButton.pack()
        ChooseSideImageButton = tk.Button(ButtonFrame)
        ChooseSideImageButton["text"] = "Browse Files ..."
        ChooseSideImageButton.pack()

        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Open Images"
        # self.hi_there["command"] = self.say_hi
        self.hi_there.pack(side=tk.BOTTOM)

        # self.quit = tk.Button(self, text="QUIT", fg="red",
        #                       command=self.master.destroy)
        # self.quit.pack(side="bottom")

    def say_hi(self):
        print("hi there, everyone!")

root = tk.Tk()
root.title("Digital BMI Estimator")
app = Application(master=root)
app.mainloop()