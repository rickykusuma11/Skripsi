import tkinter as tk

from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk


class DetectionFrame():

    def __init__(self, root) -> None:
        detectionFrame = ttk.Frame(root, padding=10)
        self.tab = detectionFrame
        ttk.Label(detectionFrame, text="Nama: Ricky Kusuma").grid(column=0,
                                                                  row=0,
                                                                  sticky=tk.W)
        ttk.Label(detectionFrame, text="NIM: 535170037").grid(column=0,
                                                              row=1,
                                                              sticky=tk.W)

        actionFrame = ttk.Frame(detectionFrame, padding=30)
        actionFrame.grid(column=0, row=2)

        # filenameInit = "/home/julius/Projects/Skripsi/DataInput.png"
        fileNameLabel = ttk.Label(actionFrame, text="File Name : ")
        fileNameLabel.grid(column=0, row=1, pady=30, columnspan=3, sticky=tk.W)
        imageCanvas = ttk.Frame(actionFrame,
                                height=250,
                                width=250,
                                borderwidth=2,
                                relief="solid")
        imageCanvas.grid(column=0, row=2)

        # img = Image.open(filenameInit)
        # imgResize = img.resize((250, 250))
        # imageFile = ImageTk.PhotoImage(imgResize)
        gambar = ttk.Label(imageCanvas)
        gambar.grid(column=0, row=3)

        def browseFiles():
            filename = filedialog.askopenfilename(
                initialdir="/home/julius/Projects/Skripsi",
                title="Select a File",
                filetypes=(("PNG files", "*.png"), ("JPG files", "*.jpg"),
                           ("All files", "*.*")))
            if filename != "":
                # Change label contents
                fileNameLabel.configure(text="File Name : " + filename)
                img = Image.open(filename)
                imgResize = img.resize((250, 250))
                self.imageFile = ImageTk.PhotoImage(imgResize)

                gambar = ttk.Label(imageCanvas, image=self.imageFile)
                gambar.grid(column=0, row=5)

        ttk.Button(actionFrame,
                   text="Select File",
                   padding=10,
                   command=browseFiles).grid(column=0, row=0, padx=30)
        ttk.Button(actionFrame,
                   text="Process",
                   padding=10,
                   command=root.destroy).grid(column=1, row=0)


def create_about_frame(root):
    aboutFrame = ttk.Frame(root, padding=10)
    return aboutFrame


def create_help_frame(root):
    helpFrame = ttk.Frame(root, padding=10)
    return helpFrame


def create_home_frame(root, onStart):
    homeFrame = ttk.Frame(root, padding=10)
    homeFrame.pack(fill=tk.X, expand=True)
    ttk.Label(homeFrame,
              text="Klasifikasi Jenis Kanker Kulit",
              font=("arial", 20, "bold")).pack(fill=tk.BOTH)

    nameFrame = ttk.Frame(homeFrame, padding=30)
    nameFrame.pack(fill=tk.X)
    ttk.Label(nameFrame,
              text="Oleh : Ricky Kusuma",
              font=("arial", 16, "bold")).grid(column=0, row=0, sticky=tk.W)
    ttk.Label(nameFrame, text="NIM : 535170037",
              font=("arial", 16, "bold")).grid(column=0, row=1, sticky=tk.W)

    ttk.Button(homeFrame, text="Start", padding=10, command=onStart).pack()

    return homeFrame


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Klasifikasi Jenis Kanker Kulit - Ricky Kusuma")

        # self.attributes('-fullscreen', True)
        self.attributes('-topmost', 1)

        # set the position of the window to the center of the screen
        self.geometry('1024x768+0+0')
        tabs = ttk.Notebook(self)

        tab2 = DetectionFrame(tabs)
        tab3 = create_about_frame(tabs)
        tab4 = create_help_frame(tabs)

        def select_detection():
            tabs.select(tab2.tab)

        tab1 = create_home_frame(tabs, select_detection)
        tabs.add(tab1, text="Home")
        tabs.add(tab2.tab, text="Detection")
        tabs.add(tab3, text="About Me")
        tabs.add(tab4, text="Help")
        tabs.pack(expand=1, fill=tk.BOTH)
        tabs.select(tab2.tab)


if __name__ == "__main__":
    app = App()
    app.mainloop()
