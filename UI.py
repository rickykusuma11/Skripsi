import tkinter as tk
import cv2 as cv

from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk


# Define Class for Crop Layer ->
class CropLayer(object):

    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0

    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])

        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H

        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]

    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY, self.startX:self.endX]]


class DetectionFrame():

    def __init__(self, root) -> None:
        self.filename = None
        self.imageOutputFrame = None

        # load model
        print("[INFO] loading edge detector...")
        net = cv.dnn.readNetFromCaffe('deploy.prototxt',
                                      'hed_pretrained_bsds.caffemodel')
        print("Model and config loaded")
        # register new layer with the model ->
        cv.dnn_registerLayer("Crop", CropLayer)

        detectionFrame = ttk.Frame(root, padding=10)
        self.tab = detectionFrame
        ttk.Label(detectionFrame, text="Nama: Ricky Kusuma").grid(column=0,
                                                                  row=0,
                                                                  sticky=tk.W)
        ttk.Label(detectionFrame, text="NIM: 535170037").grid(column=0,
                                                              row=1,
                                                              sticky=tk.W)

        actionFrame = ttk.Frame(detectionFrame, padding=30)
        actionFrame.grid(column=0, row=2, columnspan=4, sticky=tk.W)

        # filenameInit = "/home/julius/Projects/Skripsi/DataInput.png"
        fileNameLabel = ttk.Label(actionFrame, text="File Name : ")
        fileNameLabel.grid(column=0, row=1, pady=30, columnspan=3, sticky=tk.W)
        imageInputFrame = ttk.Frame(detectionFrame,
                                    height=250,
                                    width=250,
                                    borderwidth=2,
                                    relief="solid")
        imageInputFrame.grid(column=0, row=3)

        self.inputImage = ttk.Label(imageInputFrame)
        self.inputImage.pack(ipadx=100, ipady=100, fill=tk.BOTH, expand=True)

        def browseFiles():
            filename = filedialog.askopenfilename(
                initialdir="/home/julius/Projects/Skripsi",
                title="Select a File",
                filetypes=(("PNG files", "*.png"), ("JPG files", "*.jpg"),
                           ("All files", "*.*")))
            if filename != "":
                self.filename = filename
                # Change label contents
                fileNameLabel.configure(text="File Name : " + filename)
                img = Image.open(filename)
                imgResize = img.resize((250, 250))
                self.imageFile = ImageTk.PhotoImage(imgResize)

                self.inputImage.destroy()
                self.inputImage = ttk.Label(imageInputFrame,
                                            image=self.imageFile)
                self.inputImage.pack(expand=True)

                if self.imageOutputFrame:
                    self.imageOutputFrame.destroy()
                    self.arrow.destroy()

        def process():
            if self.filename == None:
                return

            # load the input image and grab the dimension ->
            image = cv.imread(self.filename)
            (H, W) = image.shape[:2]

            print("Height: ", H)
            print("Width: ", W)

            # Construct a blob out of the input image for the Holistically-Nested Edge Detector ->
            blob = cv.dnn.blobFromImage(image,
                                        scalefactor=1.0,
                                        size=(W, H),
                                        mean=(104.00698793, 116.66876762,
                                              122.67891434),
                                        swapRB=False,
                                        crop=False)

            # Set the blob as the input to the network and perform a forward pass to compute the edges ->
            print("[INFO] Performing Holistically-Nested Edge detection...")
            net.setInput(blob)
            hed = net.forward()
            print("before: ", hed)
            hed = cv.resize(hed[0, 0], (W, H))
            print("after:", hed)
            hed = (255 * hed).astype("uint8")
            print("[INFO] Calculation done")

            cv.dnn_unregisterLayer("Crop")

            cv.imwrite("output.png", hed)
            img = Image.open("output.png")
            imgResize = img.resize((250, 250))
            self.outputFile = ImageTk.PhotoImage(imgResize)

            self.arrow = ttk.Label(detectionFrame, text="->")
            self.arrow.grid(column=1, row=3, padx=30)
            self.imageOutputFrame = ttk.Frame(detectionFrame,
                                              height=250,
                                              width=250,
                                              borderwidth=2,
                                              relief="solid")
            self.imageOutputFrame.grid(column=2, row=3)
            outputImageFile = ttk.Label(self.imageOutputFrame,
                                        image=self.outputFile)
            outputImageFile.pack(expand=True)

        ttk.Button(actionFrame,
                   text="Select File",
                   padding=10,
                   command=browseFiles).grid(column=0, row=0, padx=30)
        ttk.Button(actionFrame, text="Process", padding=10,
                   command=process).grid(column=1, row=0)


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
