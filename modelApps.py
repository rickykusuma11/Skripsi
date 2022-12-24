# Test Model
from tensorflow.keras.models import load_model
#Load the h5 file in the model
model=load_model('model_resnet50.h5')

#Library
import argparse
import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np

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
        return [inputs[0][:, :, self.startY:self.endY,
                self.startX:self.endX]]

# load serialized edge detector from disk ->
print("[INFO] loading edge detector...")
net = cv.dnn.readNetFromCaffe('deploy.prototxt','hed_pretrained_bsds.caffemodel')

# register new layer with the model ->
cv.dnn_registerLayer("Crop", CropLayer)

# load the input image and grab the dimension ->
image_hed = cv.imread('./Test/Basal.01.jpg')
(H, W) = image_hed.shape[:2]

print("Height: ",H)
print("Width: ",W)

# Convert the image to grayscale, make blur it and perform Canny edge detection ->
print("[INFO] Performing Canny edge detection...")
gray = cv.cvtColor(image_hed, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 0)
canny = cv.Canny(blurred, 30, 150)

# Construct a blob out of the input image for the Holistically-Nested Edge Detector ->
blob = cv.dnn.blobFromImage(image_hed, scalefactor=1.0, size=(W, H),
    mean=(104.00698793, 116.66876762, 122.67891434),
    swapRB=False, crop=False)

# Set the blob as the input to the network and perform a forward pass to compute the edges ->
print("[INFO] Performing Holistically-Nested Edge detection...")
net.setInput(blob)
hed = net.forward()
print("before: ",hed)
hed = cv.resize(hed[0,0], (W, H))
print("after:",hed)
hed = (255 * hed).astype("uint8")
# Save hed as a file
cv.imwrite('Basal.hed.jpg',hed)

img=image.load_img('Basal.hed.jpg',target_size=
                    (224,224)) #'Datasets/test1/dog.0.jpg'
x=image.img_to_array(img)
x

#Shape of the image
x.shape
#Normalizing the image pixels values
x=x/255
#Expand the Dimensions of the image
x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape

model.predict(img_data)
#output:
# array([[0.01513638, 0.01566849, 0.9691952 ]], dtype=float32)
a=np.argmax(model.predict(img_data), axis=1)
a==1
#output:

