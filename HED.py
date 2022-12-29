#Library
import argparse
import cv2 as cv
import os
import matplotlib.pyplot as plt
import glob

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
image = cv.imread('./DataInput.png')
(H, W) = image.shape[:2]

print("Height: ",H)
print("Width: ",W)

# Construct a blob out of the input image for the Holistically-Nested Edge Detector ->
blob = cv.dnn.blobFromImage(image, scalefactor=1.0, size=(W, H),
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

# Show Input Image
plt.imshow(image)

# Show Output Holistically-Nested Edge Detection 
plt.imshow(hed)

cv.imwrite('Output.png',hed)

