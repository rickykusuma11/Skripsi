# Put the image after HED filter to Hed Images's Folder

# Library
import cv2 as cv
import glob
import os

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
net = cv.dnn.readNetFromCaffe('deploy.prototxt','hed_pretrained_bsds.caffemodel')
# register new layer with the model ->
cv.dnn_registerLayer("Crop", CropLayer)

# Basal Cell Convert to Hed
# list Cell image to store
image_target ="./Hedimages/Basal/"
image_dir ="./Cellimages/Basal/"
data_path = os.path.join(image_dir,'*.*g')
image_files = glob.glob(data_path)

# Hed Filter Process
for file in image_files:

    original_image=cv.imread(file)
    (H, W) = original_image.shape[:2]

    # Convert the image to grayscale, make blur it and perform Canny edge detection ->
    gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blurred, 30, 150)
    
    # Construct a blob out of the input image for the Holistically-Nested Edge Detector ->
    blob = cv.dnn.blobFromImage(original_image, scalefactor=1.0, size=(W, H),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, crop=False)
    
    # Set the blob as the input to the network and perform a forward pass to compute the edges ->
    net.setInput(blob)
    hed = net.forward()
    #print("before: ",hed)
    hed = cv.resize(hed[0,0], (W, H))
    #print("after:",hed)
    hed = (255 * hed).astype("uint8")

    # Save image to the Target Folder
    cv.imwrite(image_target+os.path.basename(file),hed)
    
print('OK')

# Carcinoma Cell Convert to Hed
# list Cell image to store
image_target ="./Hedimages/Carcinoma/"
image_dir ="./Cellimages/Carcinoma/"
data_path = os.path.join(image_dir,'*.*g')
image_files = glob.glob(data_path)

# Hed Filter Process
for file in image_files:

    original_image=cv.imread(file)
    (H, W) = original_image.shape[:2]

    # Convert the image to grayscale, make blur it and perform Canny edge detection ->
    gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blurred, 30, 150)
    
    # Construct a blob out of the input image for the Holistically-Nested Edge Detector ->
    blob = cv.dnn.blobFromImage(original_image, scalefactor=1.0, size=(W, H),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, crop=False)
    
    # Set the blob as the input to the network and perform a forward pass to compute the edges ->
    net.setInput(blob)
    hed = net.forward()
    #print("before: ",hed)
    hed = cv.resize(hed[0,0], (W, H))
    #print("after:",hed)
    hed = (255 * hed).astype("uint8")

    # Save image to the Target Folder
    cv.imwrite(image_target+os.path.basename(file),hed)
    
print('OK')

# Melanoma Cell Convert to Hed
# list Cell image to store
image_target ="./Hedimages/Melanoma/"
image_dir ="./Cellimages/Melanoma/"
data_path = os.path.join(image_dir,'*.*g')
image_files = glob.glob(data_path)

# Hed Filter Process
for file in image_files:

    original_image=cv.imread(file)
    (H, W) = original_image.shape[:2]

    # Convert the image to grayscale, make blur it and perform Canny edge detection ->
    gray = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blurred, 30, 150)
    
    # Construct a blob out of the input image for the Holistically-Nested Edge Detector ->
    blob = cv.dnn.blobFromImage(original_image, scalefactor=1.0, size=(W, H),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, crop=False)
    
    # Set the blob as the input to the network and perform a forward pass to compute the edges ->
    net.setInput(blob)
    hed = net.forward()
    #print("before: ",hed)
    hed = cv.resize(hed[0,0], (W, H))
    #print("after:",hed)
    hed = (255 * hed).astype("uint8")

    # Save image to the Target Folder
    cv.imwrite(image_target+os.path.basename(file),hed)
    
print('OK')