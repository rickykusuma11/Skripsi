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

#Run folder data mentah untuk diolah dengan Holistically Nested Edge Detection (HED)
# Put the image after HED filter to Hed Images's Folder

# Library
import cv2 as cv
import glob
import os

# load serialized edge detector from disk ->
net = cv.dnn.readNetFromCaffe('deploy.prototxt','hed_pretrained_bsds.caffemodel')
# register new layer with the model ->
cv.dnn_registerLayer("Crop", CropLayer)

#Folder 1
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

#Folder2
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

#Folder3
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

# Masuk ke Model Convolutional Neural Network (CNN)

# import the required libraries
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
#import VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# re-size all the images to this
IMAGE_SIZE = [224, 224]
train_path = 'Datasets/train'
valid_path = 'Datasets/test'

# Use imagenet weights
resnet = ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet',
                  include_top=False)

#Membuat Fully Connected Layers

# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False
# This helps to get number of output classes
folders = glob('Datasets/train/*')
# Our VGG16 Layers
x = Flatten()(resnet.output)
prediction = Dense(len(folders), activation='softmax')(x)
# creating object model
model = Model(inputs=resnet.input, outputs=prediction)
model.summary()

# Training & Testing

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# Using the Image Data Generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255, shear_range =
                      0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('Datasets/train',
                                         target_size = (224, 224),
                                         batch_size = 30,
                                         class_mode = 'categorical')
#output:
#Found 64 images belonging to 3 classes.
test_set = test_datagen.flow_from_directory('Datasets/test',
                                        target_size = (224, 224),
                                        batch_size = 30,
                                        class_mode = 'categorical')
#output:
#Found 58 images belonging to 3 classes.
# fit the model
r = model.fit_generator(training_set, validation_data=test_set,
                       epochs=10, steps_per_epoch=len(training_set),
                       validation_steps=len(test_set))

#Diagram Akurasi

# ploting the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')
# ploting the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

# Confussion Matrix 

# save it as a h5 file
from tensorflow.keras.models import load_model
model.save('model_resnet50.h5')

y_pred = model.predict(test_set)
y_pred

Y_pred = np.argmax(y_pred, axis=1)
Y_pred

# Create Function for Plot Confusion Matrix
def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.around(cm, decimals=2)
        cm[np.isnan(cm)] = 0.0
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools 

target_names = []
for key in training_set.class_indices:
    target_names.append(key)

print('Confusion Matrix')
cm = confusion_matrix(test_set.classes, Y_pred)
plot_confusion_matrix(cm, target_names,title='Confusion Matrix')

#Print Classification Report
print('Classification Report')
print(classification_report(test_set.classes, Y_pred, target_names=target_names))

yh = model.predict(img_data)
for i in range(len(img_data)):
  if(np.argmax(model.predict(img_data)) == 0):
      print("KarsiomaSB")
  elif(np.argmax(model.predict(img_data)) == 1 ) :
      print("KarsiomaSS")
  elif(np.argmax(model.predict(img_data)) == 2 ) :
      print("Melanoma")
  else:
      print("Lainya")
