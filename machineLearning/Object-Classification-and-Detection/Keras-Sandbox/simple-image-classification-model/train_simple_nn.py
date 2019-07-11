# USAGE
# python train_simple_nn.py --dataset animals --model output/simple_nn.model --label-bin output/simple_nn_lb.pickle --plot output/simple_nn_plot.png

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
#matplotlib: 	This is the go-to plotting package for Python
#				Refer to this blog pos for help https://www.pyimagesearch.com/2015/08/24/resolved-matplotlib-figures-not-showing-up-or-displaying/
#				We instruct matplotlib to use the 'Aff' backend for saving plots to disk

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
#sklearn: 	The scikit-learn library for binarizing our labels, splitting data for training/testing
#			and genrating a training report in our terminal
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
#keras:	High level frontend into TensorFlow and other deep learing backends
from imutils import paths
#imutils:	Convenience functions. We'll use the 'paths' module to generate a list of image
#			file paths for training
import matplotlib.pyplot as plt
import numpy as np
#numpy:	For numerical processing with python. Big dependecy
import argparse
import random
import pickle
import cv2
import os

# construct the argument parser and parse the arguments
# https://www.pyimagesearch.com/2018/03/12/python-argparse-command-line-arguments/
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset of images")
ap.add_argument("-m", "--model", required=True,
	help="path to output trained model")
ap.add_argument("-l", "--label-bin", required=True,
	help="path to output label binarizer")
ap.add_argument("-p", "--plot", required=True,
	help="path to output accuracy/loss plot")
args = vars(ap.parse_args())

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, resize the image to be 32x32 pixels (ignoring
	# aspect ratio), flatten the image into 32x32x3=3072 pixel image
	# into a list, and store the image in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (32, 32)).flatten()
	data.append(image)

	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
# It is typical to allocate a percentage of your data for training and
# a smaller percentage of your data for testing
# Both 'trainx' and testX make up the image data itself while trainY and testY
# make up the labels. Our class lables are currently represented as strings;
# however, Keras will assume that both: 1 Labels are encoded as integers
# 2 One-hot encoding is performed on these labels making each label represented
# as a vector rather than an integer
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors (for 2-class, binary
# classification you should use Keras' to_categorical function
# instead as the scikit-learn's LabelBinarizer will not return a
# vector)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)
# fit_transform Finds all unique class labels in trainY and then tranformas them into one-hot
# encoded labels
#Examples
#[1, 0, 0] # corresponds to cats
#[0, 1, 0] # corresponds to dogs
#[0, 0, 1] # corresponds to panda
# Notice how only one of the array elements is "hot" which is why we call this 'one-hot' encoding

# define the 3072-1024-512-3 architecture using Keras
# Define our neural network architecture using Keras. Here we will be using a neural network with One
# input layer, two hidden layers, and one output layer
# It's usually better to make a seperate class in a seperate file for model architecture
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))
# Line(1)The input layer and first hidden layer are defined. Will have an input shape of 3072
# as there are 32x32x3=3072 pixels in a flattened input image. The firs hiddend layer will have
# 1024 nodes
# The number of nodes in the final output layer will be the number of possible class Labels
# in this case it will be three, ('cates', dogs, and 'panda' respectively)

# initialize our initial learning rate and # of epochs to train for
INIT_LR = 0.01
EPOCHS = 75

# compile the model using SGD as our optimizer and categorical
# cross-entropy loss (you'll want to use binary_crossentropy
# for 2-class classification)
print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
# compiling our model using the Stochastic Gradient Descent optomizer with 'categorical...'
# as the loss function. 'Categorical cross-entropy is used as the loss for nearly all networks
# trained to perform classification. The only exception is for the 2-class classification where
# there are only two possible class labels. In that event you would want to swap out
# 'categorical_crossentropy' for 'binary_crossentropy'

# train the neural network / 'fit' the compiled model for our training data
# training Split -> Compiled Model -> Fit Model
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=EPOCHS, batch_size=32)
# batch_size controls the size of each group of data to pass through the networks
# Larger GPUs would be able to accomodate larger batch sizes. Start with 32or 64 and go up

# evaluate the network
# Fit Model -> Testing Split -> Predictions
# Import to evaluate on our testing data so we can obtain an unbaised representation of how
# well our model is performing with data it has never been trained on
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))

# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

# save the model and label binarizer to disk
print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
