# import the necessary packages
# Documentation: https://keras.io/
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

# A smaller variant of VGGNet -> https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/
# VGGNet-like models share two common characteristics
# 	1 Only 3x3 convolutions are used
#	2 Convolution layers are stacked on top of each other deeper in the network architecture
#	  prior to applyting a destructive pooling operation
class SmallVGGNet:
	@staticmethod
	def build(width, height, depth, classes):
		# depth: 	number of channels, the number of colors that make up a single pixels
		#			E.g. RGB color space has a depth of 3

		# initialize the model along with the input shape to be
		# "channels last" and the channels dimension itself
		model = Sequential()
		inputShape = (height, width, depth)
		chanDim = -1

		# if we are using "channels first", update the input shape
		# and channels dimension
		if K.image_data_format() == "channels_first":
			inputShape = (depth, height, width)
			chanDim = 1

		# CONV => RELU => POOL layer set
		# RELU - Rectified Linear Unt activation function (has its pros/cons)
		# The first CONV layer has 32 filters of size 3x3
		# BatchNormalization:	Used to normalize the activations of a given input volume
		# 						before passing it to the next layer in the network. I has been
		#						proven to be very effective at reducing the number of epochs required
		#						to train a CNN as well as stabalizing training itself
		# POOL Layers:	Have a primary function of progressively reducting the spatial sizes
		#				i.e. width and height of the input volume to a layer. It is common to insert
		#				POOL layers between consequtive CONV layers in a CNN architecture
		# Dropout:		In an effort to force the network to be more robust we can apply Dropout,
		#				the process of disconnecting random neurons between layers. This process is
		#				proven to reduce overfitting, increase accuracy, and allow our network to
		#				generalize better to unfamiliar images. As denoted by the parameter, 25% of
		#				the node connections are randomly disconnected (dropped out) between layers
		#				each training iteration
		model.add(Conv2D(32, (3, 3), padding="same",
			input_shape=inputShape))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 2 => POOL layer set
		# Notice the total number of filters learned by the CONV layers has doubled from 32->64
		# Increasing the total number of filters learned the deeper you go into a CNN (as your
		# input volume size becomes smaller and smaller) is common practice
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(64, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# (CONV => RELU) * 3 => POOL layer set
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(Conv2D(128, (3, 3), padding="same"))
		model.add(Activation("relu"))
		model.add(BatchNormalization(axis=chanDim))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Dropout(0.25))

		# first (and only) set of FC => RELU layers
		# Fully connected layers are denoted by 'Dense' in Keras. The final layer is fully connectec
		# with the three outputs (since we have three classes in our dataset). The 'softmax' layers
		# returns the class probabilites for each label
		model.add(Flatten())
		model.add(Dense(512))
		model.add(Activation("relu"))
		model.add(BatchNormalization())
		model.add(Dropout(0.5))

		# softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# return the constructed network architecture
		return model
