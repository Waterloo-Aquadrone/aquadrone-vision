'''
Notes by Kimathi, so hit me up if I don't make any sense
Acknowledgements: https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
GENERAL OVERVIEW---------------------------------------------------------------
- train a small network from scratch (as a baseline)
- use the bottleneck features of a pre-trained network
- fine-tune the top layers of a pre-trained network
'''

'''
STEP: ORGANIZING DATASET-----------------------------------------------------
1000 training examples per class, and 400 validation examples per class
Data set directory stucture:
    data/
        train/
            dogs/
                dog001.jpg
                dog002.jpg
                ...
            cats/
                cat001.jpg
                cat002.jpg
                ...
        validation/
            dogs/
                dog001.jpg
                dog002.jpg
                ...
            cats/
                cat001.jpg
                cat002.jpg
                ...
'''

'''
STEP: DATA PRE-PROCESSING AND DATA AUGMENTATION------------------------------
- Augment the data to increase sample size and prevent overfitting and helping
  the model generalize better
- In Keras this can be done via the ImageDataGenerator class allowing you to
  - configure random transformation and normalization operations for training
  - instantiate generations of augmented image batches (and their labels)
    via .flow(data, labels) or .flow_from_directory(directory). These
    generations can be used with the Keras model methods that accept data
    generators as inputs, fit_generator, evaluate_generator and
    predict_generator
'''
# Example----------------------------------------------------------------------
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255, # RGB coefficients  in 0-255 range to 0-1
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
# May be useful to output a couple images to see the augmentation strategy

'''
STEP: TRAINING A SMALL CONVNET
- Should be concerned about overfitting
- Main focus is the entropic capacity of the model -- how much information
  your model is allowed to store. Storing more info has potential to be more
  accurate by leveraging more found features but can also store irrelevant
  features. lower entropic capacity means storing less features which will
  mean storing most significant features found in the data
- Modulating entropic capacity (1) number of parameters in model i.e. the num
  of layers and the size of each layer (2) weight regularization (L1 or L2),
  which means forcing weights to take smaller values. if weights are large
  small changes in input can lead to large changes in the output

- RELU:         Rectified Linear Unt activation function (has its pros/cons)
- POOL Layers:	Have a primary function of progressively reducting the spatial sizes
				i.e. width and height of the input volume to a layer. It is common to insert
				POOL layers between consequtive CONV layers in a CNN architecture
- Dropout:		In an effort to force the network to be more robust we can apply Dropout,
				the process of disconnecting random neurons between layers. This process is
				proven to reduce overfitting, increase accuracy, and allow our network to
				generalize better to unfamiliar images. As denoted by the parameter, 25% of
				the node connections are randomly disconnected (dropped out) between layers
				each training iteration

'''
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Simple stack of 3 convolutional layers with a ReLU activation and followed
# by maxpooling layers
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Increasing the total number of filters learned the deeper you go into a CNN
# (as your input volume size becomes smaller and smaller) is common practice
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# the model so far outputs 3D feature maps (height, width, features)
# on top of it we stick two-fully connected layers
#
model.add(Flatten()) # this converts 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid')) # perfect for a binary classification

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')

'''
STEP: TRAIN MODEL
- Run the script lol
- Really need to figure out cloud computing
- Build a dashboard for stats to find a good number of epochs
'''

'''
STEP: USING BOTTLENECK FEATURES OF PRE-TRAINED NETWORKS
- find an architecture the has been trained on a useful dataset (e.g. VGG16)
- The strategy will be as follows: we will only instantiate the convolutional
  part of the model, everything up to the fully connected layers. We will then
  run this model on our training and validation data once, record the outputs
  (bottleneck features from the pre-trained model: the last activation maps
  before the fully connectd layers) into two numpy arrays. Then will train
  a small fully connected model on top of the stored feattures
- Reason we are storing the features offline rather than adding our fully
  connected model directly on top of a frozen convolutional base and running
  the whole thing is computational efficiency. Most pre-trained models are
  expensive to run and we want to only do it once. Note that this prevents us
  from using data augmentation
'''
'''
STEP: FINE-TUNING THE TOP LAYERS OF THE PRE-TRAINED NETWORK
- To further improve our previous results, we can try to "fine-tune" the last
  convolutional block of the VGG16 model alongside the top-level classifier.
  Fine tuning consists in starting from a trained network, then re-training it
  on a new dataset using very small weight updates, example steps can be
  - instantiate the convolutional base of VGG16 and load its weights
  - add our previously defined fully-connected model on top, and load its weights
  - freeze the layers of the VGG16 model up to the last convolutional block

Note that:
- in order to perform fine-tuning, all layers should start with properly trained
  weights: for instance you should not slap a randomly initialized
  fully-connected network on top of a pre-trained convolutional base. This is
  because the large gradient updates triggered by the randomly initialized
  weights would wreck the learned weights in the convolutional base. In our case
  this is why we first train the top-level classifier, and only then start
  fine-tuning convolutional weights alongside it.
- we choose to only fine-tune the last convolutional block rather than the
  entire network in order to prevent overfitting, since the entire network
  would have a very large entropic capacity and thus a strong tendency to
  overfit. The features learned by low-level convolutional blocks are more
  general, less abstract than those found higher-up, so it is sensible to keep
  the first few blocks fixed (more general features) and only fine-tune the last
  one (more specialized features).
- fine-tuning should be done with a very slow learning rate, and typically with
  the SGD optimizer rather than an adaptative learning rate optimizer such as
  RMSProp. This is to make sure that the magnitude of the updates stays very
  small, so as not to wreck the previously learned features.
'''
