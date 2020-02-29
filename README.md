# aquadrone2020vision

# Overview of Project

## Goals
The vision package is intended to take footage from the Zed camera, identify any objects within it, as well as their location, and send that information to the Software team in the form of: item, x, y, z.

## Code structure

There are a couple of major components to the code.
Overall flow:

Happening once in training:
Creating data run on labelled data
run train model and save it

Happening once at start of nano:
load model

Happening continuously on nano:
load image
list(object, boxPoints) = boundingBoxModel.predict(image)
for objectPair in list:
    cornerStruct = edgeDetection.findCorners(object, boxPoints)
    returnList += cornerStruct
return returnList


### Creating Data
splitVideoToFrames
labelFrames
augmentData

### Preprocessing data
createTrainingData class
- labelFrames
    - takes in video and csv with labels
    - creates data at the paramter n fps
    - saves those frames
- augmentData
    - takes labelled frames and augments them based on given parameters (hyper and or passed in)
        - x stretch percent
        - y stretch percent
        - rotation degree
        - zoom percent

### Locating and labelling objects
Currently using bounding boxes to roughly locate objects and classify them
This uses a retrained yolov3, and can output multiple objects in one image

BoundingBoxModel class
- train
    - takes in labelled images
    - creates a model which is kept in this class
- load
    - reads a pickled pre-trained model (specifically for aquadrone tasks)
- predict
    - takes in a single image
    - outputs the boxes (as a set of 4 points), objects within them, and accuracy
- save
    - saves the model to a pickle file

EdgeDetectionClass
- crop image
    - takes box (xmin, xmax, ymin, ymax) and image, crops image to box
- findCorners
    - takes in the name of the object in the box, the image, and the box
    - run crop image
    - returns the specific points according to image struct (ie, gate has 4 corners, but path might have 3 or 6)

### Outputting data
translate corners to differentials
look at individual struct by object for number of points
translate number of received points
ideas
- maybe do some kind of edge detection knowing the object?
- force number of corners / shape