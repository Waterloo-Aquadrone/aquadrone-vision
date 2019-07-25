import argparse
import os
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot

#   EXAMPLE:
#   python data_augmentation.py --image ./(path to image)

# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
    help="path to the video file")
args = vars(ap.parse_args())

img = load_img(args["image"])

data = expand_dims(img, 0)

save_here = "images"
if not os.path.isdir(save_here):
    os.makedirs(save_here)

datagen = ImageDataGenerator(zoom_range=[0.7,1.0], width_shift_range=[-100,100], height_shift_range=0.5, horizontal_flip=True,
                             rotation_range=20, brightness_range=[0.2,1.0])

datagen.fit(data)

for x, val in zip(datagen.flow(data,save_to_dir=save_here,save_prefix='aug',save_format='png'),range(20)):
    pass