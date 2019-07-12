import argparse
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

data = img_to_array(img)

samples = expand_dims(data, 0)

datagen = ImageDataGenerator(zoom_range=[0.5,1.0], width_shift_range=[-200,200], height_shift_range=0.5, horizontal_flip=True,
                             rotation_range=90, brightness_range=[0.2,1.0])

it = datagen.flow(samples, batch_size=1)



for i in range(9):
    
    pyplot.subplot(330 + 1 + i)
    batch = it.next()
    image = batch[0].astype('uint8')
    pyplot.imshow(image)
    
pyplot.show()

