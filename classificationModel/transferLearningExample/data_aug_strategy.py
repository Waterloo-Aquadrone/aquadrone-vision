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

  - code below outputs data augmentation strategy
'''

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# will not add rescaling because we want to display the full image
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

img = load_img('data/train/cats/0.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# .flow() takes data & label arrays and generates batches of augmented data
i = 0
for batch in datagen.flow(x, batch_size=1,save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break
