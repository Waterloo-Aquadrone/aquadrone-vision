import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import PIL

image_name = 'mask'
imgb = cv2.imread(image_name + '.jpg')

#obtain the threshold for Canny using the grayscale image
img_grayscale = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
v = np.median(img_grayscale)
sigma = 0.33
#phi = 0.11
#first threshold

lower_thresh = int(max(0, (1.0 - sigma) * v))
upper_thresh = int(min(255, (1.0 + sigma) * v))
#second threshold
'''
if False:
    lower_thresh = int(max(0, (1.0 - 2 * sigma) * v))
    upper_thresh = int(min(255, v))
#third threshold
lower_thresh = int(max(0, (1.0 - 2 * sigma) * v))
upper_thresh = int(min(255, (1.0 + sigma) * v))

#TODO: set lower_thresh and upper_thresh lower
'''

#get the edge
edges = cv2.Canny(imgb, lower_thresh, upper_thresh)
print(edges)
edgesimg2 = PIL.Image.fromarray(edges)
edgesimg2.save(image_name + '-threshold' + '.jpg')