import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import PIL

def threshCanny(image_name, threshold = 1):
    imgb = cv2.imread(image_name + '.jpg')
    img_grayscale = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    v = np.median(img_grayscale)    
    sigma = 0.33

    if threshold == 1:
        lower_thresh = int(max(0, (1.0 - sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))
    elif threshold == 2:
        lower_thresh = int(max(0, (1.0 - 2 * sigma) * v))
        upper_thresh = int(min(255, v))
    else:
        lower_thresh = int(max(0, (1.0 - 2 * sigma) * v))
        upper_thresh = int(min(255, (1.0 + sigma) * v))

    edges = cv2.Canny(imgb, lower_thresh, upper_thresh)
    edgesimg2 = PIL.Image.fromarray(edges)
    edgesimg2.save(image_name + '-edge' + '.jpg')