import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import PIL

image_name = 'float-0157'
imgb = cv2.imread(image_name + '.jpg')

d = 50
intensity = 85
space = 1000

#possible triplet for the bilaterial filter
#(d, intensity, space) = (50, 85, 500), (50,85,1000), (60, 90, 500), (60, 90, 1000), 
# (60, 90, 5000), (80, 85, 500), (80, 85, 1000), (80, 90, 70), (90, 85, 130)

#filtering
bilateral = cv2.bilateralFilter(imgb, d, intensity, space)   
cv2.imwrite('test.jpg', bilateral)

img = cv2.imread('test.jpg', 0)

#obtain the threshold for Canny using the grayscale image
img_grayscale = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
v = np.median(img_grayscale)
sigma = 0.33
lower_thresh = int(max(0, (1.0-sigma) * v))
upper_thresh = int(min(255, (1 + sigma) * v))
#cv2.imwrite('gray.jpg', img_grayscale)

#get the edge
edges = cv2.Canny(imgb, lower_thresh, upper_thresh)
edgesimg2 = PIL.Image.fromarray(edges)
edgesimg2.save(image_name + '-' + str(d) + ', ' + str(intensity) + ', ' + str(space) + '.jpg')
