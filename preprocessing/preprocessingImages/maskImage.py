import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt

# white color mask
img = cv2.imread("dices-2164.jpg")
#converted = convert_hls(img)
image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
lower = np.uint8([70, 150, 150])
upper = np.uint8([200, 255, 255])
white_mask = cv2.inRange(image, lower, upper)
# yellow color mask
lower = np.uint8([10, 0,   100])
upper = np.uint8([40, 255, 255])
yellow_mask = cv2.inRange(image, lower, upper)
# combine the mask
mask = cv2.bitwise_or(white_mask, yellow_mask)
result = img.copy()
cv2.imwrite("mask.jpg",white_mask)