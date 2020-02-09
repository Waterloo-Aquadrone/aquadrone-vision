import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import PIL

imgb = cv2.imread('float-0157.jpg')

i = 50
j = 85
k = 1000

bilateral = cv2.bilateralFilter(imgb, i, j, k)   
cv2.imwrite('test.jpg', bilateral)
img = cv2.imread('test.jpg', 0)
edges = cv2.Canny(img, 0, 100)
edgesimg2 = PIL.Image.fromarray(edges)
edgesimg2.save(str(i) + ', ' + str(j) + ', ' + str(k) + '.jpg')