import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import PIL
#from preprocessing.preprocessingImages.threshCannyEdgeFunction import threshCanny

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
if False:
    lower_thresh = int(max(0, (1.0 - 2 * sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))
'''

#get the edge

edges = cv2.Canny(imgb, lower_thresh, upper_thresh)
#print(edges)
#edgesimg2 = PIL.Image.fromarray(edges)
#edgesimg2.save(image_name + '-threshold3' + '.jpg')

lines = cv2.HoughLines(edges, 20, np.pi/360, 2000)
for i in range(len(lines)):
    for rho,theta in lines[i]:
#for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(imgb,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite('mask-houghlines.jpg',imgb)