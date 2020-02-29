import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import PIL

'''
This program help us to determine the parameter for the Bilateral Filtering.
'''
  
# Read the image. 
imgb = cv2.imread('float-0157.jpg') 
  
# Apply bilateral filter with d = 15,  
# sigmaColor = sigmaSpace = 75.

i = 30
j = 70
k_list = [70, 130, 500, 1000, 5000]
while i <= 90:
    j = 70
    while j <= 90:        
        for k in k_list:
            bilateral = cv2.bilateralFilter(imgb, i, j, k)   
            cv2.imwrite('test.jpg', bilateral)
            img = cv2.imread('test.jpg', 0)
            edges = cv2.Canny(img, 0, 100)
            edgesimg2 = PIL.Image.fromarray(edges)
            edgesimg2.save(str(i) + ', ' + str(j) + ', ' + str(k) + '.jpg')
            #cv2.imwrite(str(i) + ', ' + str(j) + ', ' + str(k) + '.jpg', bilateral)           
        j = j + 5    
    i = i + 10
  
# Save the output. 
