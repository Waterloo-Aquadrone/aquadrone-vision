import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import PIL

#Enter the file name
#image_name = input('Enter image file name: ')
image_name = 'dices-1256.jpg'

#Read in the image and use Canny to get the edge
img = cv2.imread(image_name, 0)
edges = cv2.Canny(img, 0, 100)

edgesimg2 = PIL.Image.fromarray(edges)
edgesimg2.save('edge-' + image_name)

'''plot
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.savefig('plot-' + image_name)

#plt.show()
<<<<<<< HEAD
'''
=======
'''
>>>>>>> 3e8374195b8463f3c0d1ac9aa90612d6b8fa6a04
