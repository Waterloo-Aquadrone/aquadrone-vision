import math
import numpy as np
import cv2 as cv

# helper for filtering points
def distance(pt1, pt2):
        (x1, y1), (x2, y2) = pt1, pt2
        dist = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        return dist


def get_contours(img):
    blockSize = 2     # the size of neighbourhood considered for corner detection
    size = 5          # aperture parameter of Sobel derivative used
    k = 0.07          # Harris detector free parameter in the equation
    thresh = 10       # threshold for point filtering


    # convert the input image into grayscale color space 
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 

    # modify the data type setting to 32-bit floating point and get corners 
    operatedImage = np.float32(grayImage) 
    dst = cv.cornerHarris(operatedImage, blockSize, size, k)

    # finding the cordinates of corners
    bwCorners = np.zeros_like(operatedImage)        
    bwCorners[dst > 0.01*dst.max()] = 255

    coor_array = np.argwhere(bwCorners)
    coor_list = [l.tolist() for l in list(coor_array)]
    coor_tuples = [tuple(l) for l in coor_list]


    # filter out the coordinates that are close, threshold defined above
    # (here I just took the first point, for improvement we can take center point)
    i = 1
    for pt1 in coor_tuples:
        for pt2 in coor_tuples[i::1]:
            if (distance(pt1, pt2) < thresh):
                coor_tuples.remove(pt2)      
        i += 1


    # show the final points on a copy of the image
    
    img2 = img.copy()
    for pt in coor_tuples:
        cv.circle(img2, tuple(reversed(pt)), 3, (0, 0, 255), -1)
    cv.imshow('Image with main corners', img2)
    cv.waitKey(0)
    

    return coor_tuples


img = cv.imread("dices-2164.jpg")
corners = get_contours(img)
print(corners)
