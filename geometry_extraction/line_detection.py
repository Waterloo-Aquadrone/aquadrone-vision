import numpy as np
import cv2 as cv

def main():
    #Duplicating colour filtering code for now, will change to pass image through funciton
    # Read image
    img = cv.imread("gate1.png")

    # Convert RGB to HSV (makes creating range of colour easier)
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define Colour Range
    lower_orange = np.array([0, 0, 50])
    upper_orange = np.array([18, 200, 255])

    # Create mask
    mask = cv.inRange(hsv_img, lower_orange, upper_orange)

    #Create minimum area bounding box for mask
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rect = cv.minAreaRect(contours[-1])
    box = cv.boxPoints(rect)
    box = np.int0(box)

    #Create skeleton of mask
    mask = cv.GaussianBlur(mask, (17, 17), 0)
    kernel = np.ones((20, 20), np.uint8)
    mask = cv.erode(mask, kernel, iterations=2)
    size = np.size(mask)
    skel = np.zeros(mask.shape, np.uint8)
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv.erode(mask, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(mask, temp)
        skel = cv.bitwise_or(skel, temp)
        mask = eroded.copy()
        zeros = size - cv.countNonZero(mask)
        if zeros == size:
            done = True

    #Generate Hough lines from skeleton
    edges = cv.Canny(skel, 250, 250)
    lines = cv.HoughLinesP(edges, rho=1, theta=np.radians(1), threshold=100, maxLineGap=500)

    dists = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        a = np.array((x1, y1))
        b = np.array((x2, y2))
        dist = np.linalg.norm(a-b)
        dists.append(dist)

    #Determine largest hough line generated
    largestLine = dists.index(max(dists))

    #Extend largest hough line and find intersection with minimum area bounding box
    m1 = (lines[largestLine][0][1] - lines[largestLine][0][3])/(lines[largestLine][0][0] - lines[largestLine][0][2])
    m2 = (box[0][1] - box[3][1])/(box[0][0] - box[3][0])
    m3 = (box[1][1] - box[2][1])/(box[1][0] - box[2][0])
    b1 = lines[largestLine][0][1] - (m1*lines[largestLine][0][0])
    b2 = box[0][1] - (m2*box[0][0])
    b3 = box[1][1] - (m3*box[1][0])

    x1 = (b2-b1)/(m1-m2)
    x2 = (b3-b1)/(m1-m3)
    y1 = (m1*x1) + b1
    y2 = (m1*x2) + b1

    #Check if any point is going out of frame
    if x1 < 0:
        x1 = 0
    if x2 < 0:
        x2 = 0
    if y1 < 0:
        y1 = 0
    if y2 < 0:
        y2 = 0

    #Print coordinates of line
    print(x1, y1, x2, y2)

    #Draw line
    cv.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)

    #Show resulting image with line detected
    cv.imshow("Image", img)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
