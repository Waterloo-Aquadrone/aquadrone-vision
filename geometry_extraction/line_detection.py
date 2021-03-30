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

    # Create mask and resulting image
    mask = cv.inRange(hsv_img, lower_orange, upper_orange)
    res = cv.bitwise_and(img, img, mask=mask)

    #Hough lines code - generating too many lines
    """"
    blur = cv.GaussianBlur(mask, (15, 15), 0)
    edges = cv.Canny(blur, 250, 250)

    lines = cv.HoughLinesP(edges, 1, np.radians(1), 20, maxLineGap=250)
    print(len(lines))
    for line in lines:
        print(line)
        x1, y1, x2, y2 = line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    """
    #Determining coordines of non zero pixels
    pixelCoords = cv.findNonZero(mask)
    print("Pixel Coordinates: " + str(pixelCoords))

    #Determine x and y values of coordinates
    xCoords = []
    yCoords = []
    for i in pixelCoords:
        xCoords.append(i[0][0])
        yCoords.append(i[0][1])

    print("x coordinates: " + str(xCoords))
    print("y coordinates: " + str(yCoords))

    #Find line of bestfit
    slope, yInt = np.polyfit(xCoords, yCoords, 1)
    print("Slope: " + str(slope) + "\nY int: " + str(yInt))

    #determine height and width of image
    height, width = mask.shape

    #Generate endpoints for line of bestfit based on edges of image
    linePoints = []

    if ((height - yInt)/slope >= 0 and (height - yInt)/slope <= width ):
        linePoints.append((int((height - yInt)/slope), 0))
    if ((-yInt)/slope >= 0 and (-yInt)/slope <= width):
        linePoints.append((int((-yInt)/slope), height))
    if ((slope * width + yInt) >= 0 and (slope * width + yInt) <= height):
        linePoints.append((width, height-int((slope * width + yInt))))
    if (yInt >= 0 and yInt <= height):
        linePoints.append((0, height-int(yInt)))

    print("Line of bestfit endpoints: " + str(linePoints))

    #Draw line of bestfit
    cv.line(img, linePoints[0], linePoints[1], (255, 0, 0), thickness=5)

    cv.imshow("Mask", mask)
    #cv.imshow("Edges", edges)
    cv.imshow("Image", img)
    #cv.imshow("Blur", blur)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
