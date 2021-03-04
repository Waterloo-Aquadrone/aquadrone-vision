import numpy as np
import cv2 as cv

def main():
    #Read image
    img = cv.imread("ImageFile")

    #Convert RGB to HSV (makes creating range of colour easier)
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    #Define Colour Range
    lower_orange = np.array([0, 0, 50])
    upper_orange = np.array([18, 200, 255])

    #Create mask and resulting image
    mask = cv.inRange(hsv_img, lower_orange, upper_orange)
    res = cv.bitwise_and(img, img, mask=mask)

    #Show images
    cv.imshow("Image", img)
    cv.imshow("Mask", mask)
    cv.imshow("Result", res)

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
