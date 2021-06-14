import cv2
import numpy as np
import sys, argparse
import random as rnd
from functools import partial

def find_center_of_gate(img):
    pass


def find_center_of_blob(thresh):

    print("new function call with threshhold set to {}".format(thresh))
    img_copy = src_img.copy()

    canny_output = cv2.Canny(src_img, thresh, thresh * 2) # canny edge detection
    contours,_ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # best algorithm?
    num_contours = len(contours)

    if (num_contours > 0):
        mu = [None]*num_contours
        for i in range(num_contours):
            mu[i] = cv2.moments(contours[i])

        mass_centers = [None]*num_contours
        for i in range(num_contours):
            # add 1e-5 to avoid division by zero -> opencv tutorial method
            mass_centers[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))

        # drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        #

        #cv2.drawContours(src_img, contours, -1, (0,0,255), 3)

        for i in range(num_contours):
            # color = (rnd.randint(0,256), rnd.randint(0,256), rnd.randint(0,256))
            color = (0, 0, 255)
            color_center = (0, 255, 0)
            cv2.drawContours(img_copy, contours, i, color, 2)
            print(int(mass_centers[i][0]))
            print(int(mass_centers[i][1]))
            print("object {}".format(i))
            cv2.circle(img_copy, (int(mass_centers[i][0]), int(mass_centers[i][1])), 10, color_center, -1)

        cv2.imshow("Image", img_copy)

def main(argv):

    parser = argparse.ArgumentParser(description='Find center of a blob.')
    parser.add_argument('--img', help='Image file name', default='test.jpg')
    args = parser.parse_args()

    try:
        cv2.samples.findFile(args.img, required=True)
    except cv2.error:
        sys.exit()

    global src_img
    src_img = cv2.imread(args.img)

    scale_percent = 50
    width = int(src_img.shape[1] * 0.5)
    height = int(src_img.shape[0] * 0.5)
    new_size = (width, height)
    src_img = cv2.resize(src_img, new_size)

    img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.blur(img_gray, (3,3))
    # figure out what image processing technique is optimal

    src_window = 'Image'
    cv2.namedWindow(src_window)
    cv2.imshow(src_window, src_img)

    max_thresh = 255
    thresh = 100 # initial threshold
    cv2.createTrackbar('Canny Thresh:', src_window, thresh, max_thresh, find_center_of_blob)
    find_center_of_blob(thresh)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1:])
