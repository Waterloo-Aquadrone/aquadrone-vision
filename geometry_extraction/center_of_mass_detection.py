import cv2
import numpy as np
import sys, argparse
import random as rnd

def find_center_of_gate(img):
    pass


def find_center_of_blob(thresh, img):

    canny_output = cv2.Canny(img, thresh, thresh * 2) # canny edge detection
    _, contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # best algorithm?
    num_contours = len(contours)

    if (num_contours > 0):
        moments = [None]*num_contours
        for i in range(num_contours):
            moments[i] = cv2.moments(contours[i])

        mass_centers = [None]*num_contours
        for i in range(num_contours):
            # add 1e-5 to avoid division by zero -> opencv tutorial method
            mass_centers[i] = (moments[i]['m10'] / (moments[i]['m00'] + 1e-5), moments[i]['m01'] / (moments[i]['m00'] + 1e-5))

        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

        for i in range(num_contours):
            color = (rnd.randint(0,256), rnd.randint(0,256), rnd.randint(0,256))
            cv2.drawContours(drawing, contours, i, color, 2)
            cv2.circle(drawing, (int(mass_centers[i][0]), int(mass_centers[i][1])), 4, color, -1)


def main(argv):

    parser = argparse.ArgumentParser(description='Find center of a blob.')
    parser.add_argument('--img', help='Image file name', default='test.jpg')
    args = parser.parse_args()

    try:
        cv2.samples.findFile(args.img, required=true)
    except cv.Exception:
        sys.exit()

    src_img = cv2.imread(args.img)
    img_gray = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    img_gray = cv.blur(img_gray, (3,3))
    # figure out what image processing technique is optimal

    src_window = 'Image'
    cv2.namedWindow(src_window)
    cv2.imshow(src_window, src_img)

    max_thresh = 255
    thresh = 0 # initial threshold
    cv.createTrackbar('Canny Thresh:', src_window, thresh, max_thresh, find_center_of_blob)
    find_center_of_blob(thresh, img_gray)

if __name__ == '__main__':
    main(sys.argv[1:])
