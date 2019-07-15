import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

img = cv2.imread('gate6.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([0, 70, 70])
upper_yellow = np.array([70, 255, 255])

mask = cv2.inRange(img, lower_yellow, upper_yellow) # applies yellow mask
res = cv2.bitwise_and(img, img, mask= mask)


cv2.imshow('Original',img)
cv2.imshow('Mask',mask)
cv2.imshow('Res',res)

cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------------------------
blur = cv2.blur(mask,(10,10)) # blurs image
cv2.imwrite("blur.png", blur)

im = cv2.imread('blur.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in range (len(contours)): # draws contours and corners
    cnt = contours[i]
    cv2.drawContours(im, contours, -1, (255,0,0), 3)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    for j in range(4):
      cv2.circle(im, (box[j][0], box[j][1]), 5, (0, 0, 255), -1)

cv2.imshow("contours", im)


#
# _, threshold = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
# contours, _=cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#
# for cnt in contours:
#     approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
#     cv2.drawContours(img, [approx], 0, (0), 5)
#     x = approx.ravel()[0]
#     y = approx.ravel()[1]
#     if len(approx) == 3:
#         cv2.putText(img, "Triangle", (x, y), font, 1, (0))
#     elif len(approx) == 4:
#         cv2.putText(img, "Rectangle", (x, y), font, 1, (0))
#     elif len(approx) == 5:
#         cv2.putText(img, "Pentagon", (x, y), font, 1, (0))
#     elif len(approx) == 6:
#         cv2.putText(img, "Pentagon", (x, y), font, 1, (0))
#     elif 6 < len(approx) < 15:
#         cv2.putText(img, "Ellipse", (x, y), font, 1, (0))
#     else:
#         cv2.putText(img, "Circle", (x, y), font, 1, (0))
#
# cv2.imshow("shapes", img)
# cv2.imshow("Threshold", threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()





plt.show()
