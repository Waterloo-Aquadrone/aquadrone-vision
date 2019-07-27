import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

img = cv2.imread('gate6.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lower_yellow = np.array([0, 70, 70])
upper_yellow = np.array([70, 255, 255])

mask = cv2.inRange(img, lower_yellow, upper_yellow)
res = cv2.bitwise_and(img, img, mask= mask)


cv2.imshow('Original',img)
cv2.imshow('Mask',mask)
cv2.imshow('Res',res)

cv2.waitKey(0)
cv2.destroyAllWindows()

blur = cv2.blur(mask,(10,10))
cv2.imwrite("blur.png", blur)

im = cv2.imread('blur.png')
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,127,255,0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(im, contours, -1, (255, 0, 0), 3)

all_pixels = []

for i in range (len(contours)):
    cnt = contours[i]


    for i in range(0, len(cnt)):
        for j in range(0, len(cnt[i])):
            all_pixels.append(cnt[i][j])

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    for j in range(4):
      cv2.circle(im, (box[j][0], box[j][1]), 5, (0, 0, 255), -1)


#for i in range(0, len(all_pixels), 5):
#    cv2.circle(im, (all_pixels[i][0], all_pixels[i][1]), 5, (255, 255, 0), -1)

topl = all_pixels[0]
topr = all_pixels[0]
botr = all_pixels[0]
botl = all_pixels[0]
top = all_pixels[0]
bot = all_pixels[0]
right = all_pixels[0]
left = all_pixels[0]

for i in all_pixels:
    if i[0] <= topl[0] and i[1]<= topl[1]:
        topl = i
    if i[0] >= topr[0] and i[1] <= topr[1]:
         topr = i
    if i[0] <= botl[0] and i[1] >= botl[1]:
        botl = i
    if i[0] >= botr[0] and i[1] >= botr[1]:
         botr = i
    if i[0] <= left[0]:
        left = i
    if i[0] >= right[0]:
        right = i
    if i[1] <= top[1]:
        top = i
    if i[1] >= bot[1]:
        bot = i

cv2.circle(im, (left[0], left[1]), 5, (0, 255, 255), -1)
cv2.circle(im, (right[0], right[1]), 5, (0, 255, 255), -1)
cv2.circle(im, (top[0], top[1]), 5, (0, 255, 255), -1)
cv2.circle(im, (bot[0], bot[1]), 5, (0, 255, 255), -1)

cv2.circle(im, (topl[0], topl[1]), 5, (0, 255, 0), -1)
cv2.circle(im, (topr[0], topr[1]), 5, (0, 255, 0), -1)
cv2.circle(im, (botl[0], botl[1]), 5, (0, 255, 0), -1)
cv2.circle(im, (botr[0], botr[1]), 5, (0, 255, 0), -1)





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