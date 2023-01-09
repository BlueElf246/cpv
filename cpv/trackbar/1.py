import cv2
import numpy as np
import os
def nothing(x):
    img = cv2.imread('cat.png')
    cv2.putText(img, str(x), (50, 150), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 0, 255))
    cv2.imshow('frame', img)
    if cv2.waitKey(10) == 27:
        pass
def nothing1(x):
    print(x)
os.chdir('/Users/datle/Documents/PycharmProjects 08.55.25/pythonProject/opencv/pic')
img= cv2.imread('cat.png')
cv2.namedWindow("frame")
cv2.createTrackbar("test", "frame", 50, 500, nothing)
cv2.createTrackbar("color/gray", "frame", 0, 2, nothing1)
while True:
    img= cv2.imread('cat.png')
    test = cv2.getTrackbarPos("test", "frame")
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(img, str(test), (50, 150), font, 4, (0, 0, 255))
    s = cv2.getTrackbarPos("color/gray", "frame")
    if s == 0:
        pass
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',img)
    if cv2.waitKey(10) ==27:
        break
