import cv2
import os
import numpy as np
import math
def rotate_img(Pt, alpha, center=None):
    Pt=np.array(Pt)
    Pt=Pt.reshape(2,1)
    sin= math.sin(alpha)
    cos= math.cos(alpha)
    M=np.array([[cos, -sin],
                [sin,cos]])
    result= np.dot(M,Pt)
    print(result)
def transalate_img(Pt, t):
    Pt.append(1)
    Pt = np.array(Pt)
    Pt = Pt.reshape(3, 1)
    M = np.array([[1, 0, t[0]],
                  [0, 1, t[1]]]).astype("float")
    result= np.dot(M,Pt)
    print(result[0][0])
def scale_img(Pt, s):
    Pt = np.array(Pt)
    Pt = Pt.reshape(2, 1)
    M= np.array([[s[0],0],
                 [0,s[1]]])
    result= np.dot(M,Pt)
    print(result)
img= [1,2]
rotate_img(img, 60)
img1=[1,2]
transalate_img(img1,[4,2])
img2=[1,2]
scale_img(img2,[2,3])
