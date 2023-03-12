import os
import cv2
import numpy as np
os.chdir("/Users/datle/Desktop/CPV/cpv/workshop6/images3")
img1= cv2.imread("img1.jpg", cv2.IMREAD_COLOR)
img2= cv2.imread("img2.jpg", cv2.IMREAD_COLOR)
img3= cv2.imread("img3.jpg", cv2.IMREAD_COLOR)

def align_image(src, target, max_feature, keep_percentage):
    # convert image to gray
    img= src.copy()
    template= target.copy()
    img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_gray= cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # detecting feature for each image using ORB
    orb= cv2.ORB_create(max_feature)
    k1,d1= orb.detectAndCompute(img_gray, None)
    k2,d2= orb.detectAndCompute(template_gray, None)

    #Match feature
    matcher= cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    print(d1)
    matches= matcher.match(d1, d2, None)
    matches= list(matches)
    # sort match by score
    matches.sort(key= lambda x: x.distance, reverse=False)

    # Remove not so good matches
    print(len(matches))
    num_good_match= int(len(matches) * keep_percentage)
    matches=matches[:num_good_match]
    print(len(matches))

    # Draw top matches
    imMatches= cv2.drawMatches(img, k1, template, k2, matches, None)
    cv2.imshow('result matches', imMatches)


    # Extract locations of good matches
    pt1= np.zeros((len(matches), 2), dtype= np.float32)
    pt2= np.zeros((len(matches), 2), dtype= np.float32)
    for i, match in enumerate(matches):
        pt1[i,:]= k1[match.queryIdx].pt
        pt2[i,:]= k2[match.trainIdx].pt

    #Find the homography
    h, mask= cv2.findHomography(pt1, pt2, cv2.RANSAC)

    #Warping image
    h1, w1= src.shape[:2]
    h2, w2= target.shape[:2]
    img1_align= cv2.warpPerspective(src, h, (w1+w2, h1))
    img1_align[0:h2, 0:w2]= target
    cv2.imshow('after aligned', img1_align)
    cv2.waitKey(0)
    return img1_align



"""Image on the 1st position attach to img in 2nd position"""
i1=align_image(img1, img2, 500, 0.5)
i2= align_image(img3, img1, 500, 0.5)


