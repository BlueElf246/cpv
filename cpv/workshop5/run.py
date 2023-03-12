import cv2
import numpy as np

MAX_FEATURES = 500
def align_image(img, template, max_feature, keep_percentage):
    # convert image to gray
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
    height, width, channel = template.shape
    img1_align= cv2.warpPerspective(img, h, (width, height))
    cv2.imshow('after aligned', img1_align)
    cv2.waitKey(0)




img1= cv2.imread("images/image1.jpg", cv2.IMREAD_COLOR)
img2= cv2.imread("images/image2.jpg", cv2.IMREAD_COLOR)
align_image(img1, img2, MAX_FEATURES, 0.5)