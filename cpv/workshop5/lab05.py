# Feature base alignment

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def take_images():
    cap = cv.VideoCapture(0)
    captured = 0
    # if captured = 2 break
    while True:
        ret, frame = cap.read()
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        if cv.waitKey(1) & 0xFF == ord('c'):
            captured += 1
            cv.imwrite('images/img' + str(captured) + '.jpg', frame)
            print('Image ' + str(captured) + ' captured')
        if captured == 2:
            break
    cap.release()
    cv.destroyAllWindows()


def preprocess_images():
    img1 = cv.imread('images/img1.jpg')
    img2 = cv.imread('images/img2.jpg')
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    return img1, img2


def feature_base_alignment(img1, img2):
    # Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    # good matches
    good = int(len(matches) * 0.25)
    matches = matches[:good]
    # Draw first 10 matches.
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.imshow(img3)
    plt.show()
    # Get the location of the best match in the original image
    img_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    # Get the location of the best match in the feature object
    obj_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    # extract the object's rotation and translation
    M, mask = cv.findHomography(obj_pts, img_pts, cv.RANSAC, 5.0)
    # warp the image
    dst = cv.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(img1, cmap='gray')
    ax[0].set_title('Image 1')
    ax[1].imshow(img2, cmap='gray')
    ax[1].set_title('Image 2')
    ax[2].imshow(dst, cmap='gray')
    ax[2].set_title('Aligned image')
    plt.show()


def main():
    # take_images()
    img1, img2 = preprocess_images()
    feature_base_alignment(img1, img2)


if __name__ == '__main__':
    main()
