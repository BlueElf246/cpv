import numpy as np
import cv2
import glob
import imutils
import os
import matplotlib.pyplot as plt
import imageio
cv2.ocl.setUseOpenCL(False)
import warnings
os.chdir("/Users/datle/Desktop/CPV/cpv/workshop6/images")
images_path= glob.glob('*.JPG')
images=[]

feature_extraction_algo = 'sift'

feature_to_match = 'bf'

for img in images_path:
    print(img)
    img1= cv2.imread(img)
    images.append(img1)
    # cv2.imshow('image', img1)
    # cv2.waitKey(0)

train_photo= images[1]

train_photo = cv2.cvtColor(train_photo,cv2.COLOR_BGR2RGB)

train_photo_gray = cv2.cvtColor(train_photo, cv2.COLOR_RGB2GRAY)

query_photo= images[0]
query_photo = cv2.cvtColor(query_photo,cv2.COLOR_BGR2RGB)
query_photo_gray = cv2.cvtColor(query_photo, cv2.COLOR_RGB2GRAY)

def show1():
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, constrained_layout=False, figsize=(16,9))
    ax1.imshow(query_photo, cmap="gray")
    ax1.set_xlabel("Query image", fontsize=14)

    ax2.imshow(train_photo, cmap="gray")
    ax2.set_xlabel("Train image (Image to be transformed)", fontsize=14)
    plt.show()

def select_descriptor_methods(image, method=None):
    assert method is not None, "Please define a feature descriptor method. accepted Values are: 'sift', 'surf'"

    if method == 'sift':
        descriptor = cv2.SIFT_create()
    elif method == 'surf':
        descriptor = cv2.SURF_create()
    elif method == 'brisk':
        descriptor = cv2.BRISK_create()
    elif method == 'orb':
        descriptor = cv2.ORB_create()

    (keypoints, features) = descriptor.detectAndCompute(image, None)

    return (keypoints, features)


keypoints_train_img, features_train_img = select_descriptor_methods(train_photo_gray, method=feature_extraction_algo)

keypoints_query_img, features_query_img = select_descriptor_methods(query_photo_gray, method=feature_extraction_algo)

# for keypoint in keypoints_query_img:
#     x, y = keypoint.pt
#     size = keypoint.size
#     orientation = keypoint.angle
#     response = keypoint.response
#     octave = keypoint.octave
#     class_id = keypoint.class_id
#
#     print(x, y)
#
#     print(size)
#
#     print(orientation)
#
#     print(response)
#     print(octave)
#     print(class_id)

# display the keypoints and features detected on both images
def show2():
    fig, (ax1,ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20,8), constrained_layout=False)

    ax1.imshow(cv2.drawKeypoints(train_photo_gray, keypoints_train_img, None, color=(0,255,0)))

    ax1.set_xlabel("(a)", fontsize=14)

    ax2.imshow(cv2.drawKeypoints(query_photo_gray,keypoints_query_img, None,color=(0,255,0)))
    ax2.set_xlabel("(b)", fontsize=14)
    plt.show()


def create_matching_object(method, crossCheck):
    "Create and return a Matcher Object"

    # For BF matcher, first we have to create the BFMatcher object using cv2.BFMatcher().
    # It takes two optional params.
    # normType - It specifies the distance measurement
    # crossCheck - which is false by default. If it is true, Matcher returns only those matches
    # with value (i,j) such that i-th descriptor in set A has j-th descriptor in set B as the best match
    # and vice-versa.
    if method == 'sift' or method == 'surf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
    elif method == 'orb' or method == 'brisk':
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
    return bf


def key_points_matching(features_train_img, features_query_img, method):
    bf = create_matching_object(method, crossCheck=True)

    # Match descriptors.
    best_matches = bf.match(features_train_img, features_query_img)

    # Sort the features in order of distance.
    # The points with small distance (more similarity) are ordered first in the vector
    rawMatches = sorted(best_matches, key=lambda x: x.distance)
    print("Raw matches with Brute force):", len(rawMatches))
    return rawMatches

def key_points_matching_KNN(features_train_img, features_query_img, ratio, method):
    bf = create_matching_object(method, crossCheck=False)
    # compute the raw matches and initialize the list of actual matches
    rawMatches = bf.knnMatch(features_train_img, features_query_img, k=2)
    print("Raw matches (knn):", len(rawMatches))
    matches = []

    # loop over the raw matches
    for m,n in rawMatches:
        # ensure the distance is within a certain ratio of each
        # other (i.e. Lowe's ratio test)
        if m.distance < n.distance * ratio:
            matches.append(m)
    return matches

print("Drawing: {} matched features Lines".format(feature_to_match))

fig = plt.figure(figsize=(20, 8))

if feature_to_match == 'bf':
    matches = key_points_matching(features_train_img, features_query_img, method=feature_extraction_algo)

    mapped_features_image = cv2.drawMatches(train_photo, keypoints_train_img, query_photo, keypoints_query_img,
                                            matches[:100],
                                            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Now for cross checking draw the feature-mapping lines also with KNN
elif feature_to_match == 'knn':
    matches = key_points_matching_KNN(features_train_img, features_query_img, ratio=0.75,
                                      method=feature_extraction_algo)

    mapped_features_image_knn = cv2.drawMatches(train_photo, keypoints_train_img, query_photo, keypoints_query_img,
                                                np.random.choice(matches, 100),
                                                None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

#plt.imshow(mapped_features_image)
# plt.axis('off')
# plt.savefig("./output/" + feature_to_match + "_matching_img_"+'.jpeg', bbox_inches='tight',
#             dpi=300, optimize=True, format='jpeg')


def homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh):
    """ converting the keypoints to numpy arrays before passing them for calculating Homography Matrix.

    Because we are supposed to pass 2 arrays of coordinates to cv2.findHomography, as in I have these points in image-1, and I have points in image-2, so now what is the homography matrix to transform the points from image 1 to image 2
    """
    keypoints_train_img = np.float32([keypoint.pt for keypoint in keypoints_train_img])
    keypoints_query_img = np.float32([keypoint.pt for keypoint in keypoints_query_img])

    ''' For findHomography() - I need to have an assumption of a minimum of correspondence points that are present between the 2 images. Here, I am assuming that Minimum Match Count to be 4 '''
    if len(matches) > 4:
        # construct the two sets of points
        points_train = np.float32([keypoints_train_img[m.queryIdx] for m in matches])
        points_query = np.float32([keypoints_query_img[m.trainIdx] for m in matches])

        # Calculate the homography between the sets of points
        (H, status) = cv2.findHomography(points_train, points_query, cv2.RANSAC, reprojThresh)

        return (matches, H, status)
    else:
        return None

M = homography_stitching(keypoints_train_img, keypoints_query_img, matches, reprojThresh=4)

if M is None:
    print("Error!")

(matches, Homography_Matrix, status) = M

print(Homography_Matrix)

# For the calculation of the width and height of the final horizontal panoramic images
# I can just add the widths of the individual images and for the height
# I can take the max from the 2 individual images.

width = query_photo.shape[1] + train_photo.shape[1]
print("width ", width)
# 2922 - Which is exactly the sum value of the width of
# my train.jpg and query.jpg


height = max(query_photo.shape[0], train_photo.shape[0])

# otherwise, apply a perspective warp to stitch the images together

# Now just plug that "Homography_Matrix"  into cv::warpedPerspective and I shall have a warped image1 into image2 frame

result = cv2.warpPerspective(train_photo, Homography_Matrix,  (width, height))

# The warpPerspective() function returns an image or video whose size is the same as the size of the original image or video. Hence set the pixels as per my query_photo

result[0:query_photo.shape[0], 0:query_photo.shape[1]] = query_photo

plt.figure(figsize=(20,10))
plt.axis('off')
plt.imshow(result)

imageio.imwrite("horizontal_panorama_img_"+'.jpeg', result)

plt.show()