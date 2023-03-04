# step for image segmentation using k-mean clustering
import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
    source:
    https://www.kdnuggets.com/2019/08/introduction-image-segmentation-k-means-clustering.html
    https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html
    https://www.youtube.com/watch?v=22mpExWh1LY
    
    1:Convert image to HSV color space
    2: reshape the image into shape (K,3), where K=MxN, M,N are number of rows, cols respectively.
    -> after transform, each row will represent a pixel that has 3 channel value
    3: convert from uint8 to float
    4: use cv2.kmeans for color clustering
    ->cv2.kmeans(samples, nclusters(K), criteria, attempts, flags)
        1. samples: It should be of np.float32 data type, and each feature should be put in a single column.

        2. nclusters(K): Number of clusters required at the end

        3. criteria: It is the iteration termination criteria. When this criterion is satisfied, the algorithm iteration stops. Actually, it should be a tuple of 3 parameters. They are `( type, max_iter, epsilon )`:

        Type of termination criteria. It has 3 flags as below:

        cv.TERM_CRITERIA_EPS — stop the algorithm iteration if specified accuracy, epsilon, is reached.
        cv.TERM_CRITERIA_MAX_ITER — stop the algorithm after the specified number of iterations, max_iter.
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER — stop the iteration when any of the above condition is met.
        4. attempts: Flag to specify the number of times the algorithm is executed using different initial labelings. The algorithm returns the labels that yield the best compactness. This compactness is returned as output.

        5. flags: This flag is used to specify how initial centers are taken. Normally two flags are

    5: the above step will return ret,label,center
        compactness : It is the sum of squared distance from each point to their corresponding centers.
        labels : This is the label array (same as 'code' in previous article) where each element marked '0', '1'.....
        centers : This is array of centers of clusters.
"""


"""
Step in k-mean clustering:

step1: Randomly generate the initial centroid(means) of the three clusters (here K=3)
step2: Create 3 clusters by assign each feature point to the nearest mean
step3: Recompute the mean of each cluster
step4: Repeat step2 and step3 until converge

some method for initialization centroid in step1:
method1: select k random feature points as initial centroid, if two point close to each other, resample
method2: select k uniformly distributed means within the distribution
method3(prefered):  do k-mean clustering on subset data and use the result as inital mean
"""

"""
How to choose number of clustering:
    One of the most popular is elbow method: 
        the x-axis is the range of k value
        the y-axis is Distortion: this is the distance metric from center centroid to all point belong to that cluster
How to improve the performance of model:
    Using additionally x,y(the coordinate of pixels) beside the 3 channel for a feature vector
    
some comment:
    1/sensitive to outlier
    2/sensitive to initialization
"""
def k_means_segment(path, K=3):
    img=cv2.imread(path)
    # conver to hsv color space
    img_hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # vectorized img
    vectorized = img_hsv.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    attempts = 10
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))
    figure_size = 5
    plt.figure(figsize=(figure_size, figure_size))
    plt.subplot(1, 2, 1), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(result_image)
    plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
    plt.show()
