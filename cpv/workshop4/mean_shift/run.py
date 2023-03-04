import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
import matplotlib.pyplot as plt


"""
source: https://www.youtube.com/watch?v=PCNz_zttmtA&t=317s

each hill represent a cluster
peak of hill represent "center" of cluster
EACH PIXEL climbs the steepest hill  within its neighborhood
Pixel assigned to the hill (cluster) it climbs

Step for mean_shift:
    1/Compute the centroid within a Windows of size W, can use Simple Mean or Weighted Means
    2/Shift the window to the centroid until convergence
    3/Declare mode and assign it a cluster label
    4/REPEAT FOR ALL PIXEL
"""

"""
Comment: 
    computationally expensive
    find arbitrary number of clusters
    No initialization required
    Robust to outliers
    Clustering depends on window size W

"""
def mean_shift(path):
    img= cv2.imread(path, cv2.IMREAD_COLOR)
    # filter to reduce noise
    img= cv2.medianBlur(img, 3)
    # flatten the image
    flate_image= img.reshape((-1,3))
    flate_image= np.float32(flate_image)
    #mean shift
    bandwidth= estimate_bandwidth(flate_image, quantile=.06, n_samples=3000)
    ms= MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
    ms.fit(flate_image)
    labeled= ms.labels_

    # get number of segments
    segments= np.unique(labeled)
    print(segments.shape[0])

    # get the average color of each segment
    total= np.zeros((segments.shape[0],3), dtype=float)
    count= np.zeros(total.shape, dtype=float)
    for i, label in enumerate(labeled):
        total[label]= total[label]+ flate_image[i]
        count[label]+=1
    avg = total/count
    avg = np.uint8(avg)

    # cast the labeled image into the corresponding average color
    res= avg[labeled]
    result= res.reshape((img.shape))

    plt.imshow(result)
    plt.show()

