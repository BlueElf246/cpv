import matplotlib.pyplot as plt

from active_contour import morphsnakes, example
from workshop4.watershed.run import water_shed
from workshop4.kmean_clusters.run import k_means_segment
from workshop4.mean_shift.run import mean_shift
import cv2
import sklearn
def menu():
    f= int(input('choose function:'))
    if f==1:
        print('active contour')
        # for more detail see: https://github.com/pmneila/morphsnakes
        img= cv2.imread("active_contour/download.jpeg")
        img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)/255.0
        callback = example.visual_callback_2d(img)

        # Morphological Chan-Vese (or ACWE)
        morphsnakes.morphological_chan_vese(img, 35,
                                   smoothing=3, lambda1=1, lambda2=1,
                                   iter_callback=callback)
        plt.show()
    if f==2:
        print('watershed')
        water_shed("/Users/datle/Desktop/CPV/cpv/workshop4/watershed/water_coins.jpeg")
    if f==3:
        print('k-means-segmentation')
        k_means_segment("/Users/datle/Desktop/CPV/cpv/workshop4/kmean_clusters/ocean.jpeg", K=2)
    if f==4:
        print("mean-shift segmentation")
        mean_shift("/Users/datle/Desktop/CPV/cpv/workshop4/kmean_clusters/ocean.jpeg")

menu()