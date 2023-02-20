import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import data, exposure
def read_img(img_path):
    img=cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img
def harris_corner(img, thresh, block_size, k_size, k):
    img=img.copy()
    img_gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray= np.array(img_gray).astype('float32')
    # Detect corners

    #blocksize: blockSize: Neighborhood size for the corner detection algorithm. It is the size of the window used to detect the corners.
    # related to size of w(x,y)
    """Larger values of blockSize result in a larger window being considered, 
    which can reduce the impact of noise and provide a more stable response measure. 
    Smaller values of blockSize result in a smaller window being considered, 
    which can improve the spatial resolution of the algorithm and provide more precise locations of the corners."""
    #k_size: Aperture parameter for the Sobel operator, 3 mean using 3x3 sobel kernel
    # k:Harris detector free parameter
    k=k/100
    if k_size % 2==0:
        k_size+=1
    dst = cv2.cornerHarris(src=img_gray, blockSize=block_size, ksize=k_size, k=k)
    # Dilate corner image to enhance corner points
    dst = cv2.dilate(dst, None)
    a_1 = thresh / 100
    print(f"thresh: {a_1}, block_size: {block_size}, k_size:{k_size}, k: {k}")
    # X=np.array(np.where(dst > a_1 * dst.max())).T
    # kmeans = KMeans(n_clusters=4, random_state=0, n_init="auto").fit(X)
    thresh=a_1 * dst.max()
    for j in range(0, dst.shape[0]):
        for i in range(0, dst.shape[1]):
            if (dst[j, i] > thresh):
                # image, center pt, radius, color, thickness
                # r= kmeans.predict(np.array([j,i]).reshape(1,-1))
                # if r[0]==1:
                #     cv2.circle(img, (i, j), 1, (0, 255, 0), 1)
                # if r[0]==0:
                #     cv2.circle(img, (i, j), 1, (0, 0, 255), 1)
                # if r[0]==2:
                #     cv2.circle(img, (i, j), 1, (255, 0, 0), 1)
                # if r[0]==3:
                #     cv2.circle(img, (i, j), 1, (255, 0, 255), 1)
                cv2.circle(img, (i, j), 1, (255, 0, 255), 1)
    # img[dst > a_1 * dst.max()] = [0, 0, 255]
    # plt.imshow(dst, cmap='gray')
    # cv2.imshow('result', img)
    # cv2.waitKey(0)
    return img
def shi_tomasi(image,a,b,c):
    # Converting to grayscale
    image=image.copy()
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners_img = cv2.goodFeaturesToTrack(image=gray_img, maxCorners=a, qualityLevel=0.09, minDistance=c)
    corners_img = np.int0(corners_img)
    for corners in corners_img:
        x, y = corners.ravel()
        # Circling the corners in green
        cv2.circle(image, (x, y), 3, [0, 255, 0], -1)
    # cv2.imshow('result', image)
    # Specifying maximum number of corners as 1000
    # 0.01 is the minimum quality level below which the corners are rejected
    # 10 is the minimum euclidean distance between two corners
    #image, maxCorners, qualityLevel, minDistance
    # cv2.imshow('result', image)
    # cv2.waitKey(0)
    return image
def get_harris_params():
    return cv2.getTrackbarPos('thresh_hold', 'result'), cv2.getTrackbarPos('block_size', 'result'), cv2.getTrackbarPos('k_size', 'result'), cv2.getTrackbarPos('k', 'result')
def harris(a):
    thresh, block_size, k_size, k=get_harris_params()
    print(thresh, block_size, k_size, k)
    img_1 = harris_corner(img, thresh, block_size, k_size, k)
    cv2.imshow('result', img_1)
def hog_i(image):

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, channel_axis=-1)
    print(fd)
    print(len(fd))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
def Canny_callback(thresh):
    thresh1 = cv2.getTrackbarPos('thresh1', 'canny')
    thresh2 = cv2.getTrackbarPos('thresh2', 'canny')
    apterture = cv2.getTrackbarPos('aperture', 'canny')
    edges = cv2.Canny(img, thresh1, thresh2, apertureSize=apterture)
    cv2.imshow('canny', edges)
def hough(img, thresh=10, rho=1, low_thresh=50, high_thresh=100):
    img= img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, low_thresh, high_thresh)
    line_image = np.copy(img)  # creating an image copy to draw lines on
    # Run Hough on the edge-detected image
    # Apply Hough Transform to detect lines
    lines = cv2.HoughLines(edges, rho=rho, theta=np.pi / 180, threshold=thresh)
    # Draw the detected lines on the original image
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    edges= cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return np.vstack((line_image,edges))
def get_hough_para():
    return cv2.getTrackbarPos('thresh','hough'), cv2.getTrackbarPos('rho', 'hough'), cv2.getTrackbarPos('low_thresh', 'hough'), cv2.getTrackbarPos('high_thresh','hough')
def hough_thresh(a):
    thresh, rho, low, high= get_hough_para()
    img_2=hough(img,thresh=thresh, rho=rho, low_thresh=low, high_thresh=high)
    cv2.imshow('hough', img_2)
while True:
    print("""
    1: Perform Harris Corner
    2: Perform HOG(Histogram of Oriented Gradients)
    3: Perform Canny Edge (edge detection)
    4: Perform Hough Transform
    5: Exit
    """)
    # f = int(input('choose option'))
    f=1
    if f==1:
        print('***Harris Corner***')
        cv2.namedWindow('result')
        # cv2.createTrackbar('qualityLevel', 'result', 10, 10, change_color)
        # cv2.createTrackbar('minDistance', 'result', 10, 1000, change_color)
        img= read_img(img_path="img_data/Cars108.png")
        cv2.createTrackbar('thresh_hold', 'result', 1, 100, harris)
        cv2.createTrackbar('block_size', 'result', 1, 10, harris)
        cv2.createTrackbar('k_size', 'result', 1, 10, harris)
        cv2.createTrackbar('k', 'result', 4, 10, harris)
        thresh, block_size, k_size, k = get_harris_params()
        img_1 = harris_corner(img, thresh, block_size, k_size, k)
        cv2.imshow('result', img_1)
        cv2.waitKey(0)
        # harris_corner(img)
        # shi_tomasi(img)
    if f==2:
        img= read_img(img_path="img_data/Cars108.png")
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        hog_i(img)
    if f==3:
        img = read_img(img_path="img_data/Cars108.png")
        cv2.namedWindow('canny')
        cv2.createTrackbar('thresh1','canny',1,255, Canny_callback)
        cv2.createTrackbar('thresh2', 'canny',1,255,Canny_callback)
        cv2.createTrackbar('aperture', 'canny',3,7, Canny_callback)
        thresh1 = cv2.getTrackbarPos('thresh1', 'canny')
        thresh2 = cv2.getTrackbarPos('thresh2', 'canny')
        apterture = cv2.getTrackbarPos('aperture', 'canny')
        edges = cv2.Canny(img, thresh1, thresh2, apertureSize=apterture)
        cv2.imshow('canny', edges)
        cv2.waitKey(0)
    if f==4:
        print('Hough Transform')
        cv2.namedWindow('hough')
        img = read_img(img_path="img_data/Cars108.png")
        cv2.createTrackbar('thresh', 'hough', 50,100,hough_thresh)
        cv2.createTrackbar('rho', 'hough', 1,3, hough_thresh)
        cv2.createTrackbar('low_thresh', 'hough',1,255, hough_thresh)
        cv2.createTrackbar('high_thresh','hough', 1,255, hough_thresh)
        # perform canny edge detection
        img_1=hough(img, thresh=100,rho=1)
        cv2.imshow('hough', img_1)
        cv2.waitKey(0)
    if f==5:
        break
