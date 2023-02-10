import math

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
img = cv2.imread("Lenna.png", cv2.IMREAD_COLOR)
# from Equalization import *
def func1(img,gamma):
    # scale image into 0,1
    invGamma= 1/ gamma
    table =np.array([((i / 255)**invGamma) for i in np.arange(0,256)])*255
    table=table.astype(np.uint8)
    adjusted=cv2.LUT(img, table)
    cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    return adjusted
def sp_noise(image,prob):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output
def on_track_bar(val):
    pass
def calculate_target_size(img_size: int, kernel_size: int) -> int:
    num_pixels = 0
    # From 0 up to img size (if img size = 224, then up to 223)
    for i in range(img_size):
        # Add the kernel size (let's say 3) to the current i
        added = i + kernel_size
        # It must be lower than the image size
        if added <= img_size:
            # Increment if so
            num_pixels += 1
    return num_pixels
def gau(x,y, sigma):
    return (1/(2*math.pi*sigma**2))*math.exp(-(x**2+y**2)/(2*sigma**2))
def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)
def median_mean_filter(img, kernel_size, mean=None):
    img_new=np.zeros_like(img, dtype=np.uint8)
    padding_width= kernel_size//2
    img_padding= np.zeros(shape=(img.shape[0] + padding_width*2, img.shape[1] + padding_width*2))
    img_padding[padding_width:-padding_width,padding_width:-padding_width]= img
    tag_size= calculate_target_size(img_padding.shape[0],kernel_size)
    # gaussian_smoothing=np.array([[1,2,1],[2,4,2],[1,2,1]])*1/16
    # print(gaussian_smoothing)
    if mean==None:
        simga = float(input('sigma:'))
        gaussian_smoothing = gkern(l=kernel_size,sig=simga)
    for x in range(tag_size):
        for y in range(tag_size):
            if mean != None:
                kernel= sorted(np.array(img_padding[y:y+kernel_size, x:x+kernel_size]).ravel())
                if mean!=True:
                    num= kernel[len(kernel)//2]
                else:
                    num= np.mean(kernel)
                img_new[y][x]=num
            elif mean is None:
                kernel=np.array(img_padding[y:y+kernel_size, x:x+kernel_size])
                img_new[y][x] = np.sum(np.multiply(kernel, gaussian_smoothing))
    return img_new
def change(val):
    if val >10:
        new= val%10
        return new
    elif val < 10:
        new= val-10
        return new
    else:
        return 1
def hist_cum_hist(img,y):
    for ch in range(img.shape[2]):
        for row in range(img.shape[0]):
            for col in range(img.shape[1]):
                y[ch,img[row, col, ch]] += 1
    cum_hist = np.zeros(y.shape, np.float64)
    cum_hist[0,0] = y[0,0]
    cum_hist[1,0] = y[1,0]
    cum_hist[2,0] = y[2,0]
    for ch in range(img.shape[2]):
        for i in range(1, cum_hist.shape[1]):
            cum_hist[ch,i] = cum_hist[ch,i - 1] + y[ch,i]
    return y, cum_hist
def drawHistogram(img, title=('b','g','r')):
    fig, ax = plt.subplots(3,1)
    x = np.linspace(0, 255, 256)
    y = np.zeros((3,256), np.float64)
    y, cum_hist= hist_cum_hist(img, y)
    for ch in range(3):
        max_hist = np.float64(np.max(y[ch,:]))
        max_cum_hist = np.float64(np.max(cum_hist[ch,:]))
        # Nomarlization for drawing
        y[ch,:] *= (255. / max_hist)
        cum_hist[ch,:] *= (255. / max_cum_hist)
        ax[ch].bar(x, y[ch,:], color=title[ch], label="histogram")
        ax[ch].plot(x, cum_hist[ch,:], color='k', label="Cummulate histogram")
        ax[ch].legend(["Cummulate histogram", "histogram"])
        ax[ch].set_title("Histogram " + title[ch])
    return ax
def fft(image):
    image=image.copy()
    dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(image))
    r= np.log(abs(dark_image_grey_fourier))
    return r
def histogram_equalization(img, bShow=False):
    y = np.zeros((256,), np.float64)
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            y[img[row, col]] += 1
    cum_hist= np.zeros_like(y)
    cum_hist[0]=y[0]
    for i in range(1, len(cum_hist)):
        cum_hist[i] = cum_hist[i - 1] + y[i]
    max_cum_hist = np.max(cum_hist) # row
    min_cum_hist = np.min(cum_hist) # row
    # 3.create map for old pixel values
    hist_map = np.zeros(shape=y.shape,dtype= np.uint8)
    for i in range(len(cum_hist)):
        hist_map[i] = np.uint8((cum_hist[i] - min_cum_hist)/(max_cum_hist-min_cum_hist) * 255)
    #4.create histogram equalised image
    new_img = img.copy()
    for row in range(new_img.shape[0]):
        for col in range(new_img.shape[1]):
            new_img[row, col] = hist_map[img[row, col]]
    return new_img
def run1():
    global img
    while True:
        print("1:color balance\n2:histogram equalization\n3:median filter\n4:mean filter\n5:Gaussian smoothing\n6:Fourier_Transform\n7:exit")
        f=input('input function:')
        if f =='1':
            cv2.namedWindow('result')
            cv2.createTrackbar('gamma','result',10,100,on_track_bar)
            cv2.setTrackbarMax('gamma', 'result', 20)
            while 1:
                img_c = img.copy()
                i=int(cv2.getTrackbarPos('gamma','result'))
                v=change(i)
                img_c=func1(img_c,v)
                cv2.imshow('result',img_c)
                k = cv2.waitKey(1) & 0xFF
                if k == 27:
                    break
            cv2.destroyWindow('result')
        if f=='2':
            cv2.namedWindow('result')
            a1=drawHistogram(img)
            img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:,:,2]=histogram_equalization(img_hsv[:,:,2], bShow=False)
            img_bgr=cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            img_total=np.hstack((img,img_hsv,img_bgr))
            a2=drawHistogram(img_bgr)
            cv2.imshow('result',img_total)
            plt.show()
        if f=='3':
            gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_noise=sp_noise(gray,0.05)
            filtered=median_mean_filter(gray_noise,3, mean=False)
            total=np.hstack((gray_noise,filtered))
            cv2.imshow('result', total)
            cv2.waitKey(0)
        if f=='4':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_noise = sp_noise(gray, 0.05)
            filtered = median_mean_filter(gray_noise, 3, mean=True)
            total = np.hstack((gray_noise, filtered))
            cv2.imshow('result', total)
            cv2.waitKey(0)
        if f=='5':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            ex= img.copy()
            filtered=gray.copy()
            filtered[:,:,2] = median_mean_filter(gray[:,:,2], 15, mean=None)
            filtered= cv2.cvtColor(filtered, cv2.COLOR_HSV2BGR)
            total = np.hstack((ex, filtered))
            cv2.imshow('result', total)
            cv2.waitKey(0)
        if f=='6':
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_noise = sp_noise(gray, 0.05)
            filtered = median_mean_filter(gray_noise, 3, mean=False)
            gray_noise_fft= fft(gray_noise)
            filtered_fft= fft(filtered)
            total = np.hstack((gray_noise_fft, filtered_fft))
            total1 = np.hstack((gray_noise, filtered))
            plt.imshow(total, cmap='gray')
            cv2.imshow('1', total1)
            plt.show()
            cv2.waitKey(0)
        if f=='7':
            break
run1()

