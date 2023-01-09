import numpy as np
import cv2
import math
ix,iy = -1,-1
drawing=False
stat=False
rec_corr=[ 0 for i in range(4)]

img = np.zeros((1024, 1024, 3), np.uint8)
img[:] = 255, 255, 255
def shortcut1():
    top_left = (rec_corr[1], rec_corr[0])
    bottom_right = (rec_corr[3], rec_corr[2])
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]
    print(width,height)
    indices = get_grid(height, width, True)
    indices[[0], :] += top_left[0]
    indices[[1], :] += top_left[1]
    return indices,width,height,top_left,bottom_right
def shortcut2(new_point):
    indices = np.where((new_point[0, :] >= 0) & (new_point[0, :] < 1024) &
                       (new_point[1, :] >= 0) & (new_point[1, :] < 1024))
    x = new_point[0, :]
    y = new_point[1, :]
    x_1 = x[indices]
    y_1 = y[indices]
    return x_1,y_1
def drawn(x_1,y_1):
    for i, j in zip(x_1, y_1):
        img[int(i), int(j), :] = [0, 255, 0]
def transalate_img(Pt, t):
    M = np.array([[1, 0, t[0]],
                  [0, 1, t[1]]]).astype("float")
    result= np.dot(M,Pt)
    return result
def rotate_img(Pt, alpha,width, height,top_left, center=None):
    angle=alpha
    if center==None:
        center = [(height - 1) / 2.0, (width - 1) / 2.0]
        center[0],center[1]=center[0]+top_left[0],center[1]+top_left[1]
    M=cv2.getRotationMatrix2D(center=center,angle= angle,scale=1)
    rot= np.dot(M,Pt)
    return rot
def scale_img(Pt, s):
    print(Pt[:20])
    M= np.array([[s[0],0,0],
                 [0,s[1],0]])
    result= np.dot(M,Pt)
    print(result[:20])
    return result
def draw_rectangle(event, x, y ,flag, param):
    global ix, iy,img,drawing,stat,rec_corr
    if event == cv2.EVENT_LBUTTONDOWN:
        ix=x
        iy=y
        drawing=True
    # EVENT_LBUTTONUP, EVENT_MOUSEMOVE
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        cv2.rectangle(img, (ix, iy), (x, y), (0, 255, 255), -1)
        print('success')
        drawing = False
        stat=True
        rec_corr[0],rec_corr[1],rec_corr[2],rec_corr[3]=ix,iy,x,y
def get_grid(x,y,homo=False):
    m= np.indices((x,y)).reshape(2,-1)
    return np.vstack((m, np.ones(shape=m.shape[1]))) if homo else m
def reset():
    global img
    img[:] = 255, 255, 255
    cv2.namedWindow('result')
def set_coor(x_1,y_1):
    rec_corr[0], rec_corr[1], rec_corr[2], rec_corr[3] = int(y_1[0]), int(x_1[0]), int(y_1[-1]), int(x_1[-1])
def get_input():
    x = int(input('input factor x'))
    y = int(input('input factor y'))
    return [x,y]
while True:
    func=int(input('choose func'))
    if func==1:
        # create background
        reset()
        pass
    if func==2:
        print('go')
        cv2.namedWindow('result')
        cv2.setMouseCallback('result', draw_rectangle)
        while stat==False:
            cv2.imshow('result', img)
            if cv2.waitKey(10)==27:
                break
        cv2.setMouseCallback('result', lambda *args: None)
        stat=False
        print(rec_corr)
    if func==3:
        indices=shortcut1()[0]
        t=get_input()
        new_point= transalate_img(indices,t)
        #reset()
        x_1,y_1=shortcut2(new_point)
        # reset()
        set_coor(x_1,y_1)
        drawn(x_1,y_1)
    if func==4:
        alpha= int(input('type alpha'))
        indices,width,height,top_left,bottom_right=shortcut1()
        new_point= rotate_img(indices,alpha=alpha,width=width,height=height,top_left=top_left)
        x_1,y_1=shortcut2(new_point)
        set_coor(x_1, y_1)
        #reset()
        drawn(x_1,y_1)
    if func == 5:
        # scale image:
        t=get_input()
        indices=shortcut1()[0]
        new_point = scale_img(indices,t)
        x_1, y_1=shortcut2(new_point)
        # reset()
        set_coor(x_1, y_1)
        drawn(x_1,y_1)
    cv2.imshow('result', img)
    if cv2.waitKey(10) == 27:
        break
    if func==6:
        break
cv2.destroyAllWindows()