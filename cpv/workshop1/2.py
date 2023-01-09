import numpy as np
import cv2
import math
ix,iy = -1,-1
drawing=False
stat=False
rec_corr=[ 0 for i in range(4)]
cv2.namedWindow('result')
img = np.zeros((1024, 1024, 3), np.uint8)
img[:] = 255, 255, 255
width = 0
height = 0
old_set=None

angle=[]
def shortcut1(use_old=False):
    global rec_corr, width, height, old_set
    top_left = (rec_corr[1], rec_corr[0])
    bottom_right = (rec_corr[3], rec_corr[2])
    # width = bottom_right[1] - top_left[1]
    # height = bottom_right[0] - top_left[0]
    indices = get_grid(height, width, True)
    indices[[0], :] += top_left[0]
    indices[[1], :] += top_left[1]
    # if use_old != False:
    #     return old_set,width,height,top_left,bottom_right
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
def rotate_img(Pt, alpha,center= None):
    y1,x1,y2,x2=rec_corr[0], rec_corr[1], rec_corr[2], rec_corr[3]
    if center==None:
        center=[0,0]
        center[0], center[1]=(x1 + x2) / 2, (y1 + y2) / 2
    M=cv2.getRotationMatrix2D(center=center,angle= alpha,scale=1)
    rot= np.dot(M,Pt)
    return rot
def scale_img(Pt, s):
    M= np.array([[s[0],0,0],
                 [0,s[1],0]])
    result= np.dot(M,Pt)
    return result
def draw_rectangle(event, x, y ,flag, param):
    global ix, iy,img,drawing,stat,rec_corr,width,height
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
        #rec_corr[0], rec_corr[1], rec_corr[2], rec_corr[3] = iy, ix, y, x
        width = max(rec_corr[2],rec_corr[0]) - min(rec_corr[2],rec_corr[0])
        height = max(rec_corr[3],rec_corr[1]) - min(rec_corr[3],rec_corr[1])
def get_grid(x,y,homo=False):
    m= np.indices((x,y)).reshape(2,-1)
    return np.vstack((m, np.ones(shape=m.shape[1]))) if homo else m
def reset():
    global img
    img[:] = 255, 255, 255
def set_coor(x_1,y_1):
    global rec_corr
    rec_corr[0], rec_corr[1], rec_corr[2], rec_corr[3] = int(y_1[0]), int(x_1[0]), int(y_1[-1]), int(x_1[-1])
    print(rec_corr)
def get_input():
    x = int(input('input factor x'))
    y = int(input('input factor y'))
    return [x,y]
def nothing(x):
    global s,z, old_set
    s=x
    if type(old_set) == np.ndarray:
        indices = old_set
    else:
        indices = shortcut1()[0]
    new_point = transalate_img(indices, [s, z])
    reset()
    x_1, y_1 = shortcut2(new_point)
    # reset()
    # set_coor(x_1,y_1)
    drawn(x_1, y_1)
    cv2.imshow('result', img)
    if cv2.waitKey(10) == 27:
        pass
def nothing1(x):
    global s, z, old_set
    z = x
    if type(old_set) == np.ndarray:
        indices = old_set
    else:
        indices = shortcut1()[0]
    new_point = transalate_img(indices, [s, z])
    reset()
    x_1, y_1 = shortcut2(new_point)
    # reset()
    # set_coor(x_1,y_1)
    drawn(x_1, y_1)
    cv2.imshow('result', img)
    if cv2.waitKey(10) == 27:
        pass
def nothing2(x):
    global width, height, old_set,angle
    angle.append(x)
    if type(old_set) != np.ndarray:
        indices, width, height, top_left, bottom_right = shortcut1()
    else:
        print('fail')
        indices = old_set
    print(x)
    new_point = rotate_img(indices, alpha=x)
    x_1, y_1 = shortcut2(new_point)
    reset()
    drawn(x_1, y_1)
    new_point = np.vstack((new_point, np.ones(shape=new_point.shape[1])))
    old_set = new_point
    cv2.imshow('result', img)
    if cv2.waitKey(10) == 27:
        pass

cv2.createTrackbar("translate_x", "result", 0, 500, nothing)
cv2.createTrackbar("translate_y", "result", 0, 500, nothing1)
cv2.createTrackbar('roate_para', 'result',  0,180,nothing2)
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
        s,z=0,0
        while True:
            s = cv2.getTrackbarPos("translate_x", "result")
            z = cv2.getTrackbarPos("translate_y", "result")
            print(s,z)
            if type(old_set) == np.ndarray:
                indices=old_set
            else:
                indices = shortcut1()[0]
            new_point= transalate_img(indices,[s,z])
            reset()
            x_1,y_1=shortcut2(new_point)
            drawn(x_1,y_1)
            cv2.imshow('result', img)
            if cv2.waitKey(10) == 27:
                break
        new_point=np.vstack((new_point, np.ones(shape=new_point.shape[1])))
        old_set=new_point
        cv2.setTrackbarPos('translate_x', 'reslut', 0)
        cv2.setTrackbarPos('translate_y', 'reslut', 0)
        set_coor(x_1,y_1)

    if func==4:
        while True:
            alpha=cv2.getTrackbarPos("rotate_para", "result")
            if type(old_set) != np.ndarray:
                indices,width,height,top_left,bottom_right=shortcut1()
            else:
                print('fail')
                indices= old_set
            new_point= rotate_img(indices,alpha=alpha)
            x_1,y_1 = shortcut2(new_point)
            reset()
            drawn(x_1,y_1)
            cv2.imshow('result', img)
            if cv2.waitKey(10) == 27:
                break
            #set_coor(x_1, y_1)
        new_point = np.vstack((new_point, np.ones(shape=new_point.shape[1])))
        old_set = new_point
    if func == 5:
        # scale image:
        y1, x1, y2, x2 = rec_corr[0], rec_corr[1], rec_corr[2], rec_corr[3]
        center = [0, 0]
        center[0], center[1] = (x1 + x2) / 2, (y1 + y2) / 2
        t=get_input()
        if type(old_set) != np.ndarray:
            indices,width,height,top_left,bottom_right=shortcut1()
        else:
            print('fail')
            indices= old_set
        new_point = scale_img(indices,t)
        new_point[0,:]-= center[0]
        new_point[1,:]-= center[1]
        x_1, y_1=shortcut2(new_point)
        set_coor(x_1, y_1)
        height *= t[0]
        width *= t[1]
        new_point = shortcut1()[0]
        reset()
        if len(angle)!=0:
            new_point = rotate_img(new_point, alpha=angle[-1])
            x_1, y_1  = shortcut2(new_point)
            set_coor(x_1, y_1)
        x_1, y_1 = shortcut2(new_point)
        drawn(x_1,y_1)
        #new_point = np.vstack((new_point, np.ones(shape=new_point.shape[1])))
        old_set = new_point
    cv2.imshow('result', img)
    if cv2.waitKey(10) == 27:
        break
    if func==6:
        break
cv2.destroyAllWindows()