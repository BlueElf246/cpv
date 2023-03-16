import numpy as np
import cv2
import math
ix,iy = -1,-1
drawing=False
stat=False
rec_corr=[ [0,0] for i in range(4)]
cv2.namedWindow('result')
img = np.zeros((1024, 1024, 3), np.uint8)
img[:] = 255, 255, 255
width = 0
height = 0
old_set=None
def get_input():
    x = float(input('input factor x'))
    y = float(input('input factor y'))
    return [x,y]
def shortcut1():
    global rec_corr
    x_tl = rec_corr[0][0]
    y_tl = rec_corr[0][1]
    x_br = rec_corr[1][0]
    y_br = rec_corr[1][1]
    x_bl = rec_corr[2][0]
    y_bl = rec_corr[2][1]
    x_tr = rec_corr[3][0]
    y_tr = rec_corr[3][1]
    Pt = np.array([[x_tl, x_br, x_bl, x_tr], [y_tl, y_br, y_bl, y_tr]], dtype=np.int64)
    Pt = Pt.reshape(2, 4)
    Pt = np.vstack((Pt, np.ones(shape=Pt.shape[1])))
    return Pt
def set_corr(result):
    global rec_corr
    rec_corr[0]=result[:,0]
    rec_corr[1]=result[:,1]
    rec_corr[2]=result[:,2]
    rec_corr[3]=result[:,3]
def draw():
    global rec_corr
    reset()
    points = np.array([rec_corr[0],rec_corr[2],
                       rec_corr[1],rec_corr[3]])
    points = points.astype(np.int64)
    cv2.fillPoly(img, pts=[points], color=(255, 0, 0))
def center1():
    global rec_corr
    x1, y1, x2, y2 = rec_corr[0][0], rec_corr[0][1], rec_corr[1][0], rec_corr[1][1]
    center = [0, 0]
    center[0], center[1] = (x1 + x2) / 2, (y1 + y2) / 2
    return center
def check(s,center,result):
    c1=center[0]
    c2=center[1]
    if s[0] >= 1 and s[1] >=1:
        result[0, :] -= c1
        result[1, :] -= c2
    elif s[0] >=1 and s[1] <1:
        result[0, :] -= c1
        result[1, :] += c2
    elif s[0] <1 and s[1] >=1:
        result[0, :] += c1
        result[1, :] -= c2
    elif s[0] < 1. and s[1] < 1.:
        result[0, :] += c1
        result[1, :] += c2
    return result
def transalate_img(t):
    Pt=shortcut1()
    M = np.array([[1, 0, t[0]],
                  [0, 1, t[1]]]).astype("int64")
    result= np.dot(M,Pt)
    set_corr(result)
def rotate_img(alpha,center= None):
    Pt = shortcut1()
    if center == None:
        center=center1()
    M=cv2.getRotationMatrix2D(center=center,angle=alpha,scale=1)
    rot= np.dot(M,Pt)
    set_corr(rot)
def scale_img(s):
    Pt = shortcut1()
    M= np.array([[s[0],0,0],
                 [0,s[1],0]],dtype=np.float32)
    center=center1()
    result= np.dot(M,Pt)
    set_corr(result)
    center_new=center1()
    center_new=np.abs(np.array(center_new)-np.array(center))
    result=check(s,center_new,result)
    result=result.astype(np.int32)
    set_corr(result)
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
        width = max(x, ix) - min(x, ix)
        height = max(iy, y) - min(iy, y)
        rec_corr[0]= [ix, iy] # top_left
        rec_corr[1]= [x,y]# bottom_right
        rec_corr[2]= [ix,iy+height] # bottom_left
        rec_corr[3]= [ix+width, iy]# top_right
        #cv2.putText(img,'top_left',(ix,iy),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
def reset():
    global img
    img[:] = 255, 255, 255
while True:
    print('1:tao hinh\t2:ve hinh\t3:translate\t4:rotate\t5:scale')
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
    if func ==3:
        #transalate image
        t= get_input()
        transalate_img(t)
        draw()
    if func ==4:
        #rotate image
        alpha=int(input('type alpha'))
        rotate_img(alpha)
        draw()
    if func ==5:
        # scale image
        t= get_input()
        scale_img(t)
        draw()
    if func ==6:
        break
    cv2.imshow('result', img)
    if cv2.waitKey(10) == 27:
        break
