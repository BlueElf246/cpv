import os
import cv2
import numpy as np
os.chdir("/Users/datle/Documents/PycharmProjects 08.55.25/pythonProject/CPV")
def get_grid(x,y,homo=False):
    m= np.indices((x,y)).reshape(2,-1)
    return np.vstack((m, np.ones(shape=m.shape[1]))) if homo else m
def rotate_img(Pt, alpha, center=None):
    global roi, width, height
    Pt.append(1)
    Pt=np.array(Pt)
    Pt=Pt.reshape(3,1)
    angle=alpha
    cos=np.cos(angle)
    sin=np.sin(angle)
    if center==None:
        center = [(width - 1) / 2.0, (height - 1) / 2.0]
        center[0],center[1]=center[0]+200,center[1]+200
    M=np.array([[cos, sin, (1-cos)*center[0]-sin*center[1]],
                [-sin,cos, sin*center[0]+(1-cos)*center[1]]])
    # M=cv2.getRotationMatrix2D(center=center,angle= angle,scale=1)
    # rot = cv2.warpAffine(Pt, M, (width, height))
    rot= np.dot(M,Pt)
    return rot

def rotate_img1(Pt, alpha,width, height,top_left, center=None):
    angle=alpha
    cos=np.cos(angle)
    sin=np.sin(angle)
    if center==None:
        center = [(height - 1) / 2.0, (width - 1) / 2.0]
        center[0],center[1]=center[0]+top_left[0],center[1]+top_left[1]
    #M=np.array([[cos, sin, (1-cos)*center[0]-sin*center[1]],[-sin,cos, sin*center[0]+(1-cos)*center[1]]])
    M=cv2.getRotationMatrix2D(center=center,angle= angle,scale=1)
    # rot = cv2.warpAffine(Pt, M, (width, height))
    rot= np.dot(M,Pt)
    return rot
img=np.zeros((512,512,3),dtype=np.uint8)
img[:]=255,255,255
img[50:400,100:200,:]=[255,0,0]

top_left = [50,100]
bottom_right = [400,200]

# img[50,100,:]=[0,0,255]
# img[400,200,:]=[255,0,255]
width = 200-100
height = 400-50

indices= get_grid(height,width,True)
indices[[0],:]+=top_left[0]
indices[[1],:]+=top_left[1]


new_pt=rotate_img1(indices,90,width,height,top_left)
indices = np.where((new_pt[0,:] >= 0) & (new_pt[0,:] < 512) &
                   (new_pt[1,:] >= 0) & (new_pt[1,:] < 512))

x=new_pt[0,:]
y=new_pt[1,:]
x_1=x[indices]
y_1=y[indices]
#new_pt[[1],:]=new_pt[[1],:][indices]
# print(new_pt[0,:])
# print(new_pt.shape)
for i,j in zip(x_1,y_1):
    img[int(i), int(j), :] = [0, 255, 255]

cv2.imshow('img',img)
cv2.waitKey(0)