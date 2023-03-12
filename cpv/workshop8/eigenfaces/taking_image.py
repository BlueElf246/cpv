import cv2
import pickle
import os
import numpy as np
face_detector = cv2.CascadeClassifier('/Users/datle/Desktop/CPV/cpv/workshop7/haarcascade_frontalface_default.xml')
def taking_image(id=1):
    img_vector=[]
    count=0
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()
        if count < 10:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray,1.1, 4 )
            for (x,y, w, h) in faces:
                cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  3)
                face= frame[y+2:y+h-2, x+2:x+w-2]
                area= w*h
                if os.path.exists(f"dataset/id{id}") == False:
                    os.mkdir(f"dataset/id{id}")
                if area >11000:
                    cv2.imwrite(f'dataset/id{id}/{count}.jpeg',cv2.resize(face, (320,320)))
                # area= w*h
                # print(area)
                # isExist = os.path.exists(f"dataset/id{id}")
                # if isExist == False:
                #     os.mkdir(f"dataset/id{id}")
                # else:
                #     cv2.imwrite(f'dataset/id{id}/{count}.jpeg', face)

            count+=1
        else:
            return np.array(img_vector)
        cv2.imshow("window", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame.release()
face_matrix= taking_image(id=1)