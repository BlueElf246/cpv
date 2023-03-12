import cv2
import pickle
import os
import numpy as np
from train import predict_image
params=pickle.load(open("model.pkl", 'rb'))
database_encoded, mean_face, eigenFace, label, image_shape= params['database_id'], params['mean_face'], \
                                                            params['eigenfaces'], params['label'], params['image_shape']
face_detector = cv2.CascadeClassifier('/Users/datle/Desktop/CPV/cpv/workshop7/haarcascade_frontalface_default.xml')
def run():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray,1.1, 4 )
        for (x,y, w, h) in faces:
            cv2.rectangle(frame, pt1 = (x,y),pt2 = (x+w, y+h), color = (255,0,0),thickness =  3)
            face= frame[y+2:y+h-2, x+2:x+w-2]
            area= w*h
            if area >11000:
                p, score = predict_image(face, database_encoded, mean_face, eigenFace, label, image_shape)
                cv2.putText(frame, str(p), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        cv2.imshow("window", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frame.release()
run()