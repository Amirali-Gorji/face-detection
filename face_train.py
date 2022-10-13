import cv2 as cv
import os 
import numpy as np


DIR = r'D:\programming\Face detection\src\pictures\train'

persons = ['hajghasem']

haar_face_classifier = cv.CascadeClassifier('haar_face.xml')


features = []
labels = []

def create_train_data():
    for person in persons:
        path = os.path.join(DIR,person)
        label = person
        print (person)
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            
            #read img
            img_array = cv.imread(img_path)
            #convert it to gray picture
            img_gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)
            
            #get position of face in img
            face_rects = haar_face_classifier.detectMultiScale(img_gray , scaleFactor=2 , minNeighbors=3)
            print(face_rects)
            for (x,y,w,h) in face_rects:
                # crop face region of interest
                face_roi = img_gray[y:y+h , x:x+w]    
                features.append(face_roi)
                labels.append(label)


create_train_data()

# print (labels)
# print (len(features))